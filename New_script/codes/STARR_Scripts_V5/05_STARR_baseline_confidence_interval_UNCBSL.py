# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB | 05_STARR_baseline_confidence_interval_UNCBSL.py
STEP 05 — Raster-based Unadjusted Baseline + CI90 / UNCBSL
============================================================

Purpose
-------
This Step 05 no longer requires control_carbon_stock_change.csv/parquet.

It reads carbon-stock or carbon-change rasters for:
  1. locked donor/reference pixels selected by Step 03 / Step 04;
  2. project/activity-area pixels produced by Step 01;

then extracts raster values at those pixel coordinates and calculates:

  - donor/reference ΔC distribution;
  - unadjusted baseline mean ΔC in tC/ha/year;
  - unadjusted baseline total for the PA in tC/year and tC/period;
  - project observed ΔC from PA raster values;
  - 90% confidence interval and UNCBSL on the locked donor/reference pixels.

Critical method rules
---------------------
1. UNCBSL is calculated only from locked, twin-tested matched donor/reference pixels.
   The full donor pool is never used.

2. N_control counts all matched/twin-tested PA-control rows.
   Duplicate reference pixels are NOT dropped, because donor reuse is part of the
   accepted matched-pair design and must remain represented in the baseline sample.

3. PA/project values are extracted from the PA raster using project pixels from Step 01.
   If project_df is not supplied, this script reloads:
      STARR_outputs/<RUN_ID>/01_extract/project_pixels_raw.parquet|csv

4. Raster inputs can be either:
   A) direct annual ΔC raster in tC/ha/year, or
   B) stock raster(s) at T0 and monitoring year Y.

5. Stock rasters can be in:
   - tC/ha, no conversion;
   - Mg AGB/ha, converted to carbon with BIOMASS_TO_CARBON_FRACTION
     and optional ROOT_TO_SHOOT_RATIO.

Outputs
-------
  05_baseline_CI_UNCBSL/
    baseline_control_deltaC_distribution.parquet|csv
    project_deltaC_distribution.parquet|csv
    baseline_CI90_UNCBSL_summary.csv
    baseline_CI90_UNCBSL_report.json
    deltaC_control_CI90.png
    deltaC_project_vs_control.png

Notebook call
-------------
control_pixels, project_pixels, summary, report, figs, out05 = (
    s05.run_baseline_ci_uncbsl_from_rasters(...)
)
"""

import warnings
warnings.filterwarnings("ignore")

import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

try:
    import rasterio
except ImportError as exc:
    raise ImportError("rasterio is required. In Colab: pip install rasterio") from exc

try:
    from pyproj import Transformer
except ImportError as exc:
    raise ImportError("pyproj is required. In Colab: pip install pyproj") from exc


def _is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


if not _is_notebook():
    matplotlib.use("Agg", force=True)


# ================================================================
# USER PARAMETERS
# ================================================================

RUN_ID = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v09_raster_ext5km"
PROJECT_NAME = "Idiofa_Lobi"

BASE_DIR_CANDIDATES = [
    "/content/drive/MyDrive/STARR_Idiofa_New_V2",
    "/content/content/MyDrive/STARR_Idiofa_New_V2",
]

OUTPUT_FORMAT = "parquet"
STRICT_LOCK_REQUIRED = True

# If stock rasters are C_t0 and C_y, ΔC = (C_y - C_t0) / MONITORING_PERIOD_YEARS.
# If direct ΔC rasters are used, this value is only reported.
MONITORING_PERIOD_YEARS = 6.0

# Methodological monitoring guard.
# The cumulative unadjusted baseline at T0 is zero by definition.
# The first valid post-T0 endpoint is T0 + MIN_MONITORING_PERIOD_YEARS.
T0_YEAR = 2018
MONITORING_YEAR = 2024
MIN_MONITORING_PERIOD_YEARS = 4.0

# Conservative CI guard against spatial autocorrelation.
# Pixel-based CI is always reported. Block-based CI is also calculated when
# spatial coordinates can be resolved; the final CI is max(pixel CI, block CI).
USE_CONSERVATIVE_BLOCK_CI = True
SPATIAL_BLOCK_SIZE_M = 500.0

# Project/activity area in hectares.
# Keep this at 0.0 in the script; pass the real PA area from the notebook.
# If left at 0.0 and no positive project_area_ha argument is provided, Step 05 stops.
PROJECT_AREA_HA = 0.0

# 90% normal CI.
CI90_Z_VALUE = 1.645

# AGB -> C conversion only used when raster_spec["units"] == "AGB_Mg_ha".
# IPCC 2019 Refinement, Vol 4, Ch 4, Table 4.4.
# Tropical moist forest default R ≈ 0.24.  Set to 0.0 only if rasters
# already include belowground biomass or if BGB is excluded by design.
BIOMASS_TO_CARBON_FRACTION = 0.47
ROOT_TO_SHOOT_RATIO = 0.4

# PM REQUEST: calcolare l'unadjusted baseline ANCHE quando la ΔC media dei
# controlli è <= 0 (baseline negativo), invece di azzerarlo con max(x,0).
# La metodologia GS azzererebbe (nessun credito da baseline negativo), ma i PM
# vogliono vedere il valore reale. Con True, mean_creditable = mean_raw.
# NB: un baseline negativo NON e' creditabile in GS; resta una stima diagnostica.
ALLOW_NEGATIVE_BASELINE = True

COORD_ROUND_DECIMALS = 7
MIN_CONTROL_PIXELS = 2
RASTER_SAMPLE_CHUNK_SIZE = 100_000

# ----------------------------------------------------------------
# Raster specs.
#
# Use ONE of the following modes for each spec:
#
# MODE A — direct annual ΔC:
# {
#   "delta_raster": "/path/to/deltaC.tif",
#   "delta_band": 1,                    # optional; default 1
#   "delta_band_name": None,             # optional alternative to band index
#   "units": "tC_ha_yr"
# }
#
# MODE B — T0/Y stock in two rasters:
# {
#   "stock_t0_raster": "/path/to/C_2018.tif",
#   "stock_y_raster":  "/path/to/C_2024.tif",
#   "stock_t0_band": 1,
#   "stock_y_band": 1,
#   "stock_t0_band_name": None,
#   "stock_y_band_name": None,
#   "units": "tC_ha"                    # or "AGB_Mg_ha"
# }
#
# MODE C — T0/Y stock in a single multiband raster:
# {
#   "stock_raster": "/path/to/C_stack.tif",
#   "stock_t0_band_name": "C_2018",
#   "stock_y_band_name": "C_2024",
#   "units": "tC_ha"
# }
# ----------------------------------------------------------------

DONOR_RASTER_SPEC = {
    "delta_raster": None,
    "stock_raster": None,
    "stock_t0_raster": None,
    "stock_y_raster": None,
    "stock_t0_band": 1,
    "stock_y_band": 1,
    "stock_t0_band_name": None,
    "stock_y_band_name": None,
    "units": "tC_ha",
}

PROJECT_RASTER_SPEC = {
    "delta_raster": None,
    "stock_raster": None,
    "stock_t0_raster": None,
    "stock_y_raster": None,
    "stock_t0_band": 1,
    "stock_y_band": 1,
    "stock_t0_band_name": None,
    "stock_y_band_name": None,
    "units": "tC_ha",
}


# ================================================================
# IO HELPERS
# ================================================================

def load_df(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported table format: {p.suffix}")


def save_df(df, path_no_suffix, output_format=OUTPUT_FORMAT):
    p = Path(path_no_suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "parquet":
        out = p.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        out = p.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def _find_table(directory, stem, required=True):
    directory = Path(directory)
    for ext in ("parquet", "csv"):
        p = directory / f"{stem}.{ext}"
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(f"{stem}.parquet/csv not found in {directory}")
    return None


def detect_base_dirs():
    bases = [Path(p) for p in BASE_DIR_CANDIDATES if Path(p).exists()]
    if not bases:
        raise FileNotFoundError(f"No base directory found: {BASE_DIR_CANDIDATES}")
    root = bases[0] / "STARR_outputs" / RUN_ID
    return root, {
        "root": root,
        "01": root / "01_extract",
        "02": root / "02_matching",
        "03": root / "03_twin_test",
        "04": root / "04_reference_area",
        "05": root / "05_baseline_CI_UNCBSL",
    }


def build_dirs(base_dirs=None):
    if base_dirs is None:
        return detect_base_dirs()

    b0 = Path(base_dirs[0])

    # Accept either:
    # - root run dir: .../STARR_outputs/<RUN_ID>
    # - global base dir: .../STARR_Idiofa_New_V2
    if (b0 / "01_extract").exists() or (b0 / "03_twin_test").exists():
        root = b0
    else:
        root = b0 / "STARR_outputs" / RUN_ID

    return root, {
        "root": root,
        "01": root / "01_extract",
        "02": root / "02_matching",
        "03": root / "03_twin_test",
        "04": root / "04_reference_area",
        "05": root / "05_baseline_CI_UNCBSL",
    }


def load_json_if_exists(path):
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _as_path(v, key):
    if v is None or str(v).strip() == "":
        return None
    p = Path(v)
    if not p.exists():
        raise FileNotFoundError(f"Raster path for '{key}' not found: {p}")
    return p


# ================================================================
# COORDINATE HELPERS
# ================================================================

def standardize_ref_coords(df):
    out = df.copy()
    if {"ref_lon", "ref_lat"}.issubset(out.columns):
        lon_col, lat_col = "ref_lon", "ref_lat"
    elif {"lon", "lat"}.issubset(out.columns):
        lon_col, lat_col = "lon", "lat"
    else:
        raise ValueError("Missing reference coordinates. Required: ref_lon/ref_lat or lon/lat.")

    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out["_ref_lon_key"] = out[lon_col].round(COORD_ROUND_DECIMALS)
    out["_ref_lat_key"] = out[lat_col].round(COORD_ROUND_DECIMALS)
    out = out[out["_ref_lon_key"].notna() & out["_ref_lat_key"].notna()].copy()
    return out, lon_col, lat_col


def standardize_project_coords(df):
    out = df.copy()
    if {"proj_lon", "proj_lat"}.issubset(out.columns):
        lon_col, lat_col = "proj_lon", "proj_lat"
    elif {"lon", "lat"}.issubset(out.columns):
        lon_col, lat_col = "lon", "lat"
    else:
        raise ValueError("Missing project coordinates. Required: proj_lon/proj_lat or lon/lat.")

    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out["_proj_lon_key"] = out[lon_col].round(COORD_ROUND_DECIMALS)
    out["_proj_lat_key"] = out[lat_col].round(COORD_ROUND_DECIMALS)
    out = out[out["_proj_lon_key"].notna() & out["_proj_lat_key"].notna()].copy()
    return out, lon_col, lat_col


def prepare_control_rows(twin_df, use_all_matched_rows=True):
    """
    Prepare donor/reference rows for Step 05.

    Default behaviour is methodology-driven for the matched-pair design:
    use all locked/twin-tested matched rows exactly as selected by Step 03.
    Reference pixels are not deduplicated, because a reused donor pixel represents
    more than one accepted PA-control match and therefore must keep its matched-row
    weight in the baseline distribution.

    Set use_all_matched_rows=False only for diagnostic sensitivity checks.
    """
    out, lon_col, lat_col = standardize_ref_coords(twin_df)
    before = len(out)
    if not use_all_matched_rows:
        out = out.drop_duplicates(subset=["_ref_lon_key", "_ref_lat_key"]).reset_index(drop=True)
        selection_mode = "coordinate_deduplicated_diagnostic"
    else:
        out = out.reset_index(drop=True)
        selection_mode = "all_matched_rows_no_deduplication"
    return out, lon_col, lat_col, before, selection_mode


def prepare_project_rows(project_df, use_all_matched_rows=True):
    """Prepare PA/project rows. By default keeps all matched PA rows."""
    out, lon_col, lat_col = standardize_project_coords(project_df)
    before = len(out)
    if not use_all_matched_rows:
        out = out.drop_duplicates(subset=["_proj_lon_key", "_proj_lat_key"]).reset_index(drop=True)
        selection_mode = "coordinate_deduplicated_diagnostic"
    else:
        out = out.reset_index(drop=True)
        selection_mode = "all_matched_rows_no_deduplication"
    return out, lon_col, lat_col, before, selection_mode


# Backward-compatible aliases. They no longer deduplicate by default.
def deduplicate_controls(twin_df):
    out, lon_col, lat_col, before, _ = prepare_control_rows(twin_df, use_all_matched_rows=True)
    return out, lon_col, lat_col, before


def deduplicate_project_pixels(project_df):
    out, lon_col, lat_col, before, _ = prepare_project_rows(project_df, use_all_matched_rows=True)
    return out, lon_col, lat_col, before


# ================================================================
# RASTER SAMPLING
# ================================================================

def _band_descriptions(src):
    desc = list(src.descriptions or [])
    return [d if d is not None else "" for d in desc]


def resolve_band(src, band=None, band_name=None):
    if band_name is not None and str(band_name).strip() != "":
        wanted = str(band_name).strip()
        desc = _band_descriptions(src)
        for i, d in enumerate(desc, start=1):
            if d == wanted:
                return i
        low = wanted.lower()
        for i, d in enumerate(desc, start=1):
            if d.lower() == low:
                return i
        raise ValueError(
            f"Band name '{wanted}' not found in {src.name}. "
            f"Available descriptions: {desc}"
        )

    if band is None:
        return 1
    band = int(band)
    if band < 1 or band > src.count:
        raise ValueError(f"Invalid band index {band} for {src.name}; raster has {src.count} bands.")
    return band


def raster_pixel_area_ha(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            return None
        # Meaningful only for projected CRS. For EPSG:4326 this is degrees².
        if src.crs.is_projected:
            return float(abs(src.transform.a * src.transform.e) / 10_000.0)
        return None


def sample_raster_values(points_df, lon_col, lat_col, raster_path,
                         band=None, band_name=None,
                         out_col="value",
                         chunk_size=RASTER_SAMPLE_CHUNK_SIZE):
    """
    Samples one raster band at WGS84 lon/lat point coordinates.
    Returns (values, sample_metadata).
    """
    p = _as_path(raster_path, out_col)

    lons = pd.to_numeric(points_df[lon_col], errors="coerce").to_numpy(dtype=np.float64)
    lats = pd.to_numeric(points_df[lat_col], errors="coerce").to_numpy(dtype=np.float64)

    if np.isnan(lons).any() or np.isnan(lats).any():
        raise ValueError(f"NaN coordinates found for sampling {out_col}.")

    with rasterio.open(p) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {p}")

        band_idx = resolve_band(src, band=band, band_name=band_name)

        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)

        values = np.full(len(points_df), np.nan, dtype=np.float64)

        for start in range(0, len(points_df), int(chunk_size)):
            end = min(start + int(chunk_size), len(points_df))
            coords = list(zip(xs[start:end], ys[start:end]))
            vals = []
            for sample in src.sample(coords, indexes=band_idx, masked=True):
                if np.ma.is_masked(sample):
                    vals.append(np.nan)
                else:
                    v = float(np.asarray(sample).reshape(-1)[0])
                    if src.nodata is not None and np.isclose(v, float(src.nodata)):
                        vals.append(np.nan)
                    elif not np.isfinite(v):
                        vals.append(np.nan)
                    else:
                        vals.append(v)
            values[start:end] = vals

        meta = {
            "path": str(p),
            "band_index": int(band_idx),
            "band_name": band_name,
            "raster_crs": str(src.crs),
            "raster_width": int(src.width),
            "raster_height": int(src.height),
            "raster_count": int(src.count),
            "raster_nodata": None if src.nodata is None else float(src.nodata),
            "pixel_area_ha": raster_pixel_area_ha(p),
        }

    return values, meta


def _conversion_factor(units):
    units = str(units or "tC_ha")
    if units == "tC_ha":
        return 1.0, "stock already in tC/ha"
    if units == "AGB_Mg_ha":
        # B6 fix: la conversione AGB→C include BGB via R (root:shoot).
        # Questo è corretto SOLO se il raster è AGB aboveground-only.
        # Se il raster include già le radici o è già C totale, R va messo a 0
        # (altrimenti gonfia del fattore (1+R), ~24%). Il default non-zero NON
        # deve passare inosservato: lo dichiariamo esplicitamente nel report.
        r = float(ROOT_TO_SHOOT_RATIO)
        cf = float(BIOMASS_TO_CARBON_FRACTION)
        conv = cf * (1.0 + r)
        bgb_note = (
            f"AGB→C: C = AGB × {cf} × (1+{r}). BGB INCLUSA via R={r}. "
            "VERIFICARE che il raster sia AGB aboveground-only: se include già "
            "le radici o è C totale, impostare ROOT_TO_SHOOT_RATIO=0 per evitare "
            f"di gonfiare il carbonio di ~{r*100:.0f}%."
        ) if r > 0 else (
            f"AGB→C: C = AGB × {cf}. BGB ESCLUSA (R=0). Corretto solo se il raster "
            "è già C totale o se BGB è esclusa per disegno."
        )
        return conv, {
            "source_units": "Mg AGB/ha",
            "target_units": "tC/ha",
            "biomass_to_carbon_fraction": cf,
            "root_to_shoot_ratio": r,
            "bgb_included": bool(r > 0),
            "combined_factor": float(conv),
            "bgb_decision_note": bgb_note,
        }
    raise ValueError("Unsupported raster units. Use 'tC_ha', 'AGB_Mg_ha', or direct delta units 'tC_ha_yr'.")


def sample_delta_from_raster_spec(points_df, lon_col, lat_col, raster_spec,
                                  prefix, monitoring_period_years=MONITORING_PERIOD_YEARS):
    """
    Adds:
      <prefix>_deltaC_tC_ha_yr
      optional sampled stock columns
    """
    spec = dict(raster_spec or {})
    out = points_df.copy()

    if monitoring_period_years <= 0:
        raise ValueError("MONITORING_PERIOD_YEARS must be > 0.")

    sample_meta = {
        "prefix": prefix,
        "mode": None,
        "monitoring_period_years": float(monitoring_period_years),
        "rasters": {},
    }

    # MODE A — direct annual delta raster
    delta_path = spec.get("delta_raster")
    if delta_path is not None and str(delta_path).strip() != "":
        vals, meta_delta = sample_raster_values(
            out, lon_col, lat_col,
            raster_path=delta_path,
            band=spec.get("delta_band", 1),
            band_name=spec.get("delta_band_name"),
            out_col=f"{prefix}_deltaC_tC_ha_yr",
        )
        out[f"{prefix}_deltaC_tC_ha_yr"] = vals.astype(np.float64)
        sample_meta["mode"] = "direct_delta_raster"
        sample_meta["delta_units"] = "tC/ha/year"
        sample_meta["rasters"]["delta"] = meta_delta
        return out, sample_meta

    # MODE B/C — stock rasters
    stock_stack = spec.get("stock_raster")
    if stock_stack is not None and str(stock_stack).strip() != "":
        t0_path = stock_stack
        y_path = stock_stack
    else:
        t0_path = spec.get("stock_t0_raster")
        y_path = spec.get("stock_y_raster")

    if t0_path is None or str(t0_path).strip() == "" or y_path is None or str(y_path).strip() == "":
        raise ValueError(
            f"{prefix}: raster_spec incomplete. Provide either delta_raster, "
            "stock_raster with T0/Y bands, or stock_t0_raster + stock_y_raster."
        )

    units = spec.get("units", "tC_ha")
    conv, conv_meta = _conversion_factor(units)

    t0_vals, meta_t0 = sample_raster_values(
        out, lon_col, lat_col,
        raster_path=t0_path,
        band=spec.get("stock_t0_band", 1),
        band_name=spec.get("stock_t0_band_name"),
        out_col=f"{prefix}_stock_t0_raw",
    )
    y_vals, meta_y = sample_raster_values(
        out, lon_col, lat_col,
        raster_path=y_path,
        band=spec.get("stock_y_band", 1),
        band_name=spec.get("stock_y_band_name"),
        out_col=f"{prefix}_stock_y_raw",
    )

    out[f"{prefix}_stock_t0_raw"] = t0_vals.astype(np.float64)
    out[f"{prefix}_stock_y_raw"] = y_vals.astype(np.float64)
    out[f"{prefix}_C_t0_tC_ha"] = out[f"{prefix}_stock_t0_raw"] * float(conv)
    out[f"{prefix}_C_y_tC_ha"] = out[f"{prefix}_stock_y_raw"] * float(conv)
    out[f"{prefix}_deltaC_tC_ha_yr"] = (
        out[f"{prefix}_C_y_tC_ha"] - out[f"{prefix}_C_t0_tC_ha"]
    ) / float(monitoring_period_years)

    sample_meta["mode"] = "stock_difference_raster"
    sample_meta["stock_units_raw"] = units
    sample_meta["delta_units"] = "tC/ha/year"
    sample_meta["conversion"] = conv_meta
    sample_meta["rasters"]["stock_t0"] = meta_t0
    sample_meta["rasters"]["stock_y"] = meta_y
    return out, sample_meta


# ================================================================
# METHODOLOGY GUARDS / SPATIAL CI HELPERS
# ================================================================

def validate_monitoring_period(t0_year=None, monitoring_year=None,
                               monitoring_period_years=None,
                               min_monitoring_period_years=MIN_MONITORING_PERIOD_YEARS):
    """
    Enforces the post-T0 monitoring rule.

    At T0 the cumulative baseline is zero. A stock-difference endpoint is valid
    only if the monitoring period is >= the minimum required number of years.
    """
    t0 = T0_YEAR if t0_year is None else int(t0_year)

    if monitoring_period_years is None:
        my = MONITORING_YEAR if monitoring_year is None else int(monitoring_year)
        period = float(my - t0)
    else:
        period = float(monitoring_period_years)
        my = int(round(t0 + period)) if monitoring_year is None else int(monitoring_year)

    if my <= t0:
        raise ValueError(f"MONITORING_YEAR must be > T0_YEAR. Got T0={t0}, monitoring={my}.")

    computed = float(my - t0)
    explicit_years = (t0_year is not None) or (monitoring_year is not None)
    if explicit_years and monitoring_period_years is not None and abs(period - computed) > 1e-6:
        raise ValueError(
            f"MONITORING_PERIOD_YEARS ({period}) does not match MONITORING_YEAR - T0_YEAR ({computed}). "
            "Fix T0_YEAR, MONITORING_YEAR, or MONITORING_PERIOD_YEARS."
        )

    if period < float(min_monitoring_period_years):
        raise ValueError(
            f"Invalid monitoring period: {period:g} years. Minimum required after T0 is "
            f"{float(min_monitoring_period_years):g} years. First valid endpoint is "
            f"{t0 + int(min_monitoring_period_years)}."
        )

    return {
        "t0_year": int(t0),
        "monitoring_year": int(my),
        "monitoring_period_years": float(period),
        "minimum_monitoring_period_years": float(min_monitoring_period_years),
        "first_valid_monitoring_year": int(t0 + int(min_monitoring_period_years)),
        "t0_cumulative_unadjusted_baseline_tC": 0.0,
        "t0_ci90_abs_tC": 0.0,
        "t0_uncbsl_status": "not_applicable_at_T0_by_definition",
    }


def _first_raster_crs(sample_meta):
    for item in (sample_meta.get("rasters") or {}).values():
        crs = item.get("raster_crs")
        if crs:
            return str(crs)
    return None


def add_spatial_blocks(df, lon_col, lat_col, sample_meta,
                       block_size_m=SPATIAL_BLOCK_SIZE_M,
                       block_col="_ci_block_id"):
    """
    Adds a spatial block id used for conservative block-based CI.

    Priority:
      1. existing projected coordinates in the pixel table;
      2. transform lon/lat to the sampled raster CRS if projected;
      3. fallback to EPSG:3857 if the raster CRS is geographic or unavailable.
    """
    out = df.copy()
    if len(out) == 0:
        out[block_col] = []
        return out, {"block_status": "empty"}

    coord_candidates = [
        ("ref_x_utm", "ref_y_utm"),
        ("donor_x_utm", "donor_y_utm"),
        ("x_utm", "y_utm"),
        ("cell_xmin", "cell_ymin"),
    ]

    xs = ys = None
    source = None
    for xc, yc in coord_candidates:
        if xc in out.columns and yc in out.columns:
            xv = pd.to_numeric(out[xc], errors="coerce").to_numpy(dtype=np.float64)
            yv = pd.to_numeric(out[yc], errors="coerce").to_numpy(dtype=np.float64)
            if np.isfinite(xv).any() and np.isfinite(yv).any():
                xs, ys = xv, yv
                source = f"columns:{xc},{yc}"
                break

    target_crs = None
    if xs is None or ys is None:
        raster_crs = _first_raster_crs(sample_meta)
        try:
            crs_obj = rasterio.crs.CRS.from_string(raster_crs) if raster_crs else None
            if crs_obj is not None and crs_obj.is_projected:
                target_crs = raster_crs
            else:
                target_crs = "EPSG:3857"
        except Exception:
            target_crs = "EPSG:3857"

        lons = pd.to_numeric(out[lon_col], errors="coerce").to_numpy(dtype=np.float64)
        lats = pd.to_numeric(out[lat_col], errors="coerce").to_numpy(dtype=np.float64)
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        source = f"transformed_lonlat_to:{target_crs}"

    block_size = float(block_size_m)
    if block_size <= 0:
        raise ValueError("SPATIAL_BLOCK_SIZE_M must be > 0.")

    gx = np.floor(xs / block_size)
    gy = np.floor(ys / block_size)
    valid = np.isfinite(gx) & np.isfinite(gy)

    out["_ci_block_x"] = gx
    out["_ci_block_y"] = gy
    block_ids = np.full(len(out), None, dtype=object)
    block_ids[valid] = [f"{int(x)}_{int(y)}" for x, y in zip(gx[valid], gy[valid])]
    out[block_col] = block_ids

    n_blocks = int(pd.Series(out.loc[valid, block_col]).nunique()) if valid.any() else 0
    return out, {
        "block_status": "computed" if n_blocks >= 2 else "insufficient_blocks",
        "block_size_m": float(block_size),
        "block_coordinate_source": source,
        "block_target_crs": target_crs,
        "n_spatial_blocks": n_blocks,
    }


def calculate_block_ci(df, delta_col="ref_deltaC_tC_ha_yr", block_col="_ci_block_id"):
    if block_col not in df.columns:
        return {
            "block_ci_status": "not_computed_missing_block_id",
            "n_control_spatial_blocks": 0,
            "std_block_mean_deltaC_tC_ha_yr": None,
            "se_block_tC_ha_yr": None,
            "ci90_abs_block_tC_ha_yr": None,
        }

    work = df[[block_col, delta_col]].copy()
    work[delta_col] = pd.to_numeric(work[delta_col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[block_col, delta_col])

    block_means = work.groupby(block_col, sort=False)[delta_col].mean().dropna()
    n_blocks = int(len(block_means))
    if n_blocks < 2:
        return {
            "block_ci_status": "not_computed_insufficient_blocks",
            "n_control_spatial_blocks": n_blocks,
            "std_block_mean_deltaC_tC_ha_yr": None,
            "se_block_tC_ha_yr": None,
            "ci90_abs_block_tC_ha_yr": None,
        }

    sigma_block = float(block_means.std(ddof=1))
    se_block = float(sigma_block / math.sqrt(n_blocks))
    ci90_abs_block = float(CI90_Z_VALUE * se_block)

    return {
        "block_ci_status": "computed",
        "n_control_spatial_blocks": n_blocks,
        "std_block_mean_deltaC_tC_ha_yr": sigma_block,
        "se_block_tC_ha_yr": se_block,
        "ci90_abs_block_tC_ha_yr": ci90_abs_block,
    }


# ================================================================
# AREA / SUMMARY
# ================================================================

def infer_project_pixel_area_ha(project_pixels, project_sample_meta):
    """
    Diagnostic fallback only: infer row pixel area from Step 01/raster metadata.

    This does NOT necessarily represent the full PA if Step 01 used PA sampling or
    if Step 05 is using matched rows. For final totals, use resolve_project_area_ha().
    """
    if "pixel_area_ha" in project_pixels.columns:
        area_vals = pd.to_numeric(project_pixels["pixel_area_ha"], errors="coerce")
        area_vals = area_vals.replace([np.inf, -np.inf], np.nan)
        if area_vals.notna().any() and float(area_vals.dropna().sum()) > 0:
            return area_vals.astype(float).to_numpy(), {
                "area_source": "project_pixels.pixel_area_ha_diagnostic_only",
                "pixel_area_ha_constant": None,
            }

    pixel_area = None
    for item in (project_sample_meta.get("rasters") or {}).values():
        if item.get("pixel_area_ha") is not None:
            pixel_area = float(item["pixel_area_ha"])
            break

    if pixel_area is None:
        raise ValueError(
            "Cannot infer diagnostic project pixel area. Add pixel_area_ha to project_pixels_raw "
            "or use projected rasters with metric pixel size."
        )

    return np.full(len(project_pixels), pixel_area, dtype=np.float64), {
        "area_source": "project_raster_resolution_diagnostic_only",
        "pixel_area_ha_constant": float(pixel_area),
    }


def resolve_project_area_ha(project_area_ha=None, project_pixels=None, project_sample_meta=None,
                            allow_pixel_area_fallback=False):
    """
    Resolve the FULL Project/Activity Area used to scale mean ΔC to total tC.

    The script-level PROJECT_AREA_HA is intentionally 0.0. The real area must be
    passed from the notebook or another upstream source. This prevents silently
    using only the area of sampled/matched rows.
    """
    candidates = []
    if project_area_ha is not None:
        candidates.append(("argument:project_area_ha", project_area_ha))
    candidates.append(("script_global:PROJECT_AREA_HA", PROJECT_AREA_HA))

    for source, value in candidates:
        try:
            v = float(value)
        except Exception:
            continue
        if np.isfinite(v) and v > 0:
            return v, {
                "area_source": source,
                "project_area_ha": v,
                "area_is_full_project_area": True,
                "area_note": "Full PA/activity area supplied externally; not inferred from matched-row count.",
            }

    if allow_pixel_area_fallback and project_pixels is not None:
        area_vals, meta = infer_project_pixel_area_ha(project_pixels, project_sample_meta or {})
        v = float(np.nansum(area_vals))
        if np.isfinite(v) and v > 0:
            meta.update({
                "project_area_ha": v,
                "area_is_full_project_area": False,
                "area_note": (
                    "Fallback area inferred from available rows. Use only for diagnostics; "
                    "set project_area_ha for final methodology outputs."
                ),
            })
            return v, meta

    raise ValueError(
        "PROJECT_AREA_HA is 0 or missing. Pass the full PA/activity area from the notebook, e.g.\n"
        "    PROJECT_AREA_HA = <value_from_initial_project_data>\n"
        "    s05.run_baseline_ci_uncbsl_from_rasters(..., project_area_ha=PROJECT_AREA_HA)\n"
        "Do not let Step 05 infer the PA area from matched/sample rows."
    )


def calculate_control_ci(control_pixels, lon_col=None, lat_col=None, donor_sample_meta=None,
                         use_conservative_block_ci=USE_CONSERVATIVE_BLOCK_CI,
                         spatial_block_size_m=SPATIAL_BLOCK_SIZE_M):
    df = control_pixels.copy()
    df["ref_deltaC_tC_ha_yr"] = pd.to_numeric(df["ref_deltaC_tC_ha_yr"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["ref_deltaC_tC_ha_yr"].notna()].copy()

    if len(df) < MIN_CONTROL_PIXELS:
        raise RuntimeError(
            f"At least {MIN_CONTROL_PIXELS} valid matched control rows are required. Found {len(df)}."
        )

    delta = df["ref_deltaC_tC_ha_yr"].astype(float).to_numpy()
    n_control = int(len(delta))
    n_unique_ref = int(df[["_ref_lon_key", "_ref_lat_key"]].drop_duplicates().shape[0]) \
        if {"_ref_lon_key", "_ref_lat_key"}.issubset(df.columns) else None

    # MEAN: su tutte le righe matchate (donor riusati inclusi) — coerente col design
    # matched-pair: ogni match rappresenta un pixel PA. Questo è corretto e voluto.
    mean_delta = float(np.mean(delta))
    median_delta = float(np.median(delta))
    sigma_control = float(np.std(delta, ddof=1))

    # A1 fix — PSEUDO-REPLICAZIONE NELLO STIMATORE DI VARIANZA:
    # SE = σ/√N usa N = numero di OSSERVAZIONI INDIPENDENTI, non il numero di
    # righe matchate. I donor riusati e i pixel a 30 m spazialmente adiacenti NON
    # sono indipendenti. Usare N=n_control (righe) gonfia N → SE sottostimato →
    # CI troppo stretto → baseline meno conservativo (opposto di quanto GS richiede).
    # Conteggio conservativo: N_effettivo = pixel di riferimento UNICI.
    n_eff = int(n_unique_ref) if (n_unique_ref and n_unique_ref >= MIN_CONTROL_PIXELS) else n_control
    se_control = float(sigma_control / math.sqrt(n_eff))
    ci90_abs_pixel = float(CI90_Z_VALUE * se_control)
    # SE/CI ingenuo (N=tutte le righe) tenuto solo come diagnostico.
    se_control_naive = float(sigma_control / math.sqrt(n_control))
    ci90_abs_pixel_naive = float(CI90_Z_VALUE * se_control_naive)

    block_meta = {
        "block_status": "not_requested",
        "block_size_m": None,
        "block_coordinate_source": None,
        "block_target_crs": None,
        "n_spatial_blocks": 0,
    }
    block_ci = {
        "block_ci_status": "not_requested",
        "n_control_spatial_blocks": 0,
        "std_block_mean_deltaC_tC_ha_yr": None,
        "se_block_tC_ha_yr": None,
        "ci90_abs_block_tC_ha_yr": None,
    }

    if use_conservative_block_ci:
        if lon_col is None or lat_col is None:
            block_ci["block_ci_status"] = "not_computed_missing_lon_lat_columns"
        else:
            df, block_meta = add_spatial_blocks(
                df, lon_col=lon_col, lat_col=lat_col,
                sample_meta=donor_sample_meta or {},
                block_size_m=spatial_block_size_m,
                block_col="_ci_block_id",
            )
            block_ci = calculate_block_ci(df, "ref_deltaC_tC_ha_yr", "_ci_block_id")

    ci_candidates = [ci90_abs_pixel]
    if block_ci.get("ci90_abs_block_tC_ha_yr") is not None:
        ci_candidates.append(float(block_ci["ci90_abs_block_tC_ha_yr"]))
    ci90_abs_final = float(np.nanmax(ci_candidates))
    ci90_source = "block_conservative" if ci90_abs_final > ci90_abs_pixel else "pixel_based"

    ci90_lower_pixel = float(mean_delta - ci90_abs_pixel)
    ci90_upper_pixel = float(mean_delta + ci90_abs_pixel)
    ci90_lower_final = float(mean_delta - ci90_abs_final)
    ci90_upper_final = float(mean_delta + ci90_abs_final)

    if mean_delta > 0:
        uncbsl_fraction = float(ci90_abs_final / mean_delta)
        uncbsl_percent = float(uncbsl_fraction * 100.0)
        uncbsl_status = "computed_from_final_conservative_CI"
        crediting_note = "UNCBSL uses the final conservative CI90 over the positive mean baseline-removal rate."
    else:
        uncbsl_fraction = None
        uncbsl_percent = None
        uncbsl_status = "not_applicable_mean_deltaC_zero_or_negative"
        crediting_note = (
            "Mean control ΔC is <= 0. UNCBSL denominator is invalid for baseline-removal "
            "crediting; creditable unadjusted baseline removals are set to zero."
        )

    return df, {
        "n_control_matched_valid_rows": n_control,
        "n_control_unique_valid_pixels": n_unique_ref,
        "n_control_unique_reference_pixels_represented": n_unique_ref,
        "n_control_effective_for_se": int(n_eff),
        "mean_deltaC_control_tC_ha_yr": mean_delta,
        "median_deltaC_control_tC_ha_yr": median_delta,
        "std_deltaC_control_tC_ha_yr": sigma_control,
        "se_control_tC_ha_yr": se_control,
        "se_control_naive_all_rows_tC_ha_yr": se_control_naive,
        "ci90_abs_pixel_naive_all_rows_tC_ha_yr": ci90_abs_pixel_naive,
        "se_effective_n_note": (
            "SE pixel-based usa N_effettivo = pixel di riferimento UNICI (anti "
            "pseudo-replicazione). La MEDIA usa tutte le righe matchate (donor "
            "riusati inclusi). Il valore *_naive_* usa N=tutte le righe ed è solo "
            "diagnostico (sottostima l'incertezza)."
        ),
        "ci90_z_value": float(CI90_Z_VALUE),
        "ci90_abs_pixel_tC_ha_yr": ci90_abs_pixel,
        "ci90_lower_pixel_tC_ha_yr": ci90_lower_pixel,
        "ci90_upper_pixel_tC_ha_yr": ci90_upper_pixel,
        "ci90_abs_tC_ha_yr": ci90_abs_final,
        "ci90_abs_final_tC_ha_yr": ci90_abs_final,
        "ci90_lower_tC_ha_yr": ci90_lower_final,
        "ci90_upper_tC_ha_yr": ci90_upper_final,
        "ci90_lower_final_tC_ha_yr": ci90_lower_final,
        "ci90_upper_final_tC_ha_yr": ci90_upper_final,
        "ci90_final_source": ci90_source,
        "use_conservative_block_ci": bool(use_conservative_block_ci),
        **block_meta,
        **block_ci,
        "uncbsl_fraction": uncbsl_fraction,
        "uncbsl_percent": uncbsl_percent,
        "uncbsl_status": uncbsl_status,
        "crediting_note": crediting_note,
    }


def calculate_project_summary(project_pixels, full_project_area_ha):
    """
    Calculate PA observed ΔC from matched PA rows and scale the matched-row mean
    to the full PA area supplied externally.
    """
    df = project_pixels.copy()
    df["proj_deltaC_tC_ha_yr"] = pd.to_numeric(df["proj_deltaC_tC_ha_yr"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["proj_deltaC_tC_ha_yr"].notna()].copy()

    if df.empty:
        raise RuntimeError("No valid project matched rows after raster sampling.")

    pa_ha = float(full_project_area_ha)
    if not np.isfinite(pa_ha) or pa_ha <= 0:
        raise ValueError("full_project_area_ha must be positive.")

    n_rows = int(len(df))
    n_unique_proj = int(df[["_proj_lon_key", "_proj_lat_key"]].drop_duplicates().shape[0]) \
        if {"_proj_lon_key", "_proj_lat_key"}.issubset(df.columns) else None

    # Equal matched-row weighting. The row area is only a scaling weight so that
    # sums are reported on the full PA area, not on the subset/sample area.
    scaled_row_area = pa_ha / n_rows
    df["project_pixel_area_ha"] = scaled_row_area

    mean_delta = float(df["proj_deltaC_tC_ha_yr"].astype(float).mean())
    total_delta_tC_yr = float(mean_delta * pa_ha)

    return df, {
        "n_project_matched_valid_rows": n_rows,
        "n_project_unique_valid_pixels": n_unique_proj,
        "n_project_unique_pixels_represented": n_unique_proj,
        "project_area_ha": pa_ha,
        "project_area_scaling_method": "matched_row_mean_scaled_to_full_project_area",
        "project_scaled_row_area_ha": float(scaled_row_area),
        "mean_deltaC_project_weighted_tC_ha_yr": mean_delta,
        "mean_deltaC_project_matched_tC_ha_yr": mean_delta,
        "total_deltaC_project_tC_yr": total_delta_tC_yr,
    }


def build_unadjusted_baseline_summary(control_summary, project_summary,
                                      monitoring_period_years=MONITORING_PERIOD_YEARS,
                                      temporal_meta=None,
                                      allow_negative_baseline=ALLOW_NEGATIVE_BASELINE):
    """
    Builds raw and creditable unadjusted baseline quantities.

    Raw baseline = mean control ΔC × PA area.
    Creditable baseline-removal rate = max(mean control ΔC, 0) × PA area,
    UNLESS allow_negative_baseline=True (PM request), in which case the
    creditable value equals the raw value even when negative.

    Negative control ΔC normally cannot generate baseline removals for crediting.
    At T0 the cumulative unadjusted baseline is zero.
    """
    mean_raw = float(control_summary["mean_deltaC_control_tC_ha_yr"])
    # PM REQUEST: niente clamp a zero se allow_negative_baseline=True.
    mean_creditable = float(mean_raw) if allow_negative_baseline else float(max(mean_raw, 0.0))
    pa_ha = float(project_summary["project_area_ha"])
    period = float(monitoring_period_years)

    ci_final = float(control_summary.get("ci90_abs_final_tC_ha_yr", control_summary.get("ci90_abs_tC_ha_yr", 0.0)))
    ci_total_yr = float(ci_final * pa_ha)
    ci_total_period = float(ci_total_yr * period)

    baseline_raw_total_yr = float(mean_raw * pa_ha)
    baseline_raw_total_period = float(baseline_raw_total_yr * period)

    baseline_creditable_total_yr = float(mean_creditable * pa_ha)
    baseline_creditable_total_period = float(baseline_creditable_total_yr * period)

    if mean_creditable > 0:
        baseline_unc_mean = float(mean_creditable + ci_final)
        baseline_unc_total_yr = float(baseline_creditable_total_yr + ci_total_yr)
        baseline_unc_total_period = float(baseline_creditable_total_period + ci_total_period)
        uncertainty_status = "computed_positive_baseline"
    elif allow_negative_baseline and mean_creditable != 0:
        # Baseline negativo mantenuto (PM): CI applicato simmetricamente.
        baseline_unc_mean = float(mean_creditable + ci_final)
        baseline_unc_total_yr = float(baseline_creditable_total_yr + ci_total_yr)
        baseline_unc_total_period = float(baseline_creditable_total_period + ci_total_period)
        uncertainty_status = "computed_negative_baseline_pm_override"
    else:
        baseline_unc_mean = 0.0
        baseline_unc_total_yr = 0.0
        baseline_unc_total_period = 0.0
        uncertainty_status = "zeroed_non_positive_baseline"

    project_total_yr = float(project_summary["total_deltaC_project_tC_yr"])
    project_total_period = float(project_total_yr * period)

    diff_creditable_yr = float(project_total_yr - baseline_creditable_total_yr)
    diff_creditable_period = float(project_total_period - baseline_creditable_total_period)
    diff_raw_yr = float(project_total_yr - baseline_raw_total_yr)
    diff_raw_period = float(project_total_period - baseline_raw_total_period)

    temporal_meta = temporal_meta or {}

    # ── PM-facing BL_unadj,y numeric quantities ───────────────────────
    # Formula (GS STARR Eq 31a): BL_unadj,y = ΔC_ref,y × A_project
    #   ΔC_ref,y  = mean_creditable  [tC/ha/yr, max(mean_control_deltaC, 0)]
    #   A_project = pa_ha            [ha]
    # These are absolute numbers, NOT fractions or percentages.
    _co2e = 44.0 / 12.0
    bl_unadj_y_tC          = baseline_creditable_total_yr          # tC/yr
    bl_unadj_y_tCO2e       = baseline_creditable_total_yr * _co2e  # tCO2e/yr
    bl_unadj_period_tC     = baseline_creditable_total_period       # tC/period
    bl_unadj_period_tCO2e  = baseline_creditable_total_period * _co2e  # tCO2e/period
    bl_unadj_raw_y_tCO2e   = baseline_raw_total_yr * _co2e         # tCO2e/yr (incl. negative)

    return {
        # ── Methodology-labelled primary outputs (BL_unadj,y = ΔC_ref,y × A_project) ──
        "BL_unadj_y_tC":              bl_unadj_y_tC,
        "BL_unadj_y_tCO2e":           bl_unadj_y_tCO2e,
        "BL_unadj_period_tC":         bl_unadj_period_tC,
        "BL_unadj_period_tCO2e":      bl_unadj_period_tCO2e,
        "BL_unadj_raw_y_tCO2e":       bl_unadj_raw_y_tCO2e,
        "BL_unadj_formula":           "BL_unadj_y = delta_C_ref_y_tC_ha_yr * project_area_ha",
        "delta_C_ref_y_tC_ha_yr":     mean_creditable,
        "project_area_ha_used":       pa_ha,
        # ── Internal / diagnostic fields ──────────────────────────────
        "unadjusted_baseline_mean_raw_tC_ha_yr": mean_raw,
        "unadjusted_baseline_total_raw_tC_yr": baseline_raw_total_yr,
        "unadjusted_baseline_total_raw_tC_period": baseline_raw_total_period,
        "unadjusted_baseline_mean_tC_ha_yr": mean_creditable,
        "unadjusted_baseline_total_tC_yr": baseline_creditable_total_yr,
        "unadjusted_baseline_total_tC_period": baseline_creditable_total_period,
        "unadjusted_baseline_negative_control_rule": (
            "PM override attivo: baseline = mean_control_deltaC (anche negativo)"
            if allow_negative_baseline
            else "creditable baseline removals use max(mean_control_deltaC, 0)"),
        "allow_negative_baseline": bool(allow_negative_baseline),
        "ci90_abs_final_total_tC_yr": ci_total_yr,
        "ci90_abs_final_total_tC_period": ci_total_period,
        "baseline_uncertainty_adjusted_mean_tC_ha_yr": baseline_unc_mean,
        "baseline_uncertainty_adjusted_total_tC_yr": baseline_unc_total_yr,
        "baseline_uncertainty_adjusted_total_tC_period": baseline_unc_total_period,
        "baseline_uncertainty_adjustment_status": uncertainty_status,
        "baseline_uncertainty_double_count_warning": (
            "ATTENZIONE doppio conteggio: baseline_uncertainty_adjusted = BL + CI = "
            "BL × (1 + CI/mean) = BL × (1 + UNCBSL). È matematicamente la STESSA "
            "incertezza già rappresentata dal fattore (1+UNCBSL) in Eq-6. NON sommare "
            "baseline_uncertainty_adjusted E applicare (1+UNCBSL): scegliere UNA via. "
            "Eq-6 usa BL_unadj × (1+DAF) × (1+UNCBSL) — quindi NON usare anche il "
            "baseline_uncertainty_adjusted in quel calcolo."
        ),
        "project_observed_total_tC_yr": project_total_yr,
        "project_observed_total_tC_period": project_total_period,
        "project_minus_unadjusted_baseline_tC_yr": diff_creditable_yr,
        "project_minus_unadjusted_baseline_tC_period": diff_creditable_period,
        "project_minus_unadjusted_baseline_raw_tC_yr": diff_raw_yr,
        "project_minus_unadjusted_baseline_raw_tC_period": diff_raw_period,
        "monitoring_period_years": period,
        "t0_year": temporal_meta.get("t0_year"),
        "monitoring_year": temporal_meta.get("monitoring_year"),
        "minimum_monitoring_period_years": temporal_meta.get("minimum_monitoring_period_years"),
        "first_valid_monitoring_year": temporal_meta.get("first_valid_monitoring_year"),
        "t0_cumulative_unadjusted_baseline_tC": 0.0,
        "t0_ci90_abs_tC": 0.0,
        "t0_uncbsl_status": "not_applicable_at_T0_by_definition",
        "note": (
            "This is the unadjusted baseline plus baseline CI/UNCBSL. It does not apply DAF, "
            "UNCAR, leakage, permanence/risk deductions, or final GSVER issuance rules."
        ),
    }


# ================================================================
# PM CREDITING HANDOFF
# ================================================================

_TC_TO_TCO2E = 44.0 / 12.0
_DAF_LUF_DEFAULT = 0.0125


def _build_pm_handoff(summary, monitoring_period_years):
    """
    Build a PM-ready handoff block with all values needed to complete
    Eq-6 (BR_crediting) and Eq-22 (nAR) of the GS STARR methodology.

    This function does NOT apply DAF or BR_gov — those are PM responsibilities.
    It provides pre-computed tCO2e conversions and explicit formulas.
    """
    period = float(monitoring_period_years)
    pa_ha = float(summary.get("project_area_ha", 0.0))

    # BL_unadj,y = ΔC_ref,y × A_project  (GS STARR Eq 31a) — absolute number, not a fraction.
    # Prefer new BL_unadj_* keys surfaced in build_unadjusted_baseline_summary; fall back to legacy.
    bl_mean_tC_ha_yr = float(summary.get("delta_C_ref_y_tC_ha_yr",
                              summary.get("unadjusted_baseline_mean_tC_ha_yr", 0.0)))
    bl_total_tC_period = float(summary.get("BL_unadj_period_tC",
                                summary.get("unadjusted_baseline_total_tC_period", 0.0)))

    # tCO2e conversions — read from summary if already computed, else convert locally
    bl_mean_tCO2e_ha_yr   = bl_mean_tC_ha_yr * _TC_TO_TCO2E
    bl_total_tCO2e_period = float(summary.get("BL_unadj_period_tCO2e",
                                               bl_total_tC_period * _TC_TO_TCO2E))

    # --- UNCBSL ---
    # B5: Eq-6 usa BL_unadj GREZZO (BL_unadj_period_tC sopra) × (1+UNCBSL).
    # NON usare summary['baseline_uncertainty_adjusted_*'] qui: quello è già
    # BL × (1+UNCBSL) e moltiplicarlo ancora per (1+UNCBSL) sarebbe doppio conteggio.
    uncbsl_frac = summary.get("uncbsl_fraction")
    uncbsl_pct = summary.get("uncbsl_percent")

    # --- CI90 in tCO2e ---
    ci90_final_tC_ha_yr = float(summary.get("ci90_abs_final_tC_ha_yr", 0.0))
    ci90_final_tCO2e_ha_yr = ci90_final_tC_ha_yr * _TC_TO_TCO2E

    # --- Project observed in tCO2e ---
    proj_total_tC_period = float(summary.get("project_observed_total_tC_period", 0.0))
    proj_total_tCO2e_period = proj_total_tC_period * _TC_TO_TCO2E

    # --- Illustrative BR_crediting (for PM reference, using DAF floor) ---
    # Eq-6: BR_crediting = MAX[ BR_unadj × (1 + DAF) × (1 + UNCBSL), BR_gov ]
    # BR_gov is project-specific and must be set by PM.
    daf_floor = _DAF_LUF_DEFAULT
    uncbsl_for_calc = float(uncbsl_frac) if uncbsl_frac is not None else 0.0
    br_crediting_illustrative_tCO2e = (
        bl_total_tCO2e_period * (1.0 + daf_floor) * (1.0 + uncbsl_for_calc)
    )

    return {
        "scope": (
            "Pre-computed values for Eq-6 and Eq-22 of GS STARR Track 1 SEMDB. "
            "DAF and BR_gov are PM responsibilities and are NOT applied here."
        ),
        "tC_to_tCO2e_factor": _TC_TO_TCO2E,
        "project_area_ha": pa_ha,
        "monitoring_period_years": period,

        # Unadjusted baseline (BL_unadj) — Eq-1
        "BL_unadj_mean_tC_ha_yr": bl_mean_tC_ha_yr,
        "BL_unadj_mean_tCO2e_ha_yr": bl_mean_tCO2e_ha_yr,
        "BL_unadj_total_tC_period": bl_total_tC_period,
        "BL_unadj_total_tCO2e_period": bl_total_tCO2e_period,

        # UNCBSL — Eq-33/34
        "UNCBSL_fraction": uncbsl_frac,
        "UNCBSL_percent": uncbsl_pct,
        "UNCBSL_status": summary.get("uncbsl_status"),

        # CI90
        "CI90_abs_final_tC_ha_yr": ci90_final_tC_ha_yr,
        "CI90_abs_final_tCO2e_ha_yr": ci90_final_tCO2e_ha_yr,
        "CI90_source": summary.get("ci90_final_source"),

        # Project observed
        "project_observed_total_tC_period": proj_total_tC_period,
        "project_observed_total_tCO2e_period": proj_total_tCO2e_period,

        # Illustrative Eq-6 (PM must verify DAF and set BR_gov)
        "illustrative_DAF_used": daf_floor,
        "illustrative_BR_crediting_tCO2e_period": br_crediting_illustrative_tCO2e,
        "illustrative_note": (
            f"BR_crediting = {bl_total_tCO2e_period:,.2f} × (1 + {daf_floor}) "
            f"× (1 + {uncbsl_for_calc:.6f}) = {br_crediting_illustrative_tCO2e:,.2f} tCO2e. "
            "PM must compare with BR_gov and take MAX per Eq-6."
        ),

        # Next steps for PM
        "pm_actions_required": [
            "1. Verify or update DAF (LUF floor = 1.25%; check host-country sectoral override)",
            "2. Determine BR_gov (NDC unconditional reforestation target, if applicable)",
            "3. Compute BR_crediting = MAX[ BL_unadj_tCO2e × (1+DAF) × (1+UNCBSL), BR_gov ]  (Eq-6)",
            "4. Compute AR_final (activity removals adjusted for UNCAR)  (Eq-14/16)",
            "5. Compute AE (activity emissions: burn, fertilizer, fuel, livestock)  (Eq-7)",
            "6. Compute LE_final (leakage: belt monitoring or yield test + 15% buffer)  (Eq-20/21)",
            "7. Compute nAR = (AR_final - AE) - (BR_crediting - BE) - LE_final  (Eq-22, BE=0)",
            "8. Apply buffer rate (10-20%) for GSVERs  (Eq-23)",
        ],
    }


# ================================================================
# PLOTS
# ================================================================

def plot_delta_ci(control_df, summary, out_dir):
    v = control_df["ref_deltaC_tC_ha_yr"].dropna().astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(v, bins=min(60, max(10, int(np.sqrt(len(v))))), alpha=0.85, edgecolor="black")
    ax.axvline(summary["mean_deltaC_control_tC_ha_yr"], linestyle="--", linewidth=1.5, label="mean control ΔC")
    ax.axvline(summary["ci90_lower_tC_ha_yr"], linestyle=":", linewidth=1.5, label="90% CI lower")
    ax.axvline(summary["ci90_upper_tC_ha_yr"], linestyle=":", linewidth=1.5, label="90% CI upper")
    ax.set_xlabel("Locked donor/reference pixel ΔC (tC/ha/year)")
    ax.set_ylabel("Pixel count")
    ax.set_title("GS STARR Track 1 SEMDB — control ΔC 90% confidence interval")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = Path(out_dir) / "deltaC_control_CI90.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig, out_path


def plot_project_vs_control(control_df, project_df, out_dir):
    c = control_df["ref_deltaC_tC_ha_yr"].dropna().astype(float)
    p = project_df["proj_deltaC_tC_ha_yr"].dropna().astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(80, max(15, int(np.sqrt(max(len(c), len(p))))))
    lo = float(np.nanpercentile(np.concatenate([c.values, p.values]), 1))
    hi = float(np.nanpercentile(np.concatenate([c.values, p.values]), 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = min(c.min(), p.min()), max(c.max(), p.max())
    edges = np.linspace(lo, hi, bins)
    ax.hist(c, bins=edges, alpha=0.55, density=True, edgecolor="black", label="Locked donor/reference")
    ax.hist(p, bins=edges, alpha=0.55, density=True, edgecolor="black", label="PA/project")
    ax.axvline(c.mean(), linestyle="--", linewidth=1.5, label="control mean")
    ax.axvline(np.average(project_df["proj_deltaC_tC_ha_yr"], weights=project_df["project_pixel_area_ha"]),
               linestyle=":", linewidth=1.5, label="project weighted mean")
    ax.set_xlabel("ΔC (tC/ha/year)")
    ax.set_ylabel("Density")
    ax.set_title("Raster-extracted ΔC — PA/project vs locked controls")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = Path(out_dir) / "deltaC_project_vs_control.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig, out_path


# ================================================================
# MAIN STEP
# ================================================================

def run_baseline_ci_uncbsl_from_rasters(
    base_dirs=None,
    output_dir=None,
    donor_raster_spec=None,
    project_raster_spec=None,
    twin_df=None,
    project_df=None,
    project_area_ha=None,
    use_all_matched_rows=True,
    use_matched_project_rows=True,
    manifest=None,
    monitoring_period_years=None,
    t0_year=None,
    monitoring_year=None,
    min_monitoring_period_years=MIN_MONITORING_PERIOD_YEARS,
    use_conservative_block_ci=USE_CONSERVATIVE_BLOCK_CI,
    spatial_block_size_m=SPATIAL_BLOCK_SIZE_M,
    allow_negative_baseline=ALLOW_NEGATIVE_BASELINE,
    verbose=True,
):
    """
    Raster-based Step 05.

    Parameters
    ----------
    base_dirs : list[path]
        Usually [OUTPUT_ROOT]. Used to reload Step 01/03/04 files if dataframes are not supplied.
    output_dir : path
        Output folder.
    donor_raster_spec : dict
        Raster spec for locked donor/reference pixels.
    project_raster_spec : dict
        Raster spec for PA/project pixels.
    twin_df : DataFrame | None
        Step 03 twin_tested_pixels. If None, reloads from 03_twin_test.
    project_df : DataFrame | None
        Optional PA/project dataframe. If None and use_matched_project_rows=True, Step 05 uses twin_df
        so that PA and control are evaluated on the same matched rows.
    project_area_ha : float | None
        Full Project/Activity Area in hectares. Required for final total baseline calculations.
    use_all_matched_rows : bool
        If True, no coordinate deduplication is applied to PA/control matched rows.
    use_matched_project_rows : bool
        If True and project_df is None, PA values are extracted from twin_df proj_lon/proj_lat.
    manifest : dict | None
        Step 04 manifest. If None, reloads from 04_reference_area.
    monitoring_period_years : float | None
        Overrides MONITORING_PERIOD_YEARS.
    """
    if monitoring_period_years is None:
        monitoring_period_years = MONITORING_PERIOD_YEARS
    monitoring_period_years = float(monitoring_period_years)

    temporal_meta = validate_monitoring_period(
        t0_year=t0_year,
        monitoring_year=monitoring_year,
        monitoring_period_years=monitoring_period_years,
        min_monitoring_period_years=min_monitoring_period_years,
    )

    root, dirs = build_dirs(base_dirs)
    out_dir = Path(output_dir) if output_dir else dirs["05"]
    out_dir.mkdir(parents=True, exist_ok=True)

    donor_spec = donor_raster_spec if donor_raster_spec is not None else DONOR_RASTER_SPEC
    project_spec = project_raster_spec if project_raster_spec is not None else PROJECT_RASTER_SPEC

    if manifest is None:
        manifest = load_json_if_exists(dirs["04"] / "reference_area_FINAL_manifest.json")

    # P3 FIX: messaggio chiaro se il manifest non è risolvibile (Step 04 non eseguito
    # o output mancante) invece di un NameError opaco a valle.
    if manifest is None:
        raise RuntimeError(
            "STEP 05 bloccato: manifest di Step 04 non disponibile.\n"
            f"  Cercato in: {dirs['04'] / 'reference_area_FINAL_manifest.json'}\n"
            "Eseguire prima Step 04 (cella §6) nella stessa sessione, oppure passare "
            "manifest=... esplicitamente, oppure verificare OUTPUT_ROOT."
        )

    if STRICT_LOCK_REQUIRED and manifest and manifest.get("lock_status") != "LOCKED":
        raise RuntimeError("STEP 05 blocked: reference area is not LOCKED in Step 04 manifest.")

    if twin_df is None:
        twin_path = _find_table(dirs["03"], "twin_tested_pixels", required=True)
        twin_df = load_df(twin_path)
    else:
        twin_path = "provided_dataframe"

    if project_df is None:
        if use_matched_project_rows:
            project_df = twin_df.copy()
            project_path = "matched_project_rows_from_twin_df"
        else:
            project_path = _find_table(dirs["01"], "project_pixels_raw", required=True)
            project_df = load_df(project_path)
    else:
        project_path = "provided_dataframe"

    if verbose:
        print(f"\n{'=' * 72}")
        print("STEP 05 — Raster-based Unadjusted Baseline + CI90 / UNCBSL")
        print(f"RUN root                  : {root}")
        print(f"Twin pixels source         : {twin_path}")
        print(f"Project pixels source      : {project_path}")
        print(f"T0 year                    : {temporal_meta['t0_year']}")
        print(f"Monitoring year            : {temporal_meta['monitoring_year']}")
        print(f"Monitoring period          : {monitoring_period_years:g} years")
        print(f"First valid endpoint       : {temporal_meta['first_valid_monitoring_year']}")
        print(f"Output                     : {out_dir}")
        print(f"{'=' * 72}")

    # 1. Locked matched controls — no deduplication by default.
    control_pixels, ref_lon_col, ref_lat_col, n_control_before, control_selection_mode = prepare_control_rows(
        twin_df, use_all_matched_rows=use_all_matched_rows
    )
    if verbose:
        print(f"\n[1] Locked controls: {n_control_before:,} rows -> {len(control_pixels):,} matched rows ({control_selection_mode})")

    control_pixels, donor_sample_meta = sample_delta_from_raster_spec(
        control_pixels,
        lon_col=ref_lon_col,
        lat_col=ref_lat_col,
        raster_spec=donor_spec,
        prefix="ref",
        monitoring_period_years=monitoring_period_years,
    )

    # 2. Project/PA matched rows — no deduplication by default.
    project_pixels, proj_lon_col, proj_lat_col, n_project_before, project_selection_mode = prepare_project_rows(
        project_df, use_all_matched_rows=use_all_matched_rows
    )
    if verbose:
        print(f"[2] Project pixels : {n_project_before:,} rows -> {len(project_pixels):,} matched rows ({project_selection_mode})")

    project_pixels, project_sample_meta = sample_delta_from_raster_spec(
        project_pixels,
        lon_col=proj_lon_col,
        lat_col=proj_lat_col,
        raster_spec=project_spec,
        prefix="proj",
        monitoring_period_years=monitoring_period_years,
    )

    # 3. Control CI / UNCBSL.
    control_valid, control_summary = calculate_control_ci(
        control_pixels,
        lon_col=ref_lon_col,
        lat_col=ref_lat_col,
        donor_sample_meta=donor_sample_meta,
        use_conservative_block_ci=use_conservative_block_ci,
        spatial_block_size_m=spatial_block_size_m,
    )

    # 4. Full PA area + project observed delta.
    full_project_area_ha, area_meta = resolve_project_area_ha(
        project_area_ha=project_area_ha,
        project_pixels=project_pixels,
        project_sample_meta=project_sample_meta,
        allow_pixel_area_fallback=False,
    )
    project_valid, project_summary = calculate_project_summary(project_pixels, full_project_area_ha)

    # 5. Unadjusted baseline total.
    baseline_summary = build_unadjusted_baseline_summary(
        control_summary,
        project_summary,
        monitoring_period_years=monitoring_period_years,
        temporal_meta=temporal_meta,
        allow_negative_baseline=allow_negative_baseline,
    )

    # 6. Combined summary.
    summary = {
        "run_id": RUN_ID,
        "project_name": PROJECT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": "GS STARR Track 1 SEMDB",
        "step": "05_raster_based_unadjusted_baseline_CI90_UNCBSL",
        "n_rows_control_before_deduplication": int(n_control_before),
        "n_rows_project_before_deduplication": int(n_project_before),
        "row_selection_mode": control_selection_mode,
        "control_row_selection_mode": control_selection_mode,
        "project_row_selection_mode": project_selection_mode,
        **control_summary,
        **project_summary,
        **baseline_summary,
        "coordinate_key_round_decimals": int(COORD_ROUND_DECIMALS),
        "area_metadata": area_meta,
    }

    # 7. Save outputs.
    control_out = save_df(control_valid, out_dir / "baseline_control_deltaC_distribution", OUTPUT_FORMAT)
    project_out = save_df(project_valid, out_dir / "project_deltaC_distribution", OUTPUT_FORMAT)

    summary_csv = out_dir / "baseline_CI90_UNCBSL_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    figs = {}
    fig1, fig1_path = plot_delta_ci(control_valid, summary, out_dir)
    fig2, fig2_path = plot_project_vs_control(control_valid, project_valid, out_dir)
    figs["control_ci"] = fig1
    figs["project_vs_control"] = fig2

    report = {
        "run_id": RUN_ID,
        "project_name": PROJECT_NAME,
        "timestamp_utc": summary["timestamp_utc"],
        "methodology": "GS STARR Track 1 SEMDB",
        "step": "05_raster_based_unadjusted_baseline_CI90_UNCBSL",
        "purpose": (
            "Extract carbon stock/change values from donor and PA rasters at previously selected "
            "pixels and calculate unadjusted baseline plus CI90/UNCBSL."
        ),
        "formula": {
            "pixel_delta_from_stock": "(C_y - C_t0) / monitoring_period_years",
            "SE_control": "sigma_control / sqrt(N_control_matched_rows)",
            "CI90_abs": "1.645 * SE_control",
            "UNCBSL": "CI90_abs_final / mean_deltaC_control, only if mean_deltaC_control > 0",
            "CI90_abs_final": "max(pixel_based_CI90, block_based_CI90 when available)",
            "unadjusted_baseline_creditable_mean": "max(mean_deltaC_control_tC_ha_yr, 0)",
            "unadjusted_baseline_total_tC_yr": "unadjusted_baseline_creditable_mean * full_project_area_ha",
            "unadjusted_baseline_total_tC_period": "unadjusted_baseline_total_tC_yr * monitoring_period_years",
        },
        "input_files": {
            "twin_tested_pixels": str(twin_path),
            "project_pixels_raw": str(project_path),
            "reference_area_manifest": str(dirs["04"] / "reference_area_FINAL_manifest.json"),
        },
        "temporal_rule": temporal_meta,
        "raster_sampling": {
            "donor_reference": donor_sample_meta,
            "project_pa": project_sample_meta,
        },
        "summary": summary,
        "compliance_notes": [
            "N_control is based on all locked/twin-tested matched control rows; duplicate reference pixels are not dropped.",
            "The full donor pool is not used for UNCBSL.",
            "The PA/project values are extracted from matched project rows by default, not deduplicated; the matched-row mean is scaled to the full PA area supplied from the notebook.",
            "At T0 the cumulative unadjusted baseline and CI are zero by definition.",
            "Post-T0 baseline calculation is blocked unless monitoring_year - T0_YEAR >= MIN_MONITORING_PERIOD_YEARS.",
            "Negative mean control ΔC is retained as diagnostic but creditable baseline removals are set to zero.",
            "CI90 is reported as pixel-based and, when possible, block-based; the final CI uses the larger value.",
            "This step reports an unadjusted baseline and baseline uncertainty; it does not apply DAF or calculate final GSVERs.",
            "If mean control ΔC <= 0, UNCBSL is not applicable for baseline-removal crediting.",
        ],
        "reference_area_manifest_summary": {
            "lock_status": manifest.get("lock_status") if manifest else None,
            "reference_area_definition": manifest.get("reference_area_definition", {}) if manifest else {},
            "monitoring": manifest.get("monitoring", {}) if manifest else {},
        },
        "pm_handoff_crediting": _build_pm_handoff(summary, monitoring_period_years),
        "outputs": {
            "control_deltaC_distribution": str(control_out),
            "project_deltaC_distribution": str(project_out),
            "summary_csv": str(summary_csv),
            "report_json": str(out_dir / "baseline_CI90_UNCBSL_report.json"),
            "fig_control_ci": str(fig1_path),
            "fig_project_vs_control": str(fig2_path),
        },
    }

    report_json = out_dir / "baseline_CI90_UNCBSL_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"\n{'=' * 72}")
        print("STEP 05 RESULTS")
        print(f"{'─' * 72}")
        print(f"Matched valid controls         : {summary['n_control_matched_valid_rows']:,}")
        print(f"Unique ref pixels represented  : {summary['n_control_unique_reference_pixels_represented']:,}")
        print(f"Matched valid PA rows          : {summary['n_project_matched_valid_rows']:,}")
        print(f"Unique PA pixels represented   : {summary['n_project_unique_pixels_represented']:,}")
        print(f"Project area used              : {summary['project_area_ha']:,.4f} ha")
        print(f"{'─' * 72}")
        # ── Primary output: BL_unadj,y = ΔC_ref,y × A_project (GS STARR Eq 31a) ──
        # These are the numbers a PM needs — absolute tC and tCO2e, NOT fractions.
        print(f"  ΔC_ref,y (creditable)        : {summary['delta_C_ref_y_tC_ha_yr']:+.6f} tC/ha/yr")
        print(f"  A_project                    : {summary['project_area_ha_used']:,.4f} ha")
        print(f"  BL_unadj,y  [tC/yr]          : {summary['BL_unadj_y_tC']:+,.4f} tC/yr")
        print(f"  BL_unadj,y  [tCO2e/yr]       : {summary['BL_unadj_y_tCO2e']:+,.4f} tCO2e/yr  ← primary PM value")
        print(f"  BL_unadj    [tCO2e/period]   : {summary['BL_unadj_period_tCO2e']:+,.4f} tCO2e   ← primary PM value")
        print(f"{'─' * 72}")
        print(f"  [diagnostic] ΔC_ref,y raw    : {summary['mean_deltaC_control_tC_ha_yr']:+.6f} tC/ha/yr (before max(x,0))")
        print(f"  [diagnostic] BL_unadj raw/yr : {summary['unadjusted_baseline_total_raw_tC_yr']:+,.4f} tC/yr")
        print(f"  N rows / N effettivo (SE)    : {summary['n_control_matched_valid_rows']:,} / {summary['n_control_effective_for_se']:,}")
        print(f"  [diagnostic] CI90 pixel (Neff): ±{summary['ci90_abs_pixel_tC_ha_yr']:.6f} tC/ha/yr")
        print(f"  [diagnostic] CI90 pixel naive : ±{summary['ci90_abs_pixel_naive_all_rows_tC_ha_yr']:.6f} tC/ha/yr (N=tutte le righe, sottostima)")
        print(f"  [diagnostic] CI90 final      : ±{summary['ci90_abs_final_tC_ha_yr']:.6f} tC/ha/yr ({summary['ci90_final_source']})")
        if summary["uncbsl_fraction"] is None:
            print(f"  [diagnostic] UNCBSL          : NOT APPLICABLE ({summary['uncbsl_status']})")
        else:
            print(f"  [diagnostic] UNCBSL fraction : {summary['uncbsl_fraction']:.6f}  ({summary['uncbsl_percent']:.2f}%)"
                  "  [uncertainty ratio, used in Eq-6 — not the baseline value]")
        print(f"  BL_unadj + CI period (tC)    : {summary['baseline_uncertainty_adjusted_total_tC_period']:,.4f} tC")
        print(f"  Project observed (tC/period) : {summary['project_observed_total_tC_period']:,.4f} tC")
        print(f"  Project − BL_unadj (tC/per.) : {summary['project_minus_unadjusted_baseline_tC_period']:,.4f} tC")
        print(f"  Output                       : {out_dir}")
        print(f"{'=' * 72}")

        # PM crediting handoff summary.
        ho = report.get("pm_handoff_crediting", {})
        if ho:
            print(f"\n{'─' * 72}")
            print("PM CREDITING HANDOFF  (values for Eq-6 / Eq-22)")
            print(f"{'─' * 72}")
            print(f"  BL_unadj,y  (tCO2e/yr)       : {summary['BL_unadj_y_tCO2e']:+,.2f}")
            print(f"  BL_unadj    (tCO2e/period)   : {summary['BL_unadj_period_tCO2e']:+,.2f}")
            print(f"  UNCBSL fraction [Eq-6 factor] : {ho.get('UNCBSL_fraction', 'N/A')}")
            print(f"  CI90 final (tCO2e/ha/yr)      : {ho.get('CI90_abs_final_tCO2e_ha_yr', 0):,.6f}")
            print(f"  Project observed (tCO2e/per.) : {ho.get('project_observed_total_tCO2e_period', 0):,.2f}")
            print(f"  Illustrative BR_crediting     : {ho.get('illustrative_BR_crediting_tCO2e_period', 0):,.2f} tCO2e")
            print(f"  DAF used (illustrative)       : {ho.get('illustrative_DAF_used', 'N/A')}")
            print(f"  >> {ho.get('illustrative_note', '')}")
            print(f"{'─' * 72}")

    return control_valid, project_valid, summary, report, figs, out_dir


# Backward-compatible names.
def run_baseline_ci_uncbsl(*args, **kwargs):
    return run_baseline_ci_uncbsl_from_rasters(*args, **kwargs)


def calculate_uncbsl(*args, **kwargs):
    return run_baseline_ci_uncbsl_from_rasters(*args, **kwargs)


def main():
    run_baseline_ci_uncbsl_from_rasters()


if __name__ == "__main__":
    main()