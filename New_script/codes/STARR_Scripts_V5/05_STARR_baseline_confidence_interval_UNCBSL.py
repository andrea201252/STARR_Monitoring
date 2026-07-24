# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB | 05_STARR_baseline_confidence_interval_UNCBSL.py
STEP 05 — Baseline non aggiustata su base raster + CI90 / UNCBSL
============================================================

Scopo
-----
Questo Step 05 non richiede più control_carbon_stock_change.csv/parquet.

Legge raster di stock di carbonio o di variazione del carbonio per:
  1. i pixel donor/reference bloccati, selezionati da Step 03 / Step 04;
  2. i pixel dell'area di progetto/attività prodotti da Step 01;

quindi estrae i valori raster a quelle coordinate di pixel e calcola:

  - la distribuzione ΔC dei donor/reference;
  - la ΔC media della baseline non aggiustata in tC/ha/anno;
  - il totale della baseline non aggiustata per la PA in tC/anno e tC/periodo;
  - la ΔC osservata di progetto dai valori raster della PA;
  - l'intervallo di confidenza al 90% e UNCBSL sui pixel donor/reference bloccati.

Regole metodologiche fondamentali
----------------------------------
1. UNCBSL è calcolato solo dai pixel donor/reference bloccati, matchati e sottoposti
   al twin test. Il pool donor completo non viene mai usato.

2. N_control conta tutte le righe PA-control matchate/sottoposte al twin test.
   I pixel di riferimento duplicati NON vengono rimossi, perché il riutilizzo dei
   donor fa parte del design matched-pair accettato e deve restare rappresentato
   nel campione della baseline.

3. I valori PA/progetto sono estratti dal raster della PA usando i pixel di progetto
   di Step 01. Se project_df non è fornito, questo script ricarica:
      STARR_outputs/<RUN_ID>/01_extract/project_pixels_raw.parquet|csv

4. Gli input raster possono essere:
   A) raster ΔC annuale diretto in tC/ha/anno, oppure
   B) raster di stock a T0 e all'anno di monitoraggio Y.

5. I raster di stock possono essere in:
   - tC/ha, nessuna conversione;
   - Mg AGB/ha, convertiti in carbonio con BIOMASS_TO_CARBON_FRACTION
     e l'opzionale ROOT_TO_SHOOT_RATIO.

Output
------
  05_baseline_CI_UNCBSL/
    baseline_control_deltaC_distribution.parquet|csv
    project_deltaC_distribution.parquet|csv
    baseline_CI90_UNCBSL_summary.csv
    baseline_CI90_UNCBSL_report.json
    deltaC_control_CI90.png
    deltaC_project_vs_control.png

Chiamata da notebook
--------------------
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
# PARAMETRI UTENTE
# ================================================================

RUN_ID = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v09_raster_ext5km"
PROJECT_NAME = "Idiofa_Lobi"

BASE_DIR_CANDIDATES = [
    "/content/drive/MyDrive/STARR_Idiofa_New_V2",
    "/content/content/MyDrive/STARR_Idiofa_New_V2",
]

OUTPUT_FORMAT = "parquet"
STRICT_LOCK_REQUIRED = True

# Nome cartella output radice — EDITABILE DAL NOTEBOOK (s05.OUTPUTS_DIRNAME = "...").
# Path radice del run: <base_dir> / OUTPUTS_DIRNAME / <RUN_ID>
OUTPUTS_DIRNAME = "STARR_outputs"

# Se i raster di stock sono C_t0 e C_y, ΔC = (C_y - C_t0) / MONITORING_PERIOD_YEARS.
# Se si usano raster ΔC diretti, questo valore viene solo riportato.
MONITORING_PERIOD_YEARS = 6.0

# Guardia metodologica sul monitoraggio.
# La baseline non aggiustata cumulata a T0 è zero per definizione.
# Il primo endpoint valido dopo T0 è T0 + MIN_MONITORING_PERIOD_YEARS.
T0_YEAR = 2018
MONITORING_YEAR = 2024
MIN_MONITORING_PERIOD_YEARS = 4.0

# Guardia conservativa sul CI contro l'autocorrelazione spaziale.
# Il CI su base pixel è sempre riportato. Il CI su base blocco è calcolato anche
# quando le coordinate spaziali sono risolvibili; il CI finale è max(CI pixel, CI blocco).
USE_CONSERVATIVE_BLOCK_CI = True
SPATIAL_BLOCK_SIZE_M = 500.0

# Area di progetto/attività in ettari.
# Mantenere questo valore a 0.0 nello script; passare la vera area PA dal notebook.
# Se lasciato a 0.0 e nessun argomento project_area_ha positivo è fornito, Step 05 si ferma.
PROJECT_AREA_HA = 0.0

# CI normale al 90%.
CI90_Z_VALUE = 1.645

# Conversione AGB -> C usata solo quando raster_spec["units"] == "AGB_Mg_ha".
# IPCC 2019 Refinement, Vol 4, Ch 4, Table 4.4.
# R di default per foresta tropicale umida ≈ 0.24.  Impostare a 0.0 solo se i raster
# includono già la biomassa ipogea o se la BGB è esclusa per disegno.
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
# Specifiche raster.
#
# Usare UNA delle seguenti modalità per ciascuna spec:
#
# MODALITÀ A — ΔC annuale diretta:
# {
#   "delta_raster": "/path/to/deltaC.tif",
#   "delta_band": 1,                    # opzionale; default 1
#   "delta_band_name": None,             # alternativa opzionale all'indice di banda
#   "units": "tC_ha_yr"
# }
#
# MODALITÀ B — stock a T0/Y in due raster:
# {
#   "stock_t0_raster": "/path/to/C_2018.tif",
#   "stock_y_raster":  "/path/to/C_2024.tif",
#   "stock_t0_band": 1,
#   "stock_y_band": 1,
#   "stock_t0_band_name": None,
#   "stock_y_band_name": None,
#   "units": "tC_ha"                    # oppure "AGB_Mg_ha"
# }
#
# MODALITÀ C — stock a T0/Y in un singolo raster multibanda:
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
# FUNZIONI IO
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
    root = bases[0] / OUTPUTS_DIRNAME / RUN_ID
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

    # Accetta una delle due:
    # - directory radice del run: .../STARR_outputs/<RUN_ID>
    # - directory di base globale: .../STARR_Idiofa_New_V2
    if (b0 / "01_extract").exists() or (b0 / "03_twin_test").exists():
        root = b0
    else:
        root = b0 / OUTPUTS_DIRNAME / RUN_ID

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
# FUNZIONI COORDINATE
# ================================================================

def standardize_ref_coords(df):
    out = df.copy()
    if {"ref_lon", "ref_lat"}.issubset(out.columns):
        lon_col, lat_col = "ref_lon", "ref_lat"
    elif {"lon", "lat"}.issubset(out.columns):
        lon_col, lat_col = "lon", "lat"
    else:
        raise ValueError("Reference coordinates missing. Required: ref_lon/ref_lat or lon/lat.")

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
        raise ValueError("Project coordinates missing. Required: proj_lon/proj_lat or lon/lat.")

    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out["_proj_lon_key"] = out[lon_col].round(COORD_ROUND_DECIMALS)
    out["_proj_lat_key"] = out[lat_col].round(COORD_ROUND_DECIMALS)
    out = out[out["_proj_lon_key"].notna() & out["_proj_lat_key"].notna()].copy()
    return out, lon_col, lat_col


def prepare_control_rows(twin_df, use_all_matched_rows=True):
    """
    Prepara le righe donor/reference per Step 05.

    Il comportamento di default è guidato dalla metodologia per il design matched-pair:
    usare tutte le righe matchate bloccate/sottoposte al twin test esattamente come
    selezionate da Step 03. I pixel di riferimento non vengono deduplicati, perché un
    pixel donor riutilizzato rappresenta più di un match PA-control accettato e quindi
    deve mantenere il suo peso di riga matchata nella distribuzione della baseline.

    Impostare use_all_matched_rows=False solo per controlli diagnostici di sensitività.
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
    """Prepara le righe PA/progetto. Di default mantiene tutte le righe PA matchate."""
    out, lon_col, lat_col = standardize_project_coords(project_df)
    before = len(out)
    if not use_all_matched_rows:
        out = out.drop_duplicates(subset=["_proj_lon_key", "_proj_lat_key"]).reset_index(drop=True)
        selection_mode = "coordinate_deduplicated_diagnostic"
    else:
        out = out.reset_index(drop=True)
        selection_mode = "all_matched_rows_no_deduplication"
    return out, lon_col, lat_col, before, selection_mode


# Alias retrocompatibili. Non deduplicano più per default.
def deduplicate_controls(twin_df):
    out, lon_col, lat_col, before, _ = prepare_control_rows(twin_df, use_all_matched_rows=True)
    return out, lon_col, lat_col, before


def deduplicate_project_pixels(project_df):
    out, lon_col, lat_col, before, _ = prepare_project_rows(project_df, use_all_matched_rows=True)
    return out, lon_col, lat_col, before


# ================================================================
# CAMPIONAMENTO RASTER
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
        raise ValueError(f"Invalid band index {band} for {src.name}; the raster has {src.count} bands.")
    return band


def raster_pixel_area_ha(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            return None
        # Significativo solo per CRS proiettati. Per EPSG:4326 sarebbe in gradi².
        if src.crs.is_projected:
            return float(abs(src.transform.a * src.transform.e) / 10_000.0)
        return None


def sample_raster_values(points_df, lon_col, lat_col, raster_path,
                         band=None, band_name=None,
                         out_col="value",
                         chunk_size=RASTER_SAMPLE_CHUNK_SIZE):
    """
    Campiona una banda raster alle coordinate dei punti lon/lat in WGS84.
    Restituisce (values, sample_metadata).
    """
    p = _as_path(raster_path, out_col)

    lons = pd.to_numeric(points_df[lon_col], errors="coerce").to_numpy(dtype=np.float64)
    lats = pd.to_numeric(points_df[lat_col], errors="coerce").to_numpy(dtype=np.float64)

    if np.isnan(lons).any() or np.isnan(lats).any():
        raise ValueError(f"Found NaN coordinates while sampling {out_col}.")

    with rasterio.open(p) as src:
        if src.crs is None:
            raise ValueError(f"The raster has no CRS: {p}")

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
            f"AGB→C: C = AGB × {cf} × (1+{r}). BGB INCLUDED via R={r}. "
            "VERIFY that the raster is AGB aboveground-only: if it already includes "
            "roots or is total C, set ROOT_TO_SHOOT_RATIO=0 to avoid "
            f"inflating carbon by ~{r*100:.0f}%."
        ) if r > 0 else (
            f"AGB→C: C = AGB × {cf}. BGB EXCLUDED (R=0). Correct only if the raster "
            "is already total C or if BGB is excluded by design."
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
    raise ValueError("Unsupported raster units. Use 'tC_ha', 'AGB_Mg_ha', or the direct delta units 'tC_ha_yr'.")


def sample_delta_from_raster_spec(points_df, lon_col, lat_col, raster_spec,
                                  prefix, monitoring_period_years=MONITORING_PERIOD_YEARS):
    """
    Aggiunge:
      <prefix>_deltaC_tC_ha_yr
      colonne di stock campionate opzionali
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

    # MODALITÀ A — raster delta annuale diretto
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

    # MODALITÀ B/C — raster di stock
    stock_stack = spec.get("stock_raster")
    if stock_stack is not None and str(stock_stack).strip() != "":
        t0_path = stock_stack
        y_path = stock_stack
    else:
        t0_path = spec.get("stock_t0_raster")
        y_path = spec.get("stock_y_raster")

    if t0_path is None or str(t0_path).strip() == "" or y_path is None or str(y_path).strip() == "":
        raise ValueError(
            f"{prefix}: incomplete raster_spec. Provide delta_raster, "
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
    Applica la regola di monitoraggio post-T0.

    A T0 la baseline cumulata è zero. Un endpoint di differenza di stock è valido
    solo se il periodo di monitoraggio è >= al numero minimo di anni richiesto.
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
            "Correct T0_YEAR, MONITORING_YEAR, or MONITORING_PERIOD_YEARS."
        )

    if period < float(min_monitoring_period_years):
        raise ValueError(
            f"Invalid monitoring period: {period:g} years. The minimum required after T0 is "
            f"{float(min_monitoring_period_years):g} years. The first valid endpoint is "
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
    Aggiunge un id di blocco spaziale usato per il CI conservativo su base blocco.

    Priorità:
      1. coordinate proiettate già presenti nella tabella dei pixel;
      2. trasformazione lon/lat nel CRS del raster campionato se proiettato;
      3. fallback a EPSG:3857 se il CRS del raster è geografico o non disponibile.
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
    Solo fallback diagnostico: inferisce l'area del pixel di riga dai metadati di Step 01/raster.

    Questo NON rappresenta necessariamente la PA completa se Step 01 ha usato il campionamento
    della PA o se Step 05 usa righe matchate. Per i totali finali, usare resolve_project_area_ha().
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
            "Cannot infer the diagnostic project pixel area. Add pixel_area_ha to "
            "project_pixels_raw or use projected rasters with a metric pixel size."
        )

    return np.full(len(project_pixels), pixel_area, dtype=np.float64), {
        "area_source": "project_raster_resolution_diagnostic_only",
        "pixel_area_ha_constant": float(pixel_area),
    }


def resolve_project_area_ha(project_area_ha=None, project_pixels=None, project_sample_meta=None,
                            allow_pixel_area_fallback=False):
    """
    Risolve l'area di progetto/attività COMPLETA usata per scalare la ΔC media a tC totali.

    Il PROJECT_AREA_HA a livello di script è intenzionalmente 0.0. L'area reale deve
    essere passata dal notebook o da un'altra sorgente a monte. Questo impedisce di
    usare silenziosamente solo l'area delle righe campionate/matchate.
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
                "area_note": "Full PA/activity area provided externally; not inferred from the number of matched rows.",
            }

    if allow_pixel_area_fallback and project_pixels is not None:
        area_vals, meta = infer_project_pixel_area_ha(project_pixels, project_sample_meta or {})
        v = float(np.nansum(area_vals))
        if np.isfinite(v) and v > 0:
            meta.update({
                "project_area_ha": v,
                "area_is_full_project_area": False,
                "area_note": (
                    "Fallback area inferred from the available rows. Use for diagnostics only; "
                    "set project_area_ha for the final methodological outputs."
                ),
            })
            return v, meta

    raise ValueError(
        "PROJECT_AREA_HA is 0 or missing. Pass the full PA/activity area from the notebook, e.g.\n"
        "    PROJECT_AREA_HA = <value_from_initial_project_data>\n"
        "    s05.run_baseline_ci_uncbsl_from_rasters(..., project_area_ha=PROJECT_AREA_HA)\n"
        "Do not let Step 05 infer the PA area from the matched/sampled rows."
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
        crediting_note = "UNCBSL uses the final conservative CI90 on the positive mean baseline removal rate."
    else:
        uncbsl_fraction = None
        uncbsl_percent = None
        uncbsl_status = "not_applicable_mean_deltaC_zero_or_negative"
        crediting_note = (
            "The mean control ΔC is <= 0. The UNCBSL denominator is not valid for "
            "crediting baseline removals; the creditable unadjusted baseline "
            "removals are set to zero."
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
            "Pixel-based SE uses N_effective = UNIQUE reference pixels (anti "
            "pseudo-replication). The MEAN uses all matched rows (reused donors "
            "included). The *_naive_* value uses N=all rows and is diagnostic "
            "only (it underestimates uncertainty)."
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
    Calcola la ΔC osservata della PA dalle righe PA matchate e scala la media delle
    righe matchate all'area PA completa fornita esternamente.
    """
    df = project_pixels.copy()
    df["proj_deltaC_tC_ha_yr"] = pd.to_numeric(df["proj_deltaC_tC_ha_yr"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["proj_deltaC_tC_ha_yr"].notna()].copy()

    if df.empty:
        raise RuntimeError("No valid matched project rows after raster sampling.")

    pa_ha = float(full_project_area_ha)
    if not np.isfinite(pa_ha) or pa_ha <= 0:
        raise ValueError("full_project_area_ha must be positive.")

    n_rows = int(len(df))
    n_unique_proj = int(df[["_proj_lon_key", "_proj_lat_key"]].drop_duplicates().shape[0]) \
        if {"_proj_lon_key", "_proj_lat_key"}.issubset(df.columns) else None

    # Pesatura uguale delle righe matchate. L'area di riga è solo un peso di scala in modo
    # che le somme siano riportate sull'area PA completa, non sull'area del sottoinsieme/campione.
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
    Costruisce le quantità della baseline non aggiustata grezze e creditabili.

    Baseline grezza = ΔC media dei controlli × area PA.
    Tasso di rimozione di baseline creditabile = max(ΔC media dei controlli, 0) × area PA,
    A MENO CHE allow_negative_baseline=True (richiesta PM), nel qual caso il
    valore creditabile è uguale al valore grezzo anche quando negativo.

    Una ΔC dei controlli negativa normalmente non può generare rimozioni di baseline per il crediting.
    A T0 la baseline non aggiustata cumulata è zero.
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

    # ── Quantità numeriche BL_unadj,y destinate ai PM ─────────────────
    # Formula (GS STARR Eq 31a): BL_unadj,y = ΔC_ref,y × A_project
    #   ΔC_ref,y  = mean_creditable  [tC/ha/yr, max(mean_control_deltaC, 0)]
    #   A_project = pa_ha            [ha]
    # Questi sono numeri assoluti, NON frazioni o percentuali.
    _co2e = 44.0 / 12.0
    bl_unadj_y_tC          = baseline_creditable_total_yr          # tC/anno
    bl_unadj_y_tCO2e       = baseline_creditable_total_yr * _co2e  # tCO2e/anno
    bl_unadj_period_tC     = baseline_creditable_total_period       # tC/periodo
    bl_unadj_period_tCO2e  = baseline_creditable_total_period * _co2e  # tCO2e/periodo
    bl_unadj_raw_y_tCO2e   = baseline_raw_total_yr * _co2e         # tCO2e/anno (incl. negativi)

    return {
        # ── Output primari etichettati per metodologia (BL_unadj,y = ΔC_ref,y × A_project) ──
        "BL_unadj_y_tC":              bl_unadj_y_tC,
        "BL_unadj_y_tCO2e":           bl_unadj_y_tCO2e,
        "BL_unadj_period_tC":         bl_unadj_period_tC,
        "BL_unadj_period_tCO2e":      bl_unadj_period_tCO2e,
        "BL_unadj_raw_y_tCO2e":       bl_unadj_raw_y_tCO2e,
        "BL_unadj_formula":           "BL_unadj_y = delta_C_ref_y_tC_ha_yr * project_area_ha",
        "delta_C_ref_y_tC_ha_yr":     mean_creditable,
        "project_area_ha_used":       pa_ha,
        # ── Campi interni / diagnostici ───────────────────────────────
        "unadjusted_baseline_mean_raw_tC_ha_yr": mean_raw,
        "unadjusted_baseline_total_raw_tC_yr": baseline_raw_total_yr,
        "unadjusted_baseline_total_raw_tC_period": baseline_raw_total_period,
        "unadjusted_baseline_mean_tC_ha_yr": mean_creditable,
        "unadjusted_baseline_total_tC_yr": baseline_creditable_total_yr,
        "unadjusted_baseline_total_tC_period": baseline_creditable_total_period,
        "unadjusted_baseline_negative_control_rule": (
            "PM override active: baseline = mean_control_deltaC (even negative)"
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
            "WARNING double count: baseline_uncertainty_adjusted = BL + CI = "
            "BL × (1 + CI/mean) = BL × (1 + UNCBSL). It is mathematically the SAME "
            "uncertainty already represented by the (1+UNCBSL) factor in Eq-6. DO NOT add "
            "baseline_uncertainty_adjusted AND apply (1+UNCBSL): choose ONE path. "
            "Eq-6 uses BL_unadj × (1+DAF) × (1+UNCBSL) — so DO NOT also use the "
            "baseline_uncertainty_adjusted in that calculation."
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
            "UNCAR, leakage, permanence/risk deductions, nor the final GSVER issuance rules."
        ),
    }


# ================================================================
# PM CREDITING HANDOFF
# ================================================================

_TC_TO_TCO2E = 44.0 / 12.0
_DAF_LUF_DEFAULT = 0.0125


def _build_pm_handoff(summary, monitoring_period_years):
    """
    Costruisce un blocco di handoff pronto per i PM con tutti i valori necessari per completare
    Eq-6 (BR_crediting) ed Eq-22 (nAR) della metodologia GS STARR.

    Questa funzione NON applica DAF o BR_gov — quelle sono responsabilità dei PM.
    Fornisce conversioni in tCO2e pre-calcolate e formule esplicite.
    """
    period = float(monitoring_period_years)
    pa_ha = float(summary.get("project_area_ha", 0.0))

    # BL_unadj,y = ΔC_ref,y × A_project  (GS STARR Eq 31a) — numero assoluto, non una frazione.
    # Preferire le nuove chiavi BL_unadj_* esposte in build_unadjusted_baseline_summary; fallback alle legacy.
    bl_mean_tC_ha_yr = float(summary.get("delta_C_ref_y_tC_ha_yr",
                              summary.get("unadjusted_baseline_mean_tC_ha_yr", 0.0)))
    bl_total_tC_period = float(summary.get("BL_unadj_period_tC",
                                summary.get("unadjusted_baseline_total_tC_period", 0.0)))

    # Conversioni tCO2e — lette dal summary se già calcolate, altrimenti convertite localmente
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

    # --- Progetto osservato in tCO2e ---
    proj_total_tC_period = float(summary.get("project_observed_total_tC_period", 0.0))
    proj_total_tCO2e_period = proj_total_tC_period * _TC_TO_TCO2E

    # --- BR_crediting illustrativo (riferimento per i PM, usando il floor DAF) ---
    # Eq-6: BR_crediting = MAX[ BR_unadj × (1 + DAF) × (1 + UNCBSL), BR_gov ]
    # BR_gov è specifico del progetto e deve essere impostato dai PM.
    daf_floor = _DAF_LUF_DEFAULT
    uncbsl_for_calc = float(uncbsl_frac) if uncbsl_frac is not None else 0.0
    br_crediting_illustrative_tCO2e = (
        bl_total_tCO2e_period * (1.0 + daf_floor) * (1.0 + uncbsl_for_calc)
    )

    return {
        "scope": (
            "Pre-calculated values for Eq-6 and Eq-22 of GS STARR Track 1 SEMDB. "
            "DAF and BR_gov are PM responsibilities and are NOT applied here."
        ),
        "tC_to_tCO2e_factor": _TC_TO_TCO2E,
        "project_area_ha": pa_ha,
        "monitoring_period_years": period,

        # Baseline non aggiustata (BL_unadj) — Eq-1
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

        # Progetto osservato
        "project_observed_total_tC_period": proj_total_tC_period,
        "project_observed_total_tCO2e_period": proj_total_tCO2e_period,

        # Eq-6 illustrativa (i PM devono verificare DAF e impostare BR_gov)
        "illustrative_DAF_used": daf_floor,
        "illustrative_BR_crediting_tCO2e_period": br_crediting_illustrative_tCO2e,
        "illustrative_note": (
            f"BR_crediting = {bl_total_tCO2e_period:,.2f} × (1 + {daf_floor}) "
            f"× (1 + {uncbsl_for_calc:.6f}) = {br_crediting_illustrative_tCO2e:,.2f} tCO2e. "
            "The PMs must compare with BR_gov and take the MAX per Eq-6."
        ),

        # Prossimi passi per i PM
        "pm_actions_required": [
            "1. Verify or update DAF (LUF floor = 1.25%; check the host-country sectoral override)",
            "2. Determine BR_gov (unconditional reforestation NDC target, if applicable)",
            "3. Compute BR_crediting = MAX[ BL_unadj_tCO2e × (1+DAF) × (1+UNCBSL), BR_gov ]  (Eq-6)",
            "4. Compute AR_final (activity removals adjusted for UNCAR)  (Eq-14/16)",
            "5. Compute AE (activity emissions: burning, fertilizers, fuel, livestock)  (Eq-7)",
            "6. Compute LE_final (leakage: buffer strip monitoring or yield test + 15% buffer)  (Eq-20/21)",
            "7. Compute nAR = (AR_final - AE) - (BR_crediting - BE) - LE_final  (Eq-22, BE=0)",
            "8. Apply the buffer rate (10-20%) for the GSVERs  (Eq-23)",
        ],
    }


# ================================================================
# PLOTS
# ================================================================

def plot_delta_ci(control_df, summary, out_dir):
    v = control_df["ref_deltaC_tC_ha_yr"].dropna().astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(v, bins=min(60, max(10, int(np.sqrt(len(v))))), alpha=0.85, edgecolor="black")
    ax.axvline(summary["mean_deltaC_control_tC_ha_yr"], linestyle="--", linewidth=1.5, label="Mean control ΔC")
    ax.axvline(summary["ci90_lower_tC_ha_yr"], linestyle=":", linewidth=1.5, label="90% CI lower")
    ax.axvline(summary["ci90_upper_tC_ha_yr"], linestyle=":", linewidth=1.5, label="90% CI upper")
    ax.set_xlabel("ΔC locked donor/reference pixels (tC/ha/year)")
    ax.set_ylabel("Pixel count")
    ax.set_title("GS STARR Track 1 SEMDB — 90% confidence interval of control ΔC")
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
    Step 05 su base raster.

    Parametri
    ---------
    base_dirs : list[path]
        Di solito [OUTPUT_ROOT]. Usato per ricaricare i file di Step 01/03/04 se i dataframe non sono forniti.
    output_dir : path
        Cartella di output.
    donor_raster_spec : dict
        Spec raster per i pixel donor/reference bloccati.
    project_raster_spec : dict
        Spec raster per i pixel PA/progetto.
    twin_df : DataFrame | None
        twin_tested_pixels di Step 03. Se None, ricarica da 03_twin_test.
    project_df : DataFrame | None
        Dataframe PA/progetto opzionale. Se None e use_matched_project_rows=True, Step 05 usa twin_df
        in modo che PA e controllo siano valutati sulle stesse righe matchate.
    project_area_ha : float | None
        Area di progetto/attività completa in ettari. Richiesta per i calcoli finali del totale della baseline.
    use_all_matched_rows : bool
        Se True, nessuna deduplicazione delle coordinate viene applicata alle righe matchate PA/controllo.
    use_matched_project_rows : bool
        Se True e project_df è None, i valori PA sono estratti da proj_lon/proj_lat di twin_df.
    manifest : dict | None
        Manifest di Step 04. Se None, ricarica da 04_reference_area.
    monitoring_period_years : float | None
        Sovrascrive MONITORING_PERIOD_YEARS.
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
            "STEP 05 blocked: Step 04 manifest not available.\n"
            f"  Searched in: {dirs['04'] / 'reference_area_FINAL_manifest.json'}\n"
            "Run Step 04 first (cell §6) in the same session, or pass "
            "manifest=... explicitly, or check OUTPUT_ROOT."
        )

    if STRICT_LOCK_REQUIRED and manifest and manifest.get("lock_status") != "LOCKED":
        raise RuntimeError("STEP 05 blocked: the reference area is not LOCKED in the Step 04 manifest.")

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
        print("STEP 05 — Raster-based unadjusted baseline + CI90 / UNCBSL")
        print(f"RUN root                   : {root}")
        print(f"Parallel pixels source     : {twin_path}")
        print(f"Project pixels source      : {project_path}")
        print(f"T0 year                    : {temporal_meta['t0_year']}")
        print(f"Monitoring year            : {temporal_meta['monitoring_year']}")
        print(f"Monitoring period          : {monitoring_period_years:g} years")
        print(f"First valid endpoint       : {temporal_meta['first_valid_monitoring_year']}")
        print(f"Output                     : {out_dir}")
        print(f"{'=' * 72}")

    # 1. Controlli matchati bloccati — nessuna deduplicazione di default.
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

    # 2. Righe matchate progetto/PA — nessuna deduplicazione di default.
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

    # 3. CI / UNCBSL dei controlli.
    control_valid, control_summary = calculate_control_ci(
        control_pixels,
        lon_col=ref_lon_col,
        lat_col=ref_lat_col,
        donor_sample_meta=donor_sample_meta,
        use_conservative_block_ci=use_conservative_block_ci,
        spatial_block_size_m=spatial_block_size_m,
    )

    # 4. Area PA completa + delta osservato di progetto.
    full_project_area_ha, area_meta = resolve_project_area_ha(
        project_area_ha=project_area_ha,
        project_pixels=project_pixels,
        project_sample_meta=project_sample_meta,
        allow_pixel_area_fallback=False,
    )
    project_valid, project_summary = calculate_project_summary(project_pixels, full_project_area_ha)

    # 5. Totale della baseline non aggiustata.
    baseline_summary = build_unadjusted_baseline_summary(
        control_summary,
        project_summary,
        monitoring_period_years=monitoring_period_years,
        temporal_meta=temporal_meta,
        allow_negative_baseline=allow_negative_baseline,
    )

    # 6. Summary combinato.
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

    # 7. Salvataggio degli output.
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
            "Extract carbon stock/change values from the donor and PA rasters at the previously "
            "selected pixels and compute the unadjusted baseline plus CI90/UNCBSL."
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
            "N_control is based on all locked matched control rows subjected to the parallel test; duplicate reference pixels are not removed.",
            "The full donor pool is not used for UNCBSL.",
            "PA/project values are extracted from the matched project rows by default, without deduplication; the mean of the matched rows is scaled to the full PA area provided by the notebook.",
            "At T0 the cumulative unadjusted baseline and the CI are zero by definition.",
            "The post-T0 baseline computation is blocked unless monitoring_year - T0_YEAR >= MIN_MONITORING_PERIOD_YEARS.",
            "A negative mean control ΔC is kept as diagnostic but creditable baseline removals are set to zero.",
            "The CI90 is reported on a pixel basis and, when possible, on a block basis; the final CI uses the larger value.",
            "This step reports an unadjusted baseline and the baseline uncertainty; it does not apply DAF nor compute the final GSVERs.",
            "If the mean control ΔC is <= 0, UNCBSL is not applicable for crediting baseline removals.",
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
        print(f"Valid matched controls         : {summary['n_control_matched_valid_rows']:,}")
        print(f"Unique reference px represented : {summary['n_control_unique_reference_pixels_represented']:,}")
        print(f"Valid matched PA rows          : {summary['n_project_matched_valid_rows']:,}")
        print(f"Unique PA px represented       : {summary['n_project_unique_pixels_represented']:,}")
        print(f"Project area used              : {summary['project_area_ha']:,.4f} ha")
        print(f"{'─' * 72}")
        # ── Output primario: BL_unadj,y = ΔC_ref,y × A_project (GS STARR Eq 31a) ──
        # Questi sono i numeri che servono a un PM — tC e tCO2e assoluti, NON frazioni.
        print(f"  ΔC_ref,y (creditable)        : {summary['delta_C_ref_y_tC_ha_yr']:+.6f} tC/ha/year")
        print(f"  A_project                    : {summary['project_area_ha_used']:,.4f} ha")
        print(f"  BL_unadj,y  [tC/year]        : {summary['BL_unadj_y_tC']:+,.4f} tC/year")
        print(f"  BL_unadj,y  [tCO2e/year]     : {summary['BL_unadj_y_tCO2e']:+,.4f} tCO2e/year  ← primary PM value")
        print(f"  BL_unadj    [tCO2e/period]   : {summary['BL_unadj_period_tCO2e']:+,.4f} tCO2e   ← primary PM value")
        print(f"{'─' * 72}")
        print(f"  [diagnostic] ΔC_ref,y raw    : {summary['mean_deltaC_control_tC_ha_yr']:+.6f} tC/ha/year (before max(x,0))")
        print(f"  [diagnostic] BL_unadj raw/year : {summary['unadjusted_baseline_total_raw_tC_yr']:+,.4f} tC/year")
        print(f"  N rows / N effective (SE)    : {summary['n_control_matched_valid_rows']:,} / {summary['n_control_effective_for_se']:,}")
        print(f"  [diagnostic] CI90 pixel (Neff): ±{summary['ci90_abs_pixel_tC_ha_yr']:.6f} tC/ha/year")
        print(f"  [diagnostic] CI90 pixel naive : ±{summary['ci90_abs_pixel_naive_all_rows_tC_ha_yr']:.6f} tC/ha/year (N=all rows, underestimate)")
        print(f"  [diagnostic] CI90 final      : ±{summary['ci90_abs_final_tC_ha_yr']:.6f} tC/ha/year ({summary['ci90_final_source']})")
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

        # Riepilogo handoff crediting per i PM.
        ho = report.get("pm_handoff_crediting", {})
        if ho:
            print(f"\n{'─' * 72}")
            print("PM HANDOFF CREDITING  (values for Eq-6 / Eq-22)")
            print(f"{'─' * 72}")
            print(f"  BL_unadj,y  (tCO2e/year)      : {summary['BL_unadj_y_tCO2e']:+,.2f}")
            print(f"  BL_unadj    (tCO2e/period)    : {summary['BL_unadj_period_tCO2e']:+,.2f}")
            print(f"  UNCBSL fraction [Eq-6 factor] : {ho.get('UNCBSL_fraction', 'N/A')}")
            print(f"  CI90 final (tCO2e/ha/year)    : {ho.get('CI90_abs_final_tCO2e_ha_yr', 0):,.6f}")
            print(f"  Project observed (tCO2e/per.) : {ho.get('project_observed_total_tCO2e_period', 0):,.2f}")
            print(f"  Illustrative BR_crediting     : {ho.get('illustrative_BR_crediting_tCO2e_period', 0):,.2f} tCO2e")
            print(f"  DAF used (illustrative)       : {ho.get('illustrative_DAF_used', 'N/A')}")
            print(f"  >> {ho.get('illustrative_note', '')}")
            print(f"{'─' * 72}")

    return control_valid, project_valid, summary, report, figs, out_dir


# Nomi retrocompatibili.
def run_baseline_ci_uncbsl(*args, **kwargs):
    return run_baseline_ci_uncbsl_from_rasters(*args, **kwargs)


def calculate_uncbsl(*args, **kwargs):
    return run_baseline_ci_uncbsl_from_rasters(*args, **kwargs)


def main():
    run_baseline_ci_uncbsl_from_rasters()


if __name__ == "__main__":
    main()