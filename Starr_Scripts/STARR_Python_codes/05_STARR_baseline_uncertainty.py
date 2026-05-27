# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB | 05_STARR_baseline_uncertainty.py
STEP 05 — Baseline Carbon Change + UNCBSL uncertainty
============================================================

Purpose
-------
Adds the missing Track 1 SEMDB baseline uncertainty calculation required by
Annex A.6 of the GS STARR consultation methodology:

    UNCBSL = 1.645 * SE_control / DeltaC_control
    SE_control = sigma_control / sqrt(N_control)

The calculation is performed on the carbon stock change of the LOCKED matched
control/reference pixels, not on the full donor pool.

Critical implementation choices
-------------------------------
1. N_control is the number of UNIQUE matched reference pixels.
   If the same donor pixel was reused during matching, duplicates are dropped
   before computing sigma and SE. This avoids artificial deflation of uncertainty.

2. DeltaC must be carbon stock change in tC/ha/year.
   The script accepts either a direct delta-C column or a pair of stock columns
   from which delta-C is computed.

3. Negative baseline stock change is set to zero for baseline-removal crediting.
   This follows the no-avoided-degradation safeguard: the method credits net
   removals, not avoided degradation.

Expected input
--------------
The script automatically looks for the pipeline outputs:
  03_twin_test/twin_tested_pixels.parquet|csv
  04_reference_area/reference_area_FINAL_manifest.json
  01_extract/extraction_report.json

It also needs monitored carbon stock change for the reference pixels. Provide it
in one of these ways:

A) Put a file named control_carbon_stock_change.parquet/csv in:
     STARR_outputs/<RUN_ID>/05_baseline_uncertainty/
   or in:
     STARR_outputs/<RUN_ID>/04_reference_area/
     STARR_outputs/<RUN_ID>/03_twin_test/

B) Or call run_baseline_uncertainty(control_change_df=your_dataframe).

Accepted carbon-change schemas
------------------------------
Direct delta-C column, any one of:
  deltaC_control_tC_ha_yr
  deltaC_ref_tC_ha_yr
  deltaC_tC_ha_yr
  delta_C_tC_ha_yr
  dC_tC_ha_yr

Stock columns in tC/ha, any one pair:
  C_ref_t0_tC_ha + C_ref_y_tC_ha
  C_control_t0_tC_ha + C_control_y_tC_ha
  carbon_t0_tC_ha + carbon_y_tC_ha

AGB columns in Mg dry matter/ha, optional fallback:
  AGB_ref_t0_Mg_ha + AGB_ref_y_Mg_ha
  AGB_control_t0_Mg_ha + AGB_control_y_Mg_ha
  agb_t0_Mg_ha + agb_y_Mg_ha

When AGB is used, carbon is calculated as:
  C = AGB * (1 + ROOT_TO_SHOOT_RATIO) * BIOMASS_TO_CARBON_FRACTION

Outputs
-------
  baseline_uncertainty_summary.csv
  baseline_uncertainty_report.json
  control_deltaC_pixels.parquet/csv
  control_deltaC_distribution.png
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


def _is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


if not _is_notebook():
    matplotlib.use("Agg")


# ================================================================
# USER PARAMETERS
# ================================================================

RUN_ID = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v06_raster"
PROJECT_NAME = "Idiofa_Lobi"

BASE_DIR_CANDIDATES = [
    "/content/drive/MyDrive/STARR_Idiofa_New_V2",
    "/content/content/MyDrive/STARR_Idiofa_New_V2",
]

OUTPUT_FORMAT = "parquet"

# Monitoring period covered by the carbon stock change.
# Example: if C_t0 is 2018 and C_y is 2024, set MONITORING_PERIOD_YEARS = 6.
MONITORING_PERIOD_YEARS = 1.0

# Project eligible area. If None, inferred from project_n × pixel area from Step 01.
PROJECT_AREA_HA = None

# Downward adjustment and government floor for Eq-6.
DAF_TOTAL = 0.0125
BR_GOV_Y_TCO2E = 0.0

# AGB fallback conversion settings.
BIOMASS_TO_CARBON_FRACTION = 0.47
ROOT_TO_SHOOT_RATIO = 0.0

# If True, negative mean DeltaC is set to zero for BR_unadj calculation.
APPLY_NO_AVOIDED_DEGRADATION_SAFEGUARD = True

# Unique control-pixel key. Use rounded coordinates to avoid floating precision noise.
COORD_ROUND_DECIMALS = 7


# ================================================================
# IO HELPERS
# ================================================================

def load_df(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File non trovato: {p}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Formato non supportato: {p.suffix}")


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


def find_file(base_dirs, stem, required=True):
    for base in base_dirs:
        base = Path(base)
        for ext in ["parquet", "csv", "json"]:
            p = base / f"{stem}.{ext}"
            if p.exists():
                return p
    if required:
        raise FileNotFoundError(f"'{stem}' non trovato in {[str(b) for b in base_dirs]}")
    return None


def detect_base_dirs():
    bases = [Path(p) for p in BASE_DIR_CANDIDATES if Path(p).exists()]
    if not bases:
        raise FileNotFoundError(f"Nessuna directory trovata: {BASE_DIR_CANDIDATES}")
    root = bases[0] / "STARR_outputs" / RUN_ID
    return root, {
        "root": root,
        "01": root / "01_extract",
        "02": root / "02_matching",
        "03": root / "03_twin_test",
        "04": root / "04_reference_area",
        "05": root / "05_baseline_uncertainty",
    }


def load_json_if_exists(path):
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ================================================================
# DATA PREP
# ================================================================

def standardize_ref_coords(df):
    out = df.copy()
    if {"ref_lon", "ref_lat"}.issubset(out.columns):
        lon_col, lat_col = "ref_lon", "ref_lat"
    elif {"lon", "lat"}.issubset(out.columns):
        lon_col, lat_col = "lon", "lat"
    else:
        raise ValueError("Coordinate reference mancanti. Servono ref_lon/ref_lat o lon/lat.")

    out["_ref_lon_key"] = pd.to_numeric(out[lon_col], errors="coerce").round(COORD_ROUND_DECIMALS)
    out["_ref_lat_key"] = pd.to_numeric(out[lat_col], errors="coerce").round(COORD_ROUND_DECIMALS)
    out = out[out["_ref_lon_key"].notna() & out["_ref_lat_key"].notna()].copy()
    return out, lon_col, lat_col


def merge_control_change(twin_df, control_change_df):
    t, _, _ = standardize_ref_coords(twin_df)
    c, _, _ = standardize_ref_coords(control_change_df)

    # Keep one monitored value per reference pixel.
    c = c.drop_duplicates(subset=["_ref_lon_key", "_ref_lat_key"]).copy()

    # Drop duplicate carbon columns already present in twin_df to avoid suffix confusion.
    carbon_like = [col for col in c.columns if col not in ("_ref_lon_key", "_ref_lat_key")]
    existing = [col for col in carbon_like if col in t.columns]
    if existing:
        t = t.drop(columns=existing, errors="ignore")

    out = t.merge(c, on=["_ref_lon_key", "_ref_lat_key"], how="left", suffixes=("", "_carbon"))
    return out


def infer_project_area_ha(extraction_report, project_area_ha=PROJECT_AREA_HA):
    if project_area_ha is not None:
        return float(project_area_ha), "manual_PROJECT_AREA_HA"

    project_n = extraction_report.get("project_n")
    pixel_size_m = extraction_report.get("pixel_size_m")
    if project_n is not None and pixel_size_m is not None:
        return float(project_n) * (float(pixel_size_m) ** 2) / 10000.0, "project_n_x_pixel_area"

    raise ValueError("PROJECT_AREA_HA non impostata e impossibile inferirla da extraction_report.json.")


# ================================================================
# DELTA-C DETECTION
# ================================================================

def first_existing_pair(df, pairs):
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            return a, b
    return None, None


def compute_delta_c(df, monitoring_period_years=MONITORING_PERIOD_YEARS):
    out = df.copy()
    if monitoring_period_years <= 0:
        raise ValueError("MONITORING_PERIOD_YEARS deve essere > 0.")

    direct_cols = [
        "deltaC_control_tC_ha_yr",
        "deltaC_ref_tC_ha_yr",
        "deltaC_tC_ha_yr",
        "delta_C_tC_ha_yr",
        "dC_tC_ha_yr",
    ]
    for col in direct_cols:
        if col in out.columns:
            out["deltaC_control_tC_ha_yr"] = pd.to_numeric(out[col], errors="coerce")
            return out, {
                "delta_source": col,
                "delta_units": "tC/ha/yr",
                "monitoring_period_years": float(monitoring_period_years),
                "conversion": "none",
            }

    c_pairs = [
        ("C_ref_t0_tC_ha", "C_ref_y_tC_ha"),
        ("C_control_t0_tC_ha", "C_control_y_tC_ha"),
        ("carbon_t0_tC_ha", "carbon_y_tC_ha"),
        ("C_t0_tC_ha", "C_y_tC_ha"),
    ]
    c0, cy = first_existing_pair(out, c_pairs)
    if c0 and cy:
        out["deltaC_control_tC_ha_yr"] = (
            pd.to_numeric(out[cy], errors="coerce") - pd.to_numeric(out[c0], errors="coerce")
        ) / float(monitoring_period_years)
        return out, {
            "delta_source": f"({cy} - {c0}) / monitoring_period_years",
            "delta_units": "tC/ha/yr",
            "monitoring_period_years": float(monitoring_period_years),
            "conversion": "none",
        }

    agb_pairs = [
        ("AGB_ref_t0_Mg_ha", "AGB_ref_y_Mg_ha"),
        ("AGB_control_t0_Mg_ha", "AGB_control_y_Mg_ha"),
        ("agb_t0_Mg_ha", "agb_y_Mg_ha"),
        ("AGB_t0_Mg_ha", "AGB_y_Mg_ha"),
    ]
    a0, ay = first_existing_pair(out, agb_pairs)
    if a0 and ay:
        conv = float(BIOMASS_TO_CARBON_FRACTION) * (1.0 + float(ROOT_TO_SHOOT_RATIO))
        out["deltaC_control_tC_ha_yr"] = (
            pd.to_numeric(out[ay], errors="coerce") - pd.to_numeric(out[a0], errors="coerce")
        ) * conv / float(monitoring_period_years)
        return out, {
            "delta_source": f"({ay} - {a0}) * {conv:.6f} / monitoring_period_years",
            "delta_units": "tC/ha/yr",
            "monitoring_period_years": float(monitoring_period_years),
            "conversion": {
                "biomass_to_carbon_fraction": float(BIOMASS_TO_CARBON_FRACTION),
                "root_to_shoot_ratio": float(ROOT_TO_SHOOT_RATIO),
            },
        }

    raise ValueError(
        "Impossibile calcolare deltaC_control_tC_ha_yr. "
        "Fornire una colonna delta-C diretta o colonne stock t0/y in tC/ha o AGB Mg/ha."
    )


# ================================================================
# UNCERTAINTY / BASELINE CALCULATION
# ================================================================

def calculate_uncbsl(control_df, project_area_ha, monitoring_period_years=MONITORING_PERIOD_YEARS,
                     daf_total=DAF_TOTAL, br_gov_y_tco2e=BR_GOV_Y_TCO2E):
    df, _, _ = standardize_ref_coords(control_df)
    df = df.drop_duplicates(subset=["_ref_lon_key", "_ref_lat_key"]).copy()
    df["deltaC_control_tC_ha_yr"] = pd.to_numeric(df["deltaC_control_tC_ha_yr"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["deltaC_control_tC_ha_yr"].notna()].copy()

    if len(df) < 2:
        raise RuntimeError("Servono almeno 2 pixel controllo validi per calcolare sigma e SE.")

    delta = df["deltaC_control_tC_ha_yr"].astype(float).values
    n_control = int(len(delta))
    mean_delta_raw = float(np.mean(delta))
    sigma_control = float(np.std(delta, ddof=1))
    se_control = float(sigma_control / math.sqrt(n_control))

    if APPLY_NO_AVOIDED_DEGRADATION_SAFEGUARD:
        mean_delta_crediting = max(mean_delta_raw, 0.0)
    else:
        mean_delta_crediting = mean_delta_raw

    if mean_delta_crediting > 0:
        uncbsl_fraction = float(1.645 * se_control / mean_delta_crediting)
        uncbsl_percent = float(uncbsl_fraction * 100.0)
        uncbsl_status = "computed"
    else:
        # Eq-33 has a zero denominator. Since BR_unadj is zero after the no-avoided-
        # degradation safeguard, the multiplicative uncertainty has no effect.
        uncbsl_fraction = 0.0
        uncbsl_percent = 0.0
        uncbsl_status = "not_applicable_zero_or_negative_baseline_rate"

    br_unadj_tco2e = float(mean_delta_crediting * float(project_area_ha) * float(monitoring_period_years) * 44.0 / 12.0)
    br_crediting_no_gov_tco2e = float(br_unadj_tco2e * (1.0 + float(daf_total)) * (1.0 + uncbsl_fraction))
    br_crediting_tco2e = float(max(br_crediting_no_gov_tco2e, float(br_gov_y_tco2e)))

    summary = {
        "N_control_unique_pixels": n_control,
        "DeltaC_control_mean_raw_tC_ha_yr": mean_delta_raw,
        "DeltaC_control_mean_for_crediting_tC_ha_yr": float(mean_delta_crediting),
        "sigma_control_tC_ha_yr": sigma_control,
        "SE_control_tC_ha_yr": se_control,
        "UNCBSL_fraction": uncbsl_fraction,
        "UNCBSL_percent": uncbsl_percent,
        "UNCBSL_status": uncbsl_status,
        "project_area_ha": float(project_area_ha),
        "monitoring_period_years": float(monitoring_period_years),
        "BR_unadj_y_tCO2e": br_unadj_tco2e,
        "DAF_total_fraction": float(daf_total),
        "BR_gov_y_tCO2e": float(br_gov_y_tco2e),
        "BR_crediting_no_gov_y_tCO2e": br_crediting_no_gov_tco2e,
        "BR_crediting_y_tCO2e": br_crediting_tco2e,
        "no_avoided_degradation_safeguard_applied": bool(
            APPLY_NO_AVOIDED_DEGRADATION_SAFEGUARD and mean_delta_raw < 0
        ),
    }
    return df, summary


# ================================================================
# PLOT
# ================================================================

def plot_delta_distribution(control_df, summary, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    v = control_df["deltaC_control_tC_ha_yr"].dropna().astype(float)
    ax.hist(v, bins=min(60, max(10, int(np.sqrt(len(v))))), alpha=0.85, edgecolor="black")
    ax.axvline(summary["DeltaC_control_mean_raw_tC_ha_yr"], linestyle="--", linewidth=1.5,
               label="mean raw ΔC")
    ax.axvline(summary["DeltaC_control_mean_for_crediting_tC_ha_yr"], linestyle=":", linewidth=1.8,
               label="crediting ΔC")
    ax.set_xlabel("Reference pixel ΔC (tC/ha/yr)")
    ax.set_ylabel("Pixel count")
    ax.set_title("GS STARR Track 1 SEMDB — control carbon stock change")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = Path(out_dir) / "control_deltaC_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    return fig, p


# ================================================================
# MAIN STEP
# ================================================================

def run_baseline_uncertainty(base_dirs=None, output_dir=None, twin_df=None,
                             control_change_df=None, extraction_report=None,
                             manifest=None, verbose=True):
    if base_dirs is None:
        root, dirs = detect_base_dirs()
    else:
        root = Path(base_dirs[0])
        dirs = {
            "root": root,
            "01": root / "01_extract",
            "02": root / "02_matching",
            "03": root / "03_twin_test",
            "04": root / "04_reference_area",
            "05": root / "05_baseline_uncertainty",
        }

    out_dir = Path(output_dir) if output_dir else dirs["05"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if extraction_report is None:
        extraction_report = load_json_if_exists(dirs["01"] / "extraction_report.json")
    if manifest is None:
        manifest = load_json_if_exists(dirs["04"] / "reference_area_FINAL_manifest.json")

    if twin_df is None:
        twin_path = find_file([dirs["03"]], "twin_tested_pixels", required=True)
        twin_df = load_df(twin_path)
    else:
        twin_path = None

    if control_change_df is None:
        search_dirs = [out_dir, dirs["04"], dirs["03"], dirs["02"]]
        carbon_path = find_file(search_dirs, "control_carbon_stock_change", required=False)
        if carbon_path is None:
            # Allow the carbon columns to already exist in twin_tested_pixels.
            control_joined = twin_df.copy()
            carbon_path = "carbon_columns_inside_twin_tested_pixels"
        else:
            control_change_df = load_df(carbon_path)
            control_joined = merge_control_change(twin_df, control_change_df)
    else:
        carbon_path = "provided_dataframe"
        control_joined = merge_control_change(twin_df, control_change_df)

    control_joined, delta_meta = compute_delta_c(control_joined, MONITORING_PERIOD_YEARS)
    project_area_ha, project_area_source = infer_project_area_ha(extraction_report, PROJECT_AREA_HA)

    control_pixels, summary = calculate_uncbsl(
        control_joined,
        project_area_ha=project_area_ha,
        monitoring_period_years=MONITORING_PERIOD_YEARS,
        daf_total=DAF_TOTAL,
        br_gov_y_tco2e=BR_GOV_Y_TCO2E,
    )

    report = {
        "run_id": RUN_ID,
        "project_name": PROJECT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": "GS STARR Track 1 SEMDB",
        "step": "05_baseline_uncertainty_UNCBSL",
        "formula": {
            "UNCBSL": "1.645 * SE_control / DeltaC_control",
            "SE_control": "sigma_control / sqrt(N_control)",
            "BR_unadj_y": "DeltaC_BSL * A_project * T_y * 44/12",
            "BR_crediting_y": "max(BR_unadj_y * (1 + DAF_total) * (1 + UNCBSL), BR_gov_y)",
        },
        "input_files": {
            "twin_tested_pixels": str(twin_path) if twin_path is not None else "provided_dataframe",
            "control_carbon_stock_change": str(carbon_path),
            "extraction_report": str(dirs["01"] / "extraction_report.json"),
            "reference_area_manifest": str(dirs["04"] / "reference_area_FINAL_manifest.json"),
        },
        "deltaC_metadata": delta_meta,
        "project_area_source": project_area_source,
        "summary": summary,
        "compliance_notes": [
            "N_control is based on unique reference pixels after dropping duplicate ref_lon/ref_lat.",
            "The full donor pool is not used for uncertainty; only locked matched control pixels are used.",
            "Negative mean DeltaC is set to zero for BR_unadj when the no-avoided-degradation safeguard is active.",
            "UNCBSL is applied as an independent conservativeness multiplier to baseline removals together with DAF.",
        ],
        "reference_area_manifest_summary": {
            "lock_status": manifest.get("lock_status"),
            "reference_area_definition": manifest.get("reference_area_definition", {}),
            "monitoring": manifest.get("monitoring", {}),
        },
    }

    control_out = save_df(control_pixels, out_dir / "control_deltaC_pixels", OUTPUT_FORMAT)
    summary_df = pd.DataFrame([summary])
    summary_csv = out_dir / "baseline_uncertainty_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    report_json = out_dir / "baseline_uncertainty_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    fig, fig_path = plot_delta_distribution(control_pixels, summary, out_dir)

    if verbose:
        print(f"\n{'=' * 70}")
        print("STEP 05 — GS STARR Track 1 SEMDB baseline uncertainty")
        print(f"RUN_ID                       : {RUN_ID}")
        print(f"Unique control pixels        : {summary['N_control_unique_pixels']:,}")
        print(f"Mean raw ΔC                  : {summary['DeltaC_control_mean_raw_tC_ha_yr']:.6f} tC/ha/yr")
        print(f"Mean crediting ΔC            : {summary['DeltaC_control_mean_for_crediting_tC_ha_yr']:.6f} tC/ha/yr")
        print(f"Sigma control                : {summary['sigma_control_tC_ha_yr']:.6f} tC/ha/yr")
        print(f"SE control                   : {summary['SE_control_tC_ha_yr']:.6f} tC/ha/yr")
        print(f"UNCBSL                       : {summary['UNCBSL_fraction']:.6f} ({summary['UNCBSL_percent']:.2f}%)")
        print(f"BR unadjusted                : {summary['BR_unadj_y_tCO2e']:.3f} tCO2e")
        print(f"BR crediting                 : {summary['BR_crediting_y_tCO2e']:.3f} tCO2e")
        print(f"Output                       : {out_dir}")
        print(f"{'=' * 70}")

    return control_pixels, summary, report, fig, out_dir


def main():
    run_baseline_uncertainty()


if __name__ == "__main__":
    main()
