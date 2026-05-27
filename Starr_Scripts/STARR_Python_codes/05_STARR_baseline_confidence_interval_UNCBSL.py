# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB | 05_STARR_baseline_confidence_interval_UNCBSL.py
STEP 05 — Baseline Confidence Interval / UNCBSL
============================================================

Purpose
-------
Calculates the 90% confidence interval of the baseline carbon-stock change
for Track 1 SEMDB and expresses it as UNCBSL.

This step intentionally stops at baseline uncertainty. It does NOT calculate:
  - activity removals,
  - UNCAR,
  - leakage,
  - net verified removals,
  - final issuable GSVERs.

Method
------
For the locked matched control/reference pixels:

    SE_control = sigma_control / sqrt(N_control)
    CI90_abs   = 1.645 * SE_control
    UNCBSL     = CI90_abs / mean_deltaC_control

Where:
  sigma_control = standard deviation of ΔC in the locked reference pixels
  N_control     = number of UNIQUE locked reference/control pixels

Critical implementation choices
-------------------------------
1. N_control counts unique reference pixels only.
   If the same donor/reference pixel appears more than once, duplicates are
   dropped by rounded reference coordinates before computing sigma and SE.

2. The full donor pool is never used for UNCBSL.
   UNCBSL is calculated only from the locked, twin-tested matched controls.

3. ΔC must be expressed in tC/ha/year.
   The script accepts either a direct ΔC column or stock columns from which
   annualized ΔC is computed.

4. If mean ΔC <= 0, the confidence interval is still reported, but UNCBSL is
   flagged as not applicable for baseline-removal crediting because the
   denominator is zero/negative. This avoids creating removals from avoided
   degradation.

Expected input
--------------
The script automatically looks for:
  STARR_outputs/<RUN_ID>/03_twin_test/twin_tested_pixels.parquet|csv
  STARR_outputs/<RUN_ID>/04_reference_area/reference_area_FINAL_manifest.json

It also needs monitored carbon stock change for the locked reference pixels.
Provide one of:

A) File named control_carbon_stock_change.parquet/csv in one of:
     STARR_outputs/<RUN_ID>/05_baseline_CI_UNCBSL/
     STARR_outputs/<RUN_ID>/04_reference_area/
     STARR_outputs/<RUN_ID>/03_twin_test/
     STARR_outputs/<RUN_ID>/02_matching/

B) Carbon-change columns already inside twin_tested_pixels.

C) Direct call:
     run_baseline_ci_uncbsl(control_change_df=your_dataframe)

Accepted carbon-change schemas
------------------------------
Direct annual ΔC column, any one of:
  deltaC_control_tC_ha_yr
  deltaC_ref_tC_ha_yr
  deltaC_tC_ha_yr
  delta_C_tC_ha_yr
  dC_tC_ha_yr

Stock columns in tC/ha, any one pair:
  C_ref_t0_tC_ha + C_ref_y_tC_ha
  C_control_t0_tC_ha + C_control_y_tC_ha
  carbon_t0_tC_ha + carbon_y_tC_ha
  C_t0_tC_ha + C_y_tC_ha

AGB columns in Mg dry matter/ha, fallback:
  AGB_ref_t0_Mg_ha + AGB_ref_y_Mg_ha
  AGB_control_t0_Mg_ha + AGB_control_y_Mg_ha
  agb_t0_Mg_ha + agb_y_Mg_ha
  AGB_t0_Mg_ha + AGB_y_Mg_ha

When AGB is used:
  C = AGB * BIOMASS_TO_CARBON_FRACTION * (1 + ROOT_TO_SHOOT_RATIO)

Outputs
-------
  05_baseline_CI_UNCBSL/
    baseline_control_deltaC_distribution.parquet|csv
    baseline_CI90_UNCBSL_summary.csv
    baseline_CI90_UNCBSL_report.json
    deltaC_control_CI90.png
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

RUN_ID = "Idiofa_Lobi_2018_buf20km_excl5km_WRB2_v06_raster"
PROJECT_NAME = "Idiofa_Lobi"

BASE_DIR_CANDIDATES = [
    "/content/drive/MyDrive/STARR_Idiofa_New_V2",
    "/content/content/MyDrive/STARR_Idiofa_New_V2",
]

OUTPUT_FORMAT = "parquet"
STRICT_LOCK_REQUIRED = True

# If C_t0 is 2018 and C_y is 2024, set MONITORING_PERIOD_YEARS = 6.
# If you provide deltaC_*_tC_ha_yr directly, this value is only reported.
MONITORING_PERIOD_YEARS = 1.0

# Formula uses 1.645 for a two-sided 90% normal CI.
CI90_Z_VALUE = 1.645

# AGB fallback conversion settings.
BIOMASS_TO_CARBON_FRACTION = 0.47
ROOT_TO_SHOOT_RATIO = 0.0

# Unique control-pixel key. Rounded coordinates avoid floating precision noise.
COORD_ROUND_DECIMALS = 7

# Minimum usable unique controls. Methodologically, this should be much higher;
# this threshold only prevents invalid standard error calculation.
MIN_CONTROL_PIXELS = 2


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
    raise ValueError(f"Unsupported format: {p.suffix}")


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
        raise FileNotFoundError(f"'{stem}' not found in {[str(b) for b in base_dirs]}")
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


def load_json_if_exists(path):
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ================================================================
# COORDINATES / JOIN
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


def merge_control_change(twin_df, control_change_df):
    t, _, _ = standardize_ref_coords(twin_df)
    c, _, _ = standardize_ref_coords(control_change_df)

    c = c.drop_duplicates(subset=["_ref_lon_key", "_ref_lat_key"]).copy()

    # If carbon columns exist in both, use the explicitly supplied control dataset.
    supplied_cols = [col for col in c.columns if col not in ("_ref_lon_key", "_ref_lat_key")]
    duplicates = [col for col in supplied_cols if col in t.columns]
    if duplicates:
        t = t.drop(columns=duplicates, errors="ignore")

    return t.merge(c, on=["_ref_lon_key", "_ref_lat_key"], how="left", suffixes=("", "_carbon"))


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
        raise ValueError("MONITORING_PERIOD_YEARS must be > 0.")

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
                "delta_units": "tC/ha/year",
                "monitoring_period_years": float(monitoring_period_years),
                "annualization": "already annualized",
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
            "delta_source": f"({cy} - {c0}) / MONITORING_PERIOD_YEARS",
            "delta_units": "tC/ha/year",
            "monitoring_period_years": float(monitoring_period_years),
            "annualization": "computed from stock difference",
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
            "delta_source": f"({ay} - {a0}) * {conv:.6f} / MONITORING_PERIOD_YEARS",
            "delta_units": "tC/ha/year",
            "monitoring_period_years": float(monitoring_period_years),
            "annualization": "computed from AGB stock difference",
            "conversion": {
                "biomass_to_carbon_fraction": float(BIOMASS_TO_CARBON_FRACTION),
                "root_to_shoot_ratio": float(ROOT_TO_SHOOT_RATIO),
                "combined_factor": float(conv),
            },
        }

    raise ValueError(
        "Cannot calculate deltaC_control_tC_ha_yr. Provide a direct delta-C column, "
        "stock t0/y columns in tC/ha, or AGB t0/y columns in Mg/ha."
    )


# ================================================================
# CI90 / UNCBSL CALCULATION
# ================================================================

def calculate_ci90_uncbsl(control_df):
    df, lon_col, lat_col = standardize_ref_coords(control_df)
    n_before = int(len(df))

    df = df.drop_duplicates(subset=["_ref_lon_key", "_ref_lat_key"]).copy()
    n_after_dedup = int(len(df))

    df["deltaC_control_tC_ha_yr"] = pd.to_numeric(df["deltaC_control_tC_ha_yr"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["deltaC_control_tC_ha_yr"].notna()].copy()

    if len(df) < MIN_CONTROL_PIXELS:
        raise RuntimeError(
            f"At least {MIN_CONTROL_PIXELS} valid unique control pixels are required. Found {len(df)}."
        )

    delta = df["deltaC_control_tC_ha_yr"].astype(float).values

    n_control = int(len(delta))
    mean_delta = float(np.mean(delta))
    median_delta = float(np.median(delta))
    sigma_control = float(np.std(delta, ddof=1))
    se_control = float(sigma_control / math.sqrt(n_control))
    ci90_abs = float(CI90_Z_VALUE * se_control)
    ci90_lower = float(mean_delta - ci90_abs)
    ci90_upper = float(mean_delta + ci90_abs)

    if mean_delta > 0:
        uncbsl_fraction = float(ci90_abs / mean_delta)
        uncbsl_percent = float(uncbsl_fraction * 100.0)
        uncbsl_status = "computed"
        crediting_note = "UNCBSL can be used as baseline uncertainty deduction."
    else:
        uncbsl_fraction = np.nan
        uncbsl_percent = np.nan
        uncbsl_status = "not_applicable_mean_deltaC_zero_or_negative"
        crediting_note = (
            "Mean control ΔC is <= 0. UNCBSL denominator is invalid for baseline-removal "
            "crediting; baseline removals should not be generated from avoided degradation."
        )

    summary = {
        "n_rows_before_deduplication": n_before,
        "n_rows_after_coordinate_deduplication": n_after_dedup,
        "n_control_unique_valid_pixels": n_control,
        "mean_deltaC_control_tC_ha_yr": mean_delta,
        "median_deltaC_control_tC_ha_yr": median_delta,
        "std_deltaC_control_tC_ha_yr": sigma_control,
        "se_control_tC_ha_yr": se_control,
        "ci90_z_value": float(CI90_Z_VALUE),
        "ci90_abs_tC_ha_yr": ci90_abs,
        "ci90_lower_tC_ha_yr": ci90_lower,
        "ci90_upper_tC_ha_yr": ci90_upper,
        "uncbsl_fraction": None if np.isnan(uncbsl_fraction) else uncbsl_fraction,
        "uncbsl_percent": None if np.isnan(uncbsl_percent) else uncbsl_percent,
        "uncbsl_status": uncbsl_status,
        "crediting_note": crediting_note,
        "coordinate_key_round_decimals": int(COORD_ROUND_DECIMALS),
        "coordinate_columns_used": {"lon": lon_col, "lat": lat_col},
    }

    return df, summary


# ================================================================
# PLOT
# ================================================================

def plot_delta_ci(control_df, summary, out_dir):
    v = control_df["deltaC_control_tC_ha_yr"].dropna().astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(v, bins=min(60, max(10, int(np.sqrt(len(v))))), alpha=0.85, edgecolor="black")
    ax.axvline(summary["mean_deltaC_control_tC_ha_yr"], linestyle="--", linewidth=1.5,
               label="mean ΔC")
    ax.axvline(summary["ci90_lower_tC_ha_yr"], linestyle=":", linewidth=1.5,
               label="90% CI lower")
    ax.axvline(summary["ci90_upper_tC_ha_yr"], linestyle=":", linewidth=1.5,
               label="90% CI upper")
    ax.set_xlabel("Locked control/reference pixel ΔC (tC/ha/year)")
    ax.set_ylabel("Pixel count")
    ax.set_title("GS STARR Track 1 SEMDB — baseline ΔC 90% confidence interval")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = Path(out_dir) / "deltaC_control_CI90.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig, out_path


# ================================================================
# MAIN STEP
# ================================================================

def run_baseline_ci_uncbsl(base_dirs=None, output_dir=None, twin_df=None,
                           control_change_df=None, manifest=None, verbose=True):
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
            "05": root / "05_baseline_CI_UNCBSL",
        }

    out_dir = Path(output_dir) if output_dir else dirs["05"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if manifest is None:
        manifest = load_json_if_exists(dirs["04"] / "reference_area_FINAL_manifest.json")

    if STRICT_LOCK_REQUIRED and manifest and manifest.get("lock_status") != "LOCKED":
        raise RuntimeError("STEP 05 blocked: reference area is not LOCKED in Step 04 manifest.")

    if twin_df is None:
        twin_path = find_file([dirs["03"]], "twin_tested_pixels", required=True)
        twin_df = load_df(twin_path)
    else:
        twin_path = "provided_dataframe"

    if control_change_df is None:
        search_dirs = [out_dir, dirs["04"], dirs["03"], dirs["02"]]
        carbon_path = find_file(search_dirs, "control_carbon_stock_change", required=False)
        if carbon_path is None:
            control_joined = twin_df.copy()
            carbon_path = "carbon_columns_inside_twin_tested_pixels"
        else:
            control_change_df = load_df(carbon_path)
            control_joined = merge_control_change(twin_df, control_change_df)
    else:
        carbon_path = "provided_dataframe"
        control_joined = merge_control_change(twin_df, control_change_df)

    control_joined, delta_meta = compute_delta_c(control_joined, MONITORING_PERIOD_YEARS)
    control_pixels, summary = calculate_ci90_uncbsl(control_joined)

    control_out = save_df(control_pixels, out_dir / "baseline_control_deltaC_distribution", OUTPUT_FORMAT)

    summary_csv = out_dir / "baseline_CI90_UNCBSL_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    report = {
        "run_id": RUN_ID,
        "project_name": PROJECT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": "GS STARR Track 1 SEMDB",
        "step": "05_baseline_confidence_interval_UNCBSL",
        "purpose": "Calculate 90% confidence interval and UNCBSL for locked matched control/reference pixels.",
        "formula": {
            "SE_control": "sigma_control / sqrt(N_control)",
            "CI90_abs": "1.645 * SE_control",
            "UNCBSL": "CI90_abs / mean_deltaC_control",
        },
        "input_files": {
            "twin_tested_pixels": str(twin_path),
            "control_carbon_stock_change": str(carbon_path),
            "reference_area_manifest": str(dirs["04"] / "reference_area_FINAL_manifest.json"),
        },
        "deltaC_metadata": delta_meta,
        "summary": summary,
        "compliance_notes": [
            "N_control is based on unique locked reference/control pixels after coordinate deduplication.",
            "The full donor pool is not used for UNCBSL.",
            "This step only calculates baseline CI90 and UNCBSL; it does not calculate activity removals, UNCAR, leakage, net removals, or final GSVERs.",
            "If mean control ΔC <= 0, UNCBSL is reported as not applicable for baseline-removal crediting.",
        ],
        "reference_area_manifest_summary": {
            "lock_status": manifest.get("lock_status"),
            "reference_area_definition": manifest.get("reference_area_definition", {}),
            "monitoring": manifest.get("monitoring", {}),
        },
        "outputs": {
            "control_deltaC_distribution": str(control_out),
            "summary_csv": str(summary_csv),
            "report_json": str(out_dir / "baseline_CI90_UNCBSL_report.json"),
            "figure": str(out_dir / "deltaC_control_CI90.png"),
        },
    }

    report_json = out_dir / "baseline_CI90_UNCBSL_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    fig, fig_path = plot_delta_ci(control_pixels, summary, out_dir)

    if verbose:
        print(f"\n{'=' * 72}")
        print("STEP 05 — Baseline Confidence Interval / UNCBSL")
        print(f"RUN_ID                         : {RUN_ID}")
        print(f"Unique valid control pixels    : {summary['n_control_unique_valid_pixels']:,}")
        print(f"Mean ΔC control                : {summary['mean_deltaC_control_tC_ha_yr']:.6f} tC/ha/year")
        print(f"Std ΔC control                 : {summary['std_deltaC_control_tC_ha_yr']:.6f} tC/ha/year")
        print(f"SE control                     : {summary['se_control_tC_ha_yr']:.6f} tC/ha/year")
        print(f"CI90 absolute                  : ±{summary['ci90_abs_tC_ha_yr']:.6f} tC/ha/year")
        print(f"CI90 range                     : [{summary['ci90_lower_tC_ha_yr']:.6f}, {summary['ci90_upper_tC_ha_yr']:.6f}] tC/ha/year")
        if summary['uncbsl_fraction'] is None:
            print(f"UNCBSL                         : NOT APPLICABLE ({summary['uncbsl_status']})")
        else:
            print(f"UNCBSL                         : {summary['uncbsl_fraction']:.6f} ({summary['uncbsl_percent']:.2f}%)")
        print(f"Output                         : {out_dir}")
        print(f"{'=' * 72}")

    return control_pixels, summary, report, fig, out_dir


# Backward-compatible alias for old notebooks.
def run_baseline_uncertainty(*args, **kwargs):
    return run_baseline_ci_uncbsl(*args, **kwargs)


def main():
    run_baseline_ci_uncbsl()


if __name__ == "__main__":
    main()
