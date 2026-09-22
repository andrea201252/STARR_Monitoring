# -*- coding: utf-8 -*-
"""
GS STARR Track 1 SEMDB — parametric runner (Steps 01-05).

Key parameter: DONOR_EXTENT_KM
─────────────────────────────────
Controls the extent of the donor pool around the PA border.
It is passed to Step 01 which applies the I/O clip BEFORE loading the rasters.
Each extent generates an isolated output directory (run_id includes the suffix).

  "full"  → no clip (original behavior)
  5       → 5 km buffer
  10      → 10 km buffer  (recommended starting point)
  20      → 20 km buffer
  30      → 30 km buffer
  40      → 40 km buffer
  50      → 50 km buffer
  60      → 60 km buffer

Step 02 is intentionally called with donor_df=None.
Step 05 requires the monitored carbon stock change (controlled externally).
"""

import gc
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════
# CONFIG — the only section to modify between one run and another
# ══════════════════════════════════════════════════════════════════════

DONOR_EXTENT_KM: float | str = 10   # "full" | 5 | 10 | 20 | 30 | 40 | 50 | 60

# Base run identifier (used for TIF file names and output directory).
# Must match the prefix of the TIFs exported by GEE:
#   covariates_project_<RUN_ID_BASE>*.tif
#   covariates_donor_<RUN_ID_BASE>*.tif
# Leave "" to find any covariates_project_*.tif TIF in the folder.
RUN_ID_BASE: str = ""   # e.g. "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster"

# Directory where the GEE TIFs are located (may contain subfolders).
# Add all relevant paths; the first existing one is used as the output base.
BASE_DIR: str = ""  # e.g. "/path/to/my/data"

# Optional shapefiles for the spatial filter (set to None to disable).
FNF_SHAPEFILE:      str | None = None  # e.g. "/path/to/FNF18_fullBuffer.shp"
ELIGIBLE_SHAPEFILE: str | None = None  # e.g. "/path/to/Eligible_FNF_fullBuffer.shp"

# Eligibility toggle on the donor. GS does not require eligibility over the ENTIRE
# area of the donor pool; we apply it for conservativeness.
#   True  → apply the Eligible_FNF filter to the donor (conservative, default).
#   False → the donor uses the entire non-forest area (GS minimum).
USE_ELIGIBILITY: bool = True

# ══════════════════════════════════════════════════════════════════════


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(run_step_05: bool = True,
         donor_extent_km: float | str = DONOR_EXTENT_KM,
         run_id_base: str = RUN_ID_BASE,
         base_dir: str = BASE_DIR,
         fnf_shapefile: str | None = FNF_SHAPEFILE,
         eligible_shapefile: str | None = ELIGIBLE_SHAPEFILE,
         use_eligibility: bool = USE_ELIGIBILITY,
         outputs_dirname: str | None = None) -> dict:
    """
    Runs the complete STARR pipeline with the chosen donor extent.

    Parameters
    ---------
    run_step_05 : bool
        If False, stops after Step 04 (useful for 00b_compare_extents).
    donor_extent_km : float | "full"
        Donor extent. Default = CONFIG value at the top of the file.
    run_id_base : str
        Base run identifier (GEE TIF prefix). Default = RUN_ID_BASE.
    base_dir : str
        Root directory where the TIFs are located. Default = BASE_DIR.
    fnf_shapefile : str | None
        FNF18 shapefile path. None = filter disabled.
    eligible_shapefile : str | None
        Eligible_FNF shapefile path. None = filter disabled.
    use_eligibility : bool
        If True (default) applies the Eligible_FNF filter to the donor
        (conservative). If False the donor uses the entire non-forest area
        (GS does not require eligibility over the whole donor pool).

    Returns
    -----------
    dict with the output path of each step and key metadata.
    """
    s01 = load_module("01_STARR_raster_extract.py",        "s01")
    s02 = load_module("02_STARR_matching_data_weights.py",  "s02")
    s03 = load_module("03_STARR_twin_test_selection.py",    "s03")
    s04 = load_module("04_STARR_reference_area_lock.py",    "s04")

    # Propagates the output folder name (editable from the notebook / from 00b) to
    # the fresh modules loaded HERE. main() reloads its own copies of s01..s05,
    # so setting s01.OUTPUTS_DIRNAME from the notebook is NOT enough: it must be passed as
    # a parameter. Step 02/03/04 inherit the level from Step 01 (base_dirs[0].parent).
    if outputs_dirname is not None:
        s01.OUTPUTS_DIRNAME = outputs_dirname

    # Builds the list of base directories from the single CONFIG path
    _base_dirs = [base_dir] if base_dir else []

    # ── STEP 01 ───────────────────────────────────────────────────────
    proj_df, donor_df, meta, out01 = s01.run_extraction(
        base_dirs=_base_dirs,
        donor_extent_km=donor_extent_km,
        run_id_base=run_id_base,
        fnf_shapefile=fnf_shapefile,
        eligible_shapefile=eligible_shapefile,
        use_eligibility=use_eligibility,
    )

    # RAM protection: Step 02 re-reads the donor from disk with only the
    # necessary columns + stratified cap. No need to keep it in RAM.
    del donor_df
    gc.collect()

    # ── STEP 02 ───────────────────────────────────────────────────────
    # base_dirs=[out01] → finds donor_pixels_raw.parquet in the dir
    # isolated for this extent. No logic change.
    matched_df, weights, imp_df, smd_df, figs02, out02 = s02.run_matching_step(
        base_dirs=[out01], proj_df=proj_df, donor_df=None, meta=meta,
    )

    # ── STEP 03 ───────────────────────────────────────────────────────
    all_pairs, twin_pixels, twin_report, figs03, out03 = s03.run_twin_test(
        base_dirs=[out02], matched_df=matched_df, proj_df=proj_df, meta=meta,
    )

    del proj_df
    gc.collect()

    # ── STEP 04 ───────────────────────────────────────────────────────
    bounds_gdf, mon_gdf, fig04, out04, manifest = s04.run_reference_area_lock(
        base_dirs=[out03], passed_df=twin_pixels, meta=meta,
        twin_report=twin_report,
    )

    # Aggregate metrics for the 00b comparator
    _ext = meta.get("donor_extent_km", donor_extent_km)
    summary = {
        "donor_extent_km":       _ext,
        "run_id":                meta.get("run_id", ""),
        "donor_n":               int(meta.get("donor_n", 0)),
        "project_n":             int(meta.get("project_n", 0)),
        "ratio_donor_project":   meta.get("ratio_donor_project", None),
        "ratio_donor_project_area": meta.get("ratio_donor_project_area", None),
        "meets_3x_guideline":    meta.get("meets_3x_guideline", None),
        "meets_3x_guideline_area": meta.get("meets_3x_guideline_area", None),
        "matched_n":             int(len(matched_df)),
        "match_coverage_pct":    round(
            len(matched_df) / max(1, len(all_pairs)) * 100, 2),
        "smd_max":               float(smd_df["SMD"].max()) if not smd_df.empty else None,
        "smd_all_passed":        bool(smd_df["passed"].all()) if not smd_df.empty else None,
        "twin_pass_n":           int(len(twin_pixels)),
        "twin_pass_pct":         round(
            len(twin_pixels) / max(1, len(all_pairs)) * 100, 2),
        "twin_aggregate_passed": twin_report.get("aggregate_twin_passed", None),
        "twin_test_compliant":   twin_report.get("twin_test_compliant", None),
        "reference_area_ha":     float(manifest.get("reference_area_definition", {})
                                       .get("total_ha", 0)),
        "out01": str(out01),
        "out02": str(out02),
        "out03": str(out03),
        "out04": str(out04),
    }

    if not run_step_05:
        print(f"\n[STOP] Stopped after Step 04. "
              f"Step 05 requires the carbon stock change data.")
        return {
            "01_extract":        out01,
            "02_matching":       out02,
            "03_twin_test":      out03,
            "04_reference_area": out04,
            "summary":           summary,
        }

    s05 = load_module(
        "05_STARR_baseline_confidence_interval_UNCBSL.py", "s05")
    if outputs_dirname is not None:
        s05.OUTPUTS_DIRNAME = outputs_dirname
    # Step 05 returns 6 values: control_pixels, project_pixels, summary, report, figs, out_dir
    control_pixels, project_pixels, ci_summary, report, fig05, out05 = \
        s05.run_baseline_ci_uncbsl()

    # BL_unadj,y = ΔC_ref,y × A_project  (GS STARR Eq 31a) — absolute tCO2e, NOT fractions.
    # Keys produced by build_unadjusted_baseline_summary (exposed by the Step 05 summary dict).
    summary["BL_unadj_y_tCO2e"]       = ci_summary.get("BL_unadj_y_tCO2e")
    summary["BL_unadj_period_tCO2e"]  = ci_summary.get("BL_unadj_period_tCO2e")
    summary["delta_C_ref_y_tC_ha_yr"] = ci_summary.get("delta_C_ref_y_tC_ha_yr")
    summary["project_area_ha"]        = ci_summary.get("project_area_ha_used")
    # CI extremes (tC/ha/yr at pixel level — per-unit-area values for diagnostics)
    summary["ci90_lower_tC_ha_yr"]    = ci_summary.get(
        "ci90_lower_final_tC_ha_yr", ci_summary.get("ci90_lower_tC_ha_yr"))
    summary["ci90_mean_tC_ha_yr"]     = ci_summary.get("mean_deltaC_control_tC_ha_yr")
    summary["ci90_upper_tC_ha_yr"]    = ci_summary.get(
        "ci90_upper_final_tC_ha_yr", ci_summary.get("ci90_upper_tC_ha_yr"))
    summary["uncbsl_fraction"]        = ci_summary.get("uncbsl_fraction")
    summary["uncbsl_status"]          = ci_summary.get("uncbsl_status")
    summary["out05"]                  = str(out05)

    return {
        "01_extract":          out01,
        "02_matching":         out02,
        "03_twin_test":        out03,
        "04_reference_area":   out04,
        "05_baseline_CI":      out05,
        "ci90_uncbsl_summary": ci_summary,
        "summary":             summary,
    }


if __name__ == "__main__":
    main(run_step_05=True)