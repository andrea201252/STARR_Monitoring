# -*- coding: utf-8 -*-
"""
GS STARR Track 1 SEMDB — runner up to Step 05 only.

Step 02 is intentionally called with donor_df=None.
Reason: donor_pixels_raw can contain millions of raster cells; Step 02 v5 loads
only the required columns directly from parquet/csv and then applies prefilter +
stratified cap before building KNN matrices.

Step 05 is the baseline 90% confidence interval / UNCBSL step.
It requires monitored carbon-stock change for the locked control pixels.
"""

import gc
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(run_step_05=True):
    s01 = load_module("01_STARR_raster_extract.py", "s01")
    s02 = load_module("02_STARR_matching_data_weights.py", "s02")
    s03 = load_module("03_STARR_twin_test_selection.py", "s03")
    s04 = load_module("04_STARR_reference_area_lock.py", "s04")

    proj_df, donor_df, meta, out01 = s01.run_extraction()

    # Critical RAM guard: Step 02 must reload donor from disk with selected columns only.
    del donor_df
    gc.collect()

    matched_df, weights, imp_df, smd_df, figs02, out02 = s02.run_matching_step(
        base_dirs=[out01], proj_df=proj_df, donor_df=None, meta=meta
    )

    all_pairs, twin_pixels, twin_report, figs03, out03 = s03.run_twin_test(
        base_dirs=[out02], matched_df=matched_df, proj_df=proj_df, meta=meta
    )

    # Project dataframe is no longer needed after Step 03.
    del proj_df
    gc.collect()

    bounds_gdf, mon_gdf, fig04, out04, manifest = s04.run_reference_area_lock(
        base_dirs=[out03], passed_df=twin_pixels, meta=meta, twin_report=twin_report
    )

    if not run_step_05:
        print("Stopped after Step 04. Step 05 requires control carbon-stock change data.")
        return {
            "01_extract": out01,
            "02_matching": out02,
            "03_twin_test": out03,
            "04_reference_area": out04,
        }

    s05 = load_module("05_STARR_baseline_confidence_interval_UNCBSL.py", "s05")
    control_pixels, summary, report, fig05, out05 = s05.run_baseline_ci_uncbsl()
    return {
        "01_extract": out01,
        "02_matching": out02,
        "03_twin_test": out03,
        "04_reference_area": out04,
        "05_baseline_CI_UNCBSL": out05,
        "ci90_uncbsl_summary": summary,
    }


if __name__ == "__main__":
    main(run_step_05=True)
