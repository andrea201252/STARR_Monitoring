# -*- coding: utf-8 -*-
"""
GS STARR Track 1 SEMDB — runner parametrico (Steps 01-05).

Parametro chiave: DONOR_EXTENT_KM
─────────────────────────────────
Controlla l'estensione del pool donor attorno al bordo della PA.
Viene passato a Step 01 che applica il clip I/O PRIMA di caricare i raster.
Ogni extent genera una directory output isolata (run_id include il suffisso).

  "full"  → nessun clip (comportamento originale)
  5       → buffer 5 km
  10      → buffer 10 km  (punto di partenza raccomandato)
  20      → buffer 20 km
  30      → buffer 30 km

Step 02 viene chiamato intenzionalmente con donor_df=None.
Step 05 richiede la variazione dello stock di carbonio monitorata (controllata esternamente).
"""

import gc
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════
# CONFIG — unica sezione da modificare tra un run e l'altro
# ══════════════════════════════════════════════════════════════════════

DONOR_EXTENT_KM: float | str = 10   # "full" | 5 | 10 | 20 | 30

# Identificatore base del run (usato per nomi file TIF e directory output).
# Deve corrispondere al prefisso dei TIF esportati da GEE:
#   covariates_project_<RUN_ID_BASE>*.tif
#   covariates_donor_<RUN_ID_BASE>*.tif
# Lasciare "" per trovare qualsiasi TIF covariates_project_*.tif nella cartella.
RUN_ID_BASE: str = ""   # es. "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster"

# Directory dove si trovano i TIF GEE (può contenere sottocartelle).
# Aggiungere tutti i percorsi rilevanti; il primo esistente viene usato come base output.
BASE_DIR: str = ""  # es. "/path/to/my/data"

# Shapefile opzionali per il filtro spaziale (impostare a None per disattivare).
FNF_SHAPEFILE:      str | None = None  # es. "/path/to/FNF18_fullBuffer.shp"
ELIGIBLE_SHAPEFILE: str | None = None  # es. "/path/to/Eligible_FNF_fullBuffer.shp"

# Toggle eligibility sul donor. GS non richiede eleggibilità su TUTTA l'area
# del pool donor; lo applichiamo per conservatività.
#   True  → applica il filtro Eligible_FNF al donor (conservativo, default).
#   False → il donor usa l'intera area non-forest (minimo GS).
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
         use_eligibility: bool = USE_ELIGIBILITY) -> dict:
    """
    Esegue la pipeline STARR completa con l'estensione donor scelta.

    Parametri
    ---------
    run_step_05 : bool
        Se False, si ferma dopo Step 04 (utile per 00b_compare_extents).
    donor_extent_km : float | "full"
        Estensione donor. Default = valore CONFIG in cima al file.
    run_id_base : str
        Identificatore base del run (prefisso TIF GEE). Default = RUN_ID_BASE.
    base_dir : str
        Directory radice dove si trovano i TIF. Default = BASE_DIR.
    fnf_shapefile : str | None
        Path shapefile FNF18. None = filtro disattivato.
    eligible_shapefile : str | None
        Path shapefile Eligible_FNF. None = filtro disattivato.
    use_eligibility : bool
        Se True (default) applica il filtro Eligible_FNF al donor
        (conservativo). Se False il donor usa l'intera area non-forest
        (GS non richiede eleggibilità su tutto il pool donor).

    Restituisce
    -----------
    dict con path di output di ogni step e metadati chiave.
    """
    s01 = load_module("01_STARR_raster_extract.py",        "s01")
    s02 = load_module("02_STARR_matching_data_weights.py",  "s02")
    s03 = load_module("03_STARR_twin_test_selection.py",    "s03")
    s04 = load_module("04_STARR_reference_area_lock.py",    "s04")

    # Costruisce la lista di directory base dal singolo path CONFIG
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

    # Protezione RAM: Step 02 rilegge il donor da disk con sole le colonne
    # necessarie + cap stratificato. Non serve tenerlo in RAM.
    del donor_df
    gc.collect()

    # ── STEP 02 ───────────────────────────────────────────────────────
    # base_dirs=[out01] → trova donor_pixels_raw.parquet nella dir
    # isolata per questo extent. Nessuna modifica logica.
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

    # Metriche aggregate per il comparatore 00b
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
        print(f"\n[STOP] Fermato dopo Step 04. "
              f"Step 05 richiede i dati di variazione dello stock di carbonio.")
        return {
            "01_extract":        out01,
            "02_matching":       out02,
            "03_twin_test":      out03,
            "04_reference_area": out04,
            "summary":           summary,
        }

    s05 = load_module(
        "05_STARR_baseline_confidence_interval_UNCBSL.py", "s05")
    # Step 05 restituisce 6 valori: control_pixels, project_pixels, summary, report, figs, out_dir
    control_pixels, project_pixels, ci_summary, report, fig05, out05 = \
        s05.run_baseline_ci_uncbsl()

    # BL_unadj,y = ΔC_ref,y × A_project  (GS STARR Eq 31a) — tCO2e assoluti, NON frazioni.
    # Chiavi prodotte da build_unadjusted_baseline_summary (esposte dal dict summary di Step 05).
    summary["BL_unadj_y_tCO2e"]       = ci_summary.get("BL_unadj_y_tCO2e")
    summary["BL_unadj_period_tCO2e"]  = ci_summary.get("BL_unadj_period_tCO2e")
    summary["delta_C_ref_y_tC_ha_yr"] = ci_summary.get("delta_C_ref_y_tC_ha_yr")
    summary["project_area_ha"]        = ci_summary.get("project_area_ha_used")
    # Estremi CI (tC/ha/yr a livello di pixel — valori per unità di area per la diagnostica)
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