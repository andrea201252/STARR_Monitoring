# -*- coding: utf-8 -*-
"""
STARR_SEMDB.pyt — ArcGIS Pro Python Toolbox
GS STARR Track 1 SEMDB — Steps 01–05 (Unadjusted Baseline + CI90 / UNCBSL)

This is the ArcGIS Pro port of the Colab pipeline. The heavy statistical core
(Step 02 matching, Step 03 twin test, Step 05 baseline/CI/UNCBSL) is IMPORTED
UNCHANGED from the same step modules used in Colab (folder ./steps), so the
numbers are identical. Only the raster/vector I/O uses GDAL/pyproj (bundled with
ArcGIS Pro) instead of rasterio/geopandas — no extra installation required.

The GEE covariate/AGB rasters are prepared exactly as before (the JavaScript
export is unchanged); this tool starts from those GeoTIFFs.

STATUS: first ArcGIS build (v0.1) — validate in ArcGIS Pro and report any arcpy
message; the statistical results match the Colab pipeline by construction.
"""

import os
import sys
import json
import importlib.util
import traceback

import arcpy


# ── module loading ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_STEPS = os.path.join(_HERE, "steps")


def _load(mod_name, filename, folder=_HERE):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise RuntimeError(f"Required file not found next to the toolbox: {path}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Tee:
    """Send every print() from the step modules to BOTH the ArcGIS tool dialog
    (arcpy.AddMessage) and a persistent .log file, line by line."""
    def __init__(self, logfile=None):
        self._buf = ""
        self._fh = logfile
    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            try:
                arcpy.AddMessage(line)
            except Exception:
                pass
            if self._fh:
                try:
                    self._fh.write(line + "\n"); self._fh.flush()
                except Exception:
                    pass
    def flush(self):
        if self._buf:
            try:
                arcpy.AddMessage(self._buf)
            except Exception:
                pass
            if self._fh:
                try:
                    self._fh.write(self._buf)
                except Exception:
                    pass
            self._buf = ""


def _win_long(path):
    """Windows MAX_PATH (260 char) guard. Deep Google-Drive / shared-drive output
    folders plus a long run id easily exceed 260 characters, which makes
    os.makedirs / file writes fail with WinError 3. On Windows, for a path that
    risks exceeding the limit, return the extended-length form (\\\\?\\ prefix)
    which lifts the 260-char cap; short paths are returned unchanged."""
    if os.name != "nt":
        return path
    p = os.path.abspath(path)
    if p.startswith("\\\\?\\"):
        return p
    if len(p) < 200:            # comfortably short → keep the plain path
        return p
    if p.startswith("\\\\"):    # UNC path \\server\share\...
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


def _win_short(path):
    """Strip a \\\\?\\ extended-length prefix. arcpy geoprocessing tools do NOT
    accept extended-length paths, so anything handed to arcpy must be plain."""
    if not path:
        return path
    p = str(path)
    if p.startswith("\\\\?\\UNC\\"):
        return "\\\\" + p[len("\\\\?\\UNC\\"):]
    if p.startswith("\\\\?\\"):
        return p[len("\\\\?\\"):]
    return p


def _open_log(out_dir, name):
    """Create <out_dir>/<name>.log and return an open file handle (or None)."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        import time as _t
        path = os.path.join(out_dir, f"{name}_{_t.strftime('%Y%m%d_%H%M%S')}.log")
        fh = open(path, "w", encoding="utf-8")
        arcpy.AddMessage(f"Log file: {path}")
        return fh
    except Exception as e:
        arcpy.AddWarning(f"Could not create log file: {e}")
        return None


def _install_parquet_fallback():
    """ArcGIS Pro usually ships pyarrow; if not, make DataFrame.to_parquet a safe
    no-op-to-CSV so the pipeline (which writes both parquet and csv) never aborts."""
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        import pandas as pd
        _orig = pd.DataFrame.to_parquet
        def _safe(self, path=None, *a, **k):
            try:
                return _orig(self, path, *a, **k)
            except Exception:
                if path:
                    self.to_csv(str(path).rsplit(".", 1)[0] + ".csv", index=False)
                return None
        pd.DataFrame.to_parquet = _safe
        return "csv_fallback"


# ═════════════════════════════════════════════════════════════════════════════
class Toolbox(object):
    def __init__(self):
        self.label = "STARR SEMDB (GS Track 1)"
        self.alias = "STARR_SEMDB"
        self.tools = [STARRBaselineTool, STARRStep05Tool]


class STARRBaselineTool(object):
    def __init__(self):
        self.label = "STARR SEMDB Baseline (Steps 01-05)"
        self.description = ("GS STARR Track 1 SEMDB: covariate extraction, "
                            "hard-caliper Mahalanobis KNN matching, twin/parallel "
                            "trends test, reference-area lock, and raster-based "
                            "unadjusted baseline with CI90 / UNCBSL.")
        self.canRunInBackground = False

    # ── parameters (mirror the Colab §1 form) ────────────────────────────────
    def getParameterInfo(self):
        P = []

        def add(name, label, dtype, default=None, optional=False, direction="Input",
                ptype="Required", filter_list=None):
            p = arcpy.Parameter(displayName=label, name=name, datatype=dtype,
                                parameterType=("Optional" if optional else ptype),
                                direction=direction)
            if default is not None:
                p.value = default
            if filter_list is not None:
                p.filter.type = "ValueList"
                p.filter.list = filter_list
            P.append(p)
            return p

        # identity
        add("project_name", "Project name", "GPString", "Idiofa_Lobi")
        add("run_id_base", "Run ID (base, no path/suffix)", "GPString",
            "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v09_raster")
        # covariate rasters (already exported by GEE)
        add("project_raster", "Project covariate raster (GeoTIFF stack)", "DERasterDataset")
        add("donor_raster", "Donor covariate raster (GeoTIFF stack)", "DERasterDataset")
        add("output_folder", "Output folder", "DEFolder")
        # spatial filters (optional)
        add("fnf_shapefile", "FNF shapefile (donor forest/non-forest)", "DEFeatureClass", optional=True)
        add("eligible_shapefile", "Eligible shapefile (donor)", "DEFeatureClass", optional=True)
        add("use_eligibility", "Apply eligibility filter to donor", "GPBoolean", True, optional=True)
        add("donor_extent_km", "Donor extent around PA edge", "GPString", "full",
            optional=True, filter_list=["full", "5", "10", "20", "30", "40", "50", "60"])
        # matching parameters
        add("k_neighbours", "K neighbours (K:1 matching)", "GPLong", 10, optional=True)
        add("knn_query_candidates", "KNN query candidates", "GPLong", 150, optional=True)
        add("n_donor_sample", "Donor sample cap (0 = all)", "GPLong", 800000, optional=True)
        # Step 05
        add("run_step_05", "Run Step 05 (baseline / CI / UNCBSL)", "GPBoolean", True, optional=True)
        add("t0_year", "T0 year", "GPLong", 2018, optional=True)
        add("monitoring_year", "Monitoring year", "GPLong", 2022, optional=True)
        add("min_monitoring_period_years", "Min monitoring period (years)", "GPDouble", 4.0, optional=True)
        add("project_area_ha", "Project area (ha)", "GPDouble", 15713.91, optional=True)
        add("raster_units", "AGB raster units", "GPString", "AGB_Mg_ha",
            optional=True, filter_list=["AGB_Mg_ha", "tC_ha"])
        add("agb_control_t0", "AGB raster — control/donor T0", "DERasterDataset", optional=True)
        add("agb_control_y", "AGB raster — control/donor monitoring year", "DERasterDataset", optional=True)
        add("agb_project_t0", "AGB raster — project T0", "DERasterDataset", optional=True)
        add("agb_project_y", "AGB raster — project monitoring year", "DERasterDataset", optional=True)
        add("use_block_ci", "Conservative block-based CI (larger of pixel/block)", "GPBoolean", True, optional=True)
        add("block_size_m", "Spatial block size (m)", "GPDouble", 500.0, optional=True)
        add("root_to_shoot_ratio", "Root:shoot ratio R  (AGB→C: C = AGB × CF × (1+R); set 0 if raster is already total carbon)", "GPDouble", 0.4, optional=True)
        add("biomass_to_carbon_fraction", "Carbon fraction CF", "GPDouble", 0.47, optional=True)
        return P

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    # ── main ─────────────────────────────────────────────────────────────────
    def execute(self, parameters, messages):
        p = {q.name: q for q in parameters}

        def val(name):
            return p[name].value

        def sval(name):
            v = p[name].valueAsText
            return v if v not in (None, "", "#") else None

        project_name = sval("project_name") or "STARR_project"
        run_id_base = sval("run_id_base") or "run"
        project_raster = sval("project_raster")
        donor_raster = sval("donor_raster")
        out_root = sval("output_folder")
        run_step_05 = bool(val("run_step_05"))

        k_neighbours = int(val("k_neighbours") or 1)
        knn_candidates = int(val("knn_query_candidates") or 150)
        n_donor = int(val("n_donor_sample") or 0)

        t0_year = int(val("t0_year") or 0)
        monitoring_year = int(val("monitoring_year") or 0)
        min_period = float(val("min_monitoring_period_years") or 4.0)
        project_area_ha = float(val("project_area_ha") or 0.0)
        raster_units = sval("raster_units") or "AGB_Mg_ha"

        run_id_full = run_id_base

        # NOTE: the chosen output folder IS the root (no extra 'STARR_outputs'
        # level — that only shortens the path). _win_long lifts the Windows
        # 260-char MAX_PATH limit for deep shared-drive folders + long run ids.
        out_dir = _win_long(os.path.join(out_root, run_id_full))
        d01 = os.path.join(out_dir, "01_extract")
        os.makedirs(d01, exist_ok=True)

        sys.path.insert(0, _HERE)
        sys.path.insert(0, _STEPS)

        # logging FIRST (before imports), tee'd to the dialog and a .log file, so
        # even an import error is captured.
        old_stdout = sys.stdout
        logfh = _open_log(out_dir, "STARR_run")
        sys.stdout = _Tee(logfh)
        try:
            arcpy.AddMessage("=" * 64)
            arcpy.AddMessage(f"STARR SEMDB | {project_name} | {run_id_full}")
            arcpy.AddMessage("=" * 64)

            backend = _install_parquet_fallback()
            arcpy.AddMessage(f"Parquet backend: {backend}")
            gio = _load("starr_geo_io", "starr_geo_io.py")
            s02 = _load("s02", "02_STARR_matching_data_weights.py", _STEPS)
            s03 = _load("s03", "03_STARR_twin_test_selection.py", _STEPS)
            s05 = _load("s05", "05_STARR_baseline_confidence_interval_UNCBSL.py", _STEPS)
            _sk = getattr(s02.NearestNeighbors, "__module__", "") != "_sklearn_fallback"
            arcpy.AddMessage(f"Modules loaded (scikit-learn present: {_sk}; "
                             f"else numpy/scipy fallback — identical results).")

            # matching parameters (fresh process — plain attribute set is enough)
            s02.K_NEIGHBOURS = k_neighbours
            s02.KNN_QUERY_CANDIDATES = knn_candidates
            s02.N_DONOR_SAMPLE = (None if n_donor <= 0 else n_donor)

            # ── STEP 01 — covariate extraction (GDAL) ─────────────────────────
            arcpy.AddMessage("\n[STEP 01] Reading covariate rasters (GDAL)...")
            # Fallback CRS (used only if a covariate raster has no embedded CRS):
            # take it from the FNF shapefile (or the eligible one), as requested.
            _fallback_wkt = (gio.vector_srs_wkt(sval("fnf_shapefile"))
                             or gio.vector_srs_wkt(sval("eligible_shapefile")))
            if _fallback_wkt:
                arcpy.AddMessage("  (fallback CRS from FNF/eligible shapefile is available)")
            proj_df, _, meta = gio.read_covariate_raster_to_df(
                project_raster, tile_name="project", fallback_srs_wkt=_fallback_wkt)
            donor_df, _, _ = gio.read_covariate_raster_to_df(
                donor_raster, tile_name="donor", fallback_srs_wkt=_fallback_wkt)
            if proj_df.empty or donor_df.empty:
                raise RuntimeError("Empty project or donor extraction — check the rasters/band names.")
            meta["run_id"] = run_id_full
            meta["t0_year"] = t0_year
            arcpy.AddMessage(f"  Project pixels : {len(proj_df):,}")
            arcpy.AddMessage(f"  Donor pixels   : {len(donor_df):,}")
            arcpy.AddMessage(f"  Covariates     : {meta.get('continuous_covariates')}")

            # optional donor spatial filters via arcpy (Spatial Analyst)
            donor_df = _apply_donor_filters(
                arcpy, gio, donor_df, proj_df,
                fnf=sval("fnf_shapefile"),
                eligible=(sval("eligible_shapefile") if bool(val("use_eligibility")) else None),
                extent_km=sval("donor_extent_km"),
                messages=arcpy)

            # ── STEP 02 — matching (identical core) ───────────────────────────
            arcpy.AddMessage("\n[STEP 02] Hard-caliper Mahalanobis KNN matching...")
            d02 = os.path.join(out_dir, "02_matching")
            matched_df, weights, imp_df, smd_df, figs02, out02 = s02.run_matching_step(
                base_dirs=[d01], output_dir=d02,
                proj_df=proj_df, donor_df=donor_df, meta=meta, verbose=True)
            arcpy.AddMessage(f"  Matched rows : {len(matched_df):,}")
            if not smd_df.empty:
                arcpy.AddMessage(f"  SMD max      : {round(float(smd_df['SMD'].max()), 4)}")

            # ── STEP 03 — twin / parallel-trends test (identical core) ────────
            arcpy.AddMessage("\n[STEP 03] Twin / parallel-trends test...")
            d03 = os.path.join(out_dir, "03_twin_test")
            all_pairs, twin_pixels, twin_report, figs03, out03 = s03.run_twin_test(
                base_dirs=[out02], output_dir=d03,
                matched_df=matched_df, proj_df=proj_df, meta=meta, verbose=True)
            arcpy.AddMessage(f"  Twin-tested pixels : {len(twin_pixels):,}")

            # ── STEP 04 — reference-area manifest (LOCK) ──────────────────────
            arcpy.AddMessage("\n[STEP 04] Locking reference area...")
            d04 = os.path.join(out_dir, "04_reference_area")
            os.makedirs(d04, exist_ok=True)
            manifest = _build_reference_manifest(twin_pixels, meta, project_name, run_id_full)
            with open(os.path.join(d04, "reference_area_FINAL_manifest.json"), "w") as fh:
                json.dump(manifest, fh, indent=2)
            _write_reference_points(arcpy, twin_pixels, d04)
            arcpy.AddMessage(f"  Reference pixels (unique) : "
                             f"{manifest['reference_area_definition'].get('n_unique_pixels')}")

            # ── STEP 05 — baseline / CI / UNCBSL (identical math, GDAL sampling)
            if run_step_05:
                arcpy.AddMessage("\n[STEP 05] Unadjusted baseline + CI90 / UNCBSL...")
                # monkeypatch the samplers → GDAL, keep every downstream number
                s05.sample_raster_values = gio.sample_raster_values
                s05.raster_pixel_area_ha = gio.raster_pixel_area_ha
                s05.PROJECT_NAME = project_name
                s05.RUN_ID = run_id_full
                _r2s = val("root_to_shoot_ratio")
                _cf = val("biomass_to_carbon_fraction")
                if _r2s is not None:
                    s05.ROOT_TO_SHOOT_RATIO = float(_r2s)
                if _cf is not None:
                    s05.BIOMASS_TO_CARBON_FRACTION = float(_cf)
                arcpy.AddMessage(f"  Carbon: CF={s05.BIOMASS_TO_CARBON_FRACTION} R(root:shoot)={s05.ROOT_TO_SHOOT_RATIO}")

                donor_spec = _agb_spec(sval("agb_control_t0"), sval("agb_control_y"), raster_units)
                project_spec = _agb_spec(sval("agb_project_t0"), sval("agb_project_y"), raster_units)
                d05 = os.path.join(out_dir, "05_baseline_CI_UNCBSL")
                _cv, _pv, summ, report, _figs, _od = s05.run_baseline_ci_uncbsl_from_rasters(
                    base_dirs=[out_dir], output_dir=d05,
                    donor_raster_spec=donor_spec, project_raster_spec=project_spec,
                    twin_df=twin_pixels, project_df=None,
                    project_area_ha=project_area_ha,
                    manifest=manifest,
                    monitoring_period_years=float(monitoring_year - t0_year),
                    t0_year=t0_year, monitoring_year=monitoring_year,
                    min_monitoring_period_years=min_period,
                    use_conservative_block_ci=bool(val("use_block_ci")),
                    spatial_block_size_m=float(val("block_size_m") or 500.0),
                    project_name=project_name, run_id=run_id_full, verbose=True)
                summ = summ if isinstance(summ, dict) else {}
                arcpy.AddMessage("  --- Unadjusted baseline ---")
                for kk in ("delta_C_ref_y_tC_ha_yr", "BL_unadj_period_tCO2e",
                           "mean_deltaC_project_weighted_tC_ha_yr", "uncbsl_percent"):
                    if kk in summ:
                        arcpy.AddMessage(f"    {kk}: {summ[kk]}")
                _bs = report.get("biomass_statistics", {}) if isinstance(report, dict) else {}
                if _bs:
                    _u = _bs.get('source_units', '')
                    arcpy.AddMessage(f"  --- stock mean [biomass {_u} | tC/ha | tCO2e/ha] (n) ---")
                    for _lab, _k in [("project T0", "project_t0"), ("project Ty", "project_monitoring"),
                                     ("donor   T0", "donor_t0"), ("donor   Ty", "donor_monitoring")]:
                        _g = _bs.get(_k, {})
                        if _g.get("n"):
                            _b = _g.get("biomass", {}).get("mean")
                            _c = _g.get("carbon_tC_ha", {}).get("mean")
                            _o = _g.get("co2e_tCO2e_ha", {}).get("mean")
                            arcpy.AddMessage(f"    {_lab}: {_b:.2f} | {_c:.2f} | {_o:.2f} (n={_g['n']:,})")

            arcpy.AddMessage("\nDONE. Outputs in: " + out_dir)
        except Exception as e:
            try: sys.stdout.flush()
            except Exception: pass
            sys.stdout = old_stdout
            arcpy.AddError("STARR tool failed: " + str(e))
            arcpy.AddError(traceback.format_exc())
            raise
        finally:
            try: sys.stdout.flush()
            except Exception: pass
            sys.stdout = old_stdout
            try:
                if logfh: logfh.close()
            except Exception: pass
        return


# ═════════════════════════════════════════════════════════════════════════════
class STARRStep05Tool(object):
    """Re-run ONLY Step 05 (baseline / CI / UNCBSL) on an existing run, without
    redoing the matching. Reads twin_tested_pixels (Step 03) and the reference
    manifest (Step 04) from the run folder and samples the AGB rasters."""

    def __init__(self):
        self.label = "STARR — Step 05 only (baseline / CI / UNCBSL)"
        self.description = ("Recompute the raster-based unadjusted baseline, CI90 "
                            "and UNCBSL for an existing run (reuses Steps 03/04 "
                            "outputs; no re-matching).")
        self.canRunInBackground = False

    def getParameterInfo(self):
        P = []

        def add(name, label, dtype, default=None, optional=False, filter_list=None):
            p = arcpy.Parameter(displayName=label, name=name, datatype=dtype,
                                parameterType=("Optional" if optional else "Required"),
                                direction="Input")
            if default is not None:
                p.value = default
            if filter_list is not None:
                p.filter.type = "ValueList"; p.filter.list = filter_list
            P.append(p)
            return p

        add("run_folder", "Existing run folder (contains 03_twin_test / 04_reference_area)", "DEFolder")
        add("project_name", "Project name", "GPString", "Idiofa_Lobi")
        add("run_id", "Run ID", "GPString", "run")
        add("t0_year", "T0 year", "GPLong", 2018)
        add("monitoring_year", "Monitoring year", "GPLong", 2022)
        add("min_monitoring_period_years", "Min monitoring period (years)", "GPDouble", 4.0, optional=True)
        add("project_area_ha", "Project area (ha)", "GPDouble", 15713.91)
        add("raster_units", "AGB raster units", "GPString", "AGB_Mg_ha",
            optional=True, filter_list=["AGB_Mg_ha", "tC_ha"])
        add("agb_control_t0", "AGB raster — control/donor T0", "DERasterDataset")
        add("agb_control_y", "AGB raster — control/donor monitoring year", "DERasterDataset")
        add("agb_project_t0", "AGB raster — project T0", "DERasterDataset")
        add("agb_project_y", "AGB raster — project monitoring year", "DERasterDataset")
        add("use_block_ci", "Conservative block-based CI", "GPBoolean", True, optional=True)
        add("block_size_m", "Spatial block size (m)", "GPDouble", 500.0, optional=True)
        add("root_to_shoot_ratio", "Root:shoot ratio R  (AGB→C: C = AGB × CF × (1+R); set 0 if raster is already total carbon)", "GPDouble", 0.4, optional=True)
        add("biomass_to_carbon_fraction", "Carbon fraction CF", "GPDouble", 0.47, optional=True)
        return P

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        p = {q.name: q for q in parameters}

        def val(n):
            return p[n].value

        def sval(n):
            v = p[n].valueAsText
            return v if v not in (None, "", "#") else None

        run_folder = sval("run_folder")
        project_name = sval("project_name") or "STARR_project"
        run_id = sval("run_id") or "run"
        t0_year = int(val("t0_year") or 0)
        monitoring_year = int(val("monitoring_year") or 0)
        min_period = float(val("min_monitoring_period_years") or 4.0)
        project_area_ha = float(val("project_area_ha") or 0.0)
        raster_units = sval("raster_units") or "AGB_Mg_ha"

        sys.path.insert(0, _HERE); sys.path.insert(0, _STEPS)
        run_folder = _win_long(run_folder)
        d05 = os.path.join(run_folder, "05_baseline_CI_UNCBSL")

        old_stdout = sys.stdout
        logfh = _open_log(d05, "STARR_step05")
        sys.stdout = _Tee(logfh)
        try:
            arcpy.AddMessage("=" * 64)
            arcpy.AddMessage(f"STARR — Step 05 only | {project_name} | {run_id}")
            arcpy.AddMessage("=" * 64)
            backend = _install_parquet_fallback()
            arcpy.AddMessage(f"Parquet backend: {backend}")
            gio = _load("starr_geo_io", "starr_geo_io.py")
            s05 = _load("s05", "05_STARR_baseline_confidence_interval_UNCBSL.py", _STEPS)

            s05.sample_raster_values = gio.sample_raster_values
            s05.raster_pixel_area_ha = gio.raster_pixel_area_ha
            s05.PROJECT_NAME = project_name
            s05.RUN_ID = run_id
            _r2s = val("root_to_shoot_ratio")
            _cf = val("biomass_to_carbon_fraction")
            if _r2s is not None:
                s05.ROOT_TO_SHOOT_RATIO = float(_r2s)
            if _cf is not None:
                s05.BIOMASS_TO_CARBON_FRACTION = float(_cf)
            arcpy.AddMessage(f"  Carbon: CF={s05.BIOMASS_TO_CARBON_FRACTION} R(root:shoot)={s05.ROOT_TO_SHOOT_RATIO}")

            donor_spec = _agb_spec(sval("agb_control_t0"), sval("agb_control_y"), raster_units)
            project_spec = _agb_spec(sval("agb_project_t0"), sval("agb_project_y"), raster_units)
            d05 = os.path.join(run_folder, "05_baseline_CI_UNCBSL")
            _cv, _pv, summ, report, _figs, _od = s05.run_baseline_ci_uncbsl_from_rasters(
                base_dirs=[run_folder], output_dir=d05,
                donor_raster_spec=donor_spec, project_raster_spec=project_spec,
                twin_df=None, project_df=None,          # reloaded from 03_twin_test
                manifest=None,                          # reloaded from 04_reference_area
                project_area_ha=project_area_ha,
                monitoring_period_years=float(monitoring_year - t0_year),
                t0_year=t0_year, monitoring_year=monitoring_year,
                min_monitoring_period_years=min_period,
                use_conservative_block_ci=bool(val("use_block_ci")),
                spatial_block_size_m=float(val("block_size_m") or 500.0),
                project_name=project_name, run_id=run_id, verbose=True)
            summ = summ if isinstance(summ, dict) else {}
            for kk in ("delta_C_ref_y_tC_ha_yr", "BL_unadj_period_tCO2e",
                       "mean_deltaC_project_weighted_tC_ha_yr", "uncbsl_percent"):
                if kk in summ:
                    arcpy.AddMessage(f"    {kk}: {summ[kk]}")
            _bs = report.get("biomass_statistics", {}) if isinstance(report, dict) else {}
            if _bs:
                _u = _bs.get('source_units', '')
                arcpy.AddMessage(f"  --- stock mean [biomass {_u} | tC/ha | tCO2e/ha] (n) ---")
                for _lab, _k in [("project T0", "project_t0"), ("project Ty", "project_monitoring"),
                                 ("donor   T0", "donor_t0"), ("donor   Ty", "donor_monitoring")]:
                    _g = _bs.get(_k, {})
                    if _g.get("n"):
                        _b = _g.get("biomass", {}).get("mean")
                        _c = _g.get("carbon_tC_ha", {}).get("mean")
                        _o = _g.get("co2e_tCO2e_ha", {}).get("mean")
                        arcpy.AddMessage(f"    {_lab}: {_b:.2f} | {_c:.2f} | {_o:.2f} (n={_g['n']:,})")
            arcpy.AddMessage("\nDONE. Outputs in: " + d05)
        except Exception as e:
            try: sys.stdout.flush()
            except Exception: pass
            sys.stdout = old_stdout
            arcpy.AddError("STARR Step 05 tool failed: " + str(e))
            arcpy.AddError(traceback.format_exc())
            raise
        finally:
            try: sys.stdout.flush()
            except Exception: pass
            sys.stdout = old_stdout
            try:
                if logfh: logfh.close()
            except Exception: pass
        return


# ── helpers (arcpy / gdal) ───────────────────────────────────────────────────
def _agb_spec(t0_path, y_path, units):
    return {
        "delta_raster": None, "stock_raster": None,
        "stock_t0_raster": t0_path, "stock_y_raster": y_path,
        "stock_t0_band": 1, "stock_y_band": 1,
        "stock_t0_band_name": None, "stock_y_band_name": None,
        "units": units,
    }


def _build_reference_manifest(twin_pixels, meta, project_name, run_id):
    import numpy as np
    n_unique = None
    if {"ref_lon", "ref_lat"}.issubset(twin_pixels.columns):
        n_unique = int(twin_pixels[["ref_lon", "ref_lat"]].drop_duplicates().shape[0])
    px_ha = meta.get("pixel_area_ha")
    total_ha = (float(n_unique) * float(px_ha)) if (n_unique and px_ha) else None
    return {
        "project_name": project_name,
        "run_id": run_id,
        "lock_status": "LOCKED",
        "area_source": "twin_tested_reference_pixels",
        "reference_area_definition": {
            "n_unique_pixels": n_unique,
            "pixel_area_ha": px_ha,
            "total_ha": total_ha,
        },
    }


def _write_reference_points(arcpy, df, out_dir):
    """Write the reference (control) points. Always a CSV (reliable, no path-length
    or arcpy limits); the shapefile is best-effort (arcpy rejects extended-length
    \\\\?\\ paths and shapefiles have a ~260-char path limit, so it may be skipped
    on very deep output folders — the CSV is the fallback)."""
    if not {"ref_lon", "ref_lat"}.issubset(df.columns):
        return
    # 1) CSV — always works (open() accepts \\?\ paths, no length limit)
    keep = [c for c in ("ref_lon", "ref_lat", "proj_lon", "proj_lat",
                        "ref_texture", "proj_texture", "match_distance")
            if c in df.columns]
    csv_path = os.path.join(out_dir, "reference_points.csv")
    try:
        df[keep].to_csv(csv_path, index=False)
        arcpy.AddMessage(f"  reference_points.csv written ({len(df):,} rows)")
    except Exception as e:
        arcpy.AddWarning(f"Could not write reference_points.csv: {e}")

    # 2) Shapefile — arcpy cannot use \\?\ paths and shapefiles have a ~260-char
    #    limit, so write it in a SHORT temp folder, then move the sidecar files
    #    (.shp/.shx/.dbf/.prj/.cpg) to the final (possibly very long) directory
    #    with Python, whose open()/shutil DO handle \\?\ long paths.
    import tempfile as _tf, shutil as _sh, glob as _glob
    tmpd = None
    try:
        sr = arcpy.SpatialReference(4326)
        tmpd = _tf.mkdtemp(prefix="starr_refpts_")            # short path, arcpy-safe
        tmp_shp = os.path.join(tmpd, "reference_points.shp")
        arcpy.management.CreateFeatureclass(tmpd, "reference_points.shp",
                                            "POINT", spatial_reference=sr)
        with arcpy.da.InsertCursor(tmp_shp, ["SHAPE@XY"]) as cur:
            for lon, lat in df[["ref_lon", "ref_lat"]].dropna().itertuples(index=False):
                cur.insertRow([(float(lon), float(lat))])
        del cur
        try:
            arcpy.management.ClearWorkspaceCache()   # release the schema lock
        except Exception:
            pass
        # Move ONLY the real shapefile components — never arcpy's transient
        # .lock / .sr.lock files (they are held open → Permission denied).
        moved = 0
        for f in _glob.glob(os.path.join(tmpd, "reference_points.*")):
            base = os.path.basename(f).lower()
            if ".lock" in base:
                continue
            dest = os.path.join(out_dir, os.path.basename(f))   # out_dir may be \\?\...
            try:
                if os.path.exists(dest):
                    os.remove(dest)
                _sh.move(f, dest)
                moved += 1
            except Exception as _me:
                arcpy.AddWarning(f"    (skipped sidecar {os.path.basename(f)}: {_me})")
        if moved:
            arcpy.AddMessage(f"  reference_points.shp written ({moved} sidecar files)")
        else:
            arcpy.AddWarning("  reference_points.shp not written; use reference_points.csv.")
    except Exception as e:
        arcpy.AddWarning(f"reference_points.shp not written ({e}); "
                         f"use reference_points.csv instead.")
    finally:
        if tmpd:
            try:
                _sh.rmtree(tmpd, ignore_errors=True)
            except Exception:
                pass


def _apply_donor_filters(arcpy, gio, donor_df, proj_df, fnf, eligible, extent_km, messages):
    """
    Optional donor spatial filters, arcpy-native (point-in-polygon via a temp
    in-memory point FC + Select Layer By Location). Skipped (with a message) when
    no shapefile is given — the GEE export already clips the donor to the
    ecoregion∩buffer∩eligible area, so these are additional, conservative filters.
    """
    import numpy as np
    masks = [(fnf, "FNF"), (eligible, "eligible")]
    active = [(shp, tag) for shp, tag in masks if shp]
    # donor extent (km) around the PA bounding box
    if extent_km and str(extent_km).lower() != "full":
        try:
            km = float(extent_km)
            lon0, lon1 = proj_df["lon"].min(), proj_df["lon"].max()
            lat0, lat1 = proj_df["lat"].min(), proj_df["lat"].max()
            dlat = km / 111.0
            dlon = km / (111.0 * max(0.1, np.cos(np.deg2rad((lat0 + lat1) / 2.0))))
            keep = (donor_df["lon"].between(lon0 - dlon, lon1 + dlon)
                    & donor_df["lat"].between(lat0 - dlat, lat1 + dlat))
            n0 = len(donor_df); donor_df = donor_df[keep].reset_index(drop=True)
            arcpy.AddMessage(f"  Donor extent {km:g} km: {n0:,} -> {len(donor_df):,}")
        except Exception as e:
            arcpy.AddWarning(f"  Donor extent filter skipped: {e}")
    if not active:
        arcpy.AddMessage("  Donor shapefile filters: none provided (relying on GEE pre-clip).")
        return donor_df
    for shp, tag in active:
        try:
            donor_df = _mask_points_by_polygon(arcpy, donor_df, shp, tag)
        except Exception as e:
            arcpy.AddWarning(f"  {tag} filter skipped ({e}); relying on GEE pre-clip.")
    return donor_df


def _mask_points_by_polygon(arcpy, donor_df, shp, tag):
    """Keep donor rows whose (lon,lat) fall inside the polygon shapefile."""
    import numpy as np
    tmp = "in_memory/_starr_donor_pts"
    if arcpy.Exists(tmp):
        arcpy.management.Delete(tmp)
    sr = arcpy.SpatialReference(4326)
    arcpy.management.CreateFeatureclass("in_memory", "_starr_donor_pts", "POINT",
                                        spatial_reference=sr)
    arcpy.management.AddField(tmp, "row_id", "LONG")
    with arcpy.da.InsertCursor(tmp, ["SHAPE@XY", "row_id"]) as cur:
        for i, (lon, lat) in enumerate(donor_df[["lon", "lat"]].itertuples(index=False)):
            cur.insertRow([(float(lon), float(lat)), i])
    lyr = arcpy.management.MakeFeatureLayer(tmp, "_starr_donor_lyr").getOutput(0)
    arcpy.management.SelectLayerByLocation(lyr, "WITHIN", shp)
    keep_ids = {r[0] for r in arcpy.da.SearchCursor(lyr, ["row_id"])}
    n0 = len(donor_df)
    donor_df = donor_df.iloc[sorted(keep_ids)].reset_index(drop=True)
    arcpy.AddMessage(f"  {tag} filter: {n0:,} -> {len(donor_df):,}")
    arcpy.management.Delete(tmp)
    return donor_df
