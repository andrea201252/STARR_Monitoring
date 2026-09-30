# STARR SEMDB — ArcGIS Pro Python Toolbox

ArcGIS Pro port of the GS STARR Track 1 SEMDB pipeline (Steps 01–05:
covariate extraction → hard-caliper Mahalanobis KNN matching → twin /
parallel-trends test → reference-area lock → unadjusted baseline + CI90 /
UNCBSL).

## Why the results are identical to the Colab version

The statistical core is **the same code**. This package reuses, unchanged, the
Colab step modules:

- `steps/02_STARR_matching_data_weights.py` — matching (Mahalanobis + hard calipers)
- `steps/03_STARR_twin_test_selection.py` — twin / parallel-trends test
- `steps/05_STARR_baseline_confidence_interval_UNCBSL.py` — baseline, CI90, UNCBSL

Only the **I/O layer** is different: instead of `rasterio` / `geopandas` (not in
ArcGIS Pro) it uses **GDAL / OGR / pyproj**, which ship inside the ArcGIS Pro
Python environment (`arcgispro-py3`) — **no installation required**. `rasterio`
is a thin wrapper over GDAL, so the pixel reads are the same values, and the
carbon math (AGB→C, ΔC, CI, UNCBSL) is executed by the original functions.
The Step 05 raster sampler is swapped for a GDAL one via a monkeypatch, so every
downstream number is produced by the unchanged Step 05 code.

## Requirements

- ArcGIS Pro 3.x, using its bundled Python `arcgispro-py3`. Needs only packages
  that are **always present**: numpy, pandas, scipy, matplotlib, GDAL/osgeo.
- **pyproj is NOT required.** Coordinate reprojection uses GDAL/osr (always
  present with GDAL); pyproj is only a fallback if it happens to be importable.
  If a covariate raster has no embedded CRS, the tool falls back to the **FNF
  shapefile's CRS**.
- **scikit-learn is NOT required.** Step 02 uses it if present; if it is missing
  (as on many ArcGIS installs) the toolbox automatically falls back to
  `steps/_sklearn_fallback.py` — numpy/scipy drop-ins for `StandardScaler`,
  `LedoitWolf` and `NearestNeighbors` that reproduce scikit-learn **identically**
  (validated: Ledoit-Wolf covariance bit-identical, same k-NN neighbours). No
  installation needed either way. The run log states which path was used.
- No extension is required (the default donor point-in-polygon filter uses core arcpy).
- The GEE covariate/AGB GeoTIFFs (produced by the unchanged `Covariate_Extraction.js`).

## Logs

Every step's console output is shown live in the tool dialog **and** written to a
timestamped `.log` file in the output folder (`STARR_run_*.log`, or
`STARR_step05_*.log` for the Step-05-only tool), so you always have the full log
even after the dialog closes.

## Install

1. Copy the whole `STARR_ArcGISPro/` folder somewhere stable (keep
   `STARR_SEMDB.pyt`, `starr_geo_io.py` and the `steps/` folder together).
2. In ArcGIS Pro: **Catalog** ▸ right-click **Toolboxes** ▸ *Add Toolbox* ▸ pick
   `STARR_SEMDB.pyt`.
3. Open **STARR SEMDB Baseline (Steps 01-05)**.

## Inputs (tool dialog — mirrors the Colab §1 form)

| Field | Notes |
|---|---|
| Project name / Run ID | appear in every output / JSON |
| Project covariate raster | `covariates_project_*.tif` from GEE (multi-band) |
| Donor covariate raster | `covariates_donor_*.tif` from GEE (multi-band) |
| Output folder | outputs go to `<folder>/STARR_outputs/<run_id>/…` |
| FNF / Eligible shapefile | optional donor spatial filters |
| Donor extent (km) | `full` or 5/10/20/30 around the PA edge |
| K neighbours | **K:1 matching** (each project pixel → up to K control donors) |
| KNN query candidates | candidates screened before the hard calipers |
| Donor sample cap | 0 = use all donors |
| T0 / monitoring year, min period, project area (ha) | Step 05 |
| AGB rasters (control/project × T0/Y) + units | Step 05 (`AGB_Mg_ha` or `tC_ha`) |
| Root:shoot ratio R, carbon fraction CF | Step 05 carbon conversion: C = AGB × CF × (1+R). Set **R = 0** if the raster already includes roots or is already total carbon. |

## Output folder & Windows long paths

The output folder you pick **is** the run root; the tool writes
`<output folder>/<run id>/01_extract`, `…/02_matching`, … directly under it (no
extra `STARR_outputs` level). Deep shared-drive folders (e.g. `G:\Drive
condivisi\…`) plus a long run id can exceed the Windows 260-character path limit;
the tool automatically switches to Windows extended-length paths (`\\?\`) so this
no longer fails. If you still hit a path error, pick a shorter output folder.

**Band names matter**: the covariate GeoTIFF must carry band descriptions
(`WRB2_CODE`, `SOC_g_kg`, `NDVI_t0`, `elevation`, `slope_deg`, `dist_roads_km`,
`precip_mm_yr`, `NDVI_<year>`, optional `tenure`). GEE exports these; GDAL reads
them. If a raster has generic band names, set them once with
`gdal_edit`/`arcpy` or re-export from GEE with band names.

## Outputs

Same layout and files as Colab, under `<output>/STARR_outputs/<run_id>/`:
`01_extract/`, `02_matching/`, `03_twin_test/`, `04_reference_area/`,
`05_baseline_CI_UNCBSL/` (with `baseline_CI90_UNCBSL_report.json` +
`…_summary.csv`). See `STARR_Step05_output_variables.md` for every field.

`04_reference_area/` also holds the geometry of the matched pixels (EPSG:4326,
de-duplicated), for both the reference (donor/control) and the project (PA):
`reference_points.shp` / `project_points.shp` (pixel centres) and
`reference_grid.shp` / `project_grid.shp` (one square cell per pixel, ±½ pixel),
plus `reference_points.csv` / `project_points.csv` as fallbacks. Shapefiles are
written in a short temp folder and moved to the (possibly very long) output path.

## Two tools in this toolbox

1. **STARR SEMDB Baseline (Steps 01-05)** — the full pipeline.
2. **STARR — Step 05 only (baseline / CI / UNCBSL)** — recompute the baseline for
   an existing run (reuses the Steps 03/04 outputs already on disk, no
   re-matching). Point it at the run folder + the AGB rasters.

## Status & known limits (ArcGIS build v0.2)

- The **statistical results match the Colab pipeline by construction** (same
  modules). Validated offline: pixel-grid/coordinate math identical to Step 01;
  Step 05 GDAL sampler reproduces the AGB→C→ΔC math exactly; matching (per-texture
  KNN + K:1) verified; **conservative block-based CI works** (via pyproj, no rasterio).
- **Needs a first run inside ArcGIS Pro** to validate the arcpy/GDAL calls on your
  machine (I cannot run arcpy offline). Send me any `arcpy`/GDAL message and I fix it.
- The optional **donor shapefile filter** uses an arcpy point-in-polygon on a
  temporary point layer; for very large donor pools prefer the GEE pre-clip
  (ecoregion ∩ buffer ∩ eligible) already applied at export, plus the `Donor
  extent (km)` box.
- **Covariate extraction stays raster-based** (GEE). Moving the covariate
  generation itself into ArcGIS was intentionally left out of scope.
