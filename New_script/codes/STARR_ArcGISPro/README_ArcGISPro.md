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

- ArcGIS Pro 3.x (uses its bundled Python `arcgispro-py3`: numpy, pandas,
  scikit-learn, scipy, matplotlib, GDAL/osgeo, pyproj — all already present).
- **Spatial Analyst** is only needed if you use the optional donor
  shapefile filters via raster masking; the default point-in-polygon filter uses
  core arcpy (no extension).
- The GEE covariate/AGB GeoTIFFs (produced by the unchanged `Covariate_Extraction.js`).

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
