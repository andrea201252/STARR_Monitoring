# GS STARR Track 1 SEMDB — Raster-first Matching Pipeline

This repository implements a raster-first workflow for the Gold Standard STARR Track 1 SEMDB approach.

The pipeline extracts project and donor covariate pixels from GEE-exported rasters, applies spatial eligibility masks, performs Mahalanobis KNN matching with hard calipers, applies a parallel-trend / twin-test filter, locks the final reference area, and calculates baseline uncertainty / UNCBSL from locked matched control pixels.

The most important implementation rule is that all paths, run IDs, raster names and output folders must remain aligned across all steps.

---

# 1. Pipeline overview

The workflow is organized into sequential scripts.

| Step | Script | Purpose |
|---|---|---|
| 00 | `00_STARR_run_steps_01_to_05.py` | Main parametric runner for Steps 01–05 |
| 00b | `00b_STARR_compare_extents.py` | Optional multi-extent comparison runner |
| 01 | `01_STARR_raster_extract.py` | Raster extraction and spatial masking |
| 02 | `02_STARR_matching_data_weights.py` | Hard-caliper + Mahalanobis KNN matching |
| 03 | `03_STARR_twin_test_selection.py` | Parallel trend / twin-test selection |
| 04 | `04_STARR_reference_area_lock.py` | Final reference area construction and lock |
| 05 | `05_STARR_baseline_confidence_interval_UNCBSL.py` | Baseline CI90 / UNCBSL calculation |
| 05 alternative | `05_STARR_baseline_uncertainty.py` | Extended baseline uncertainty + BR calculation variant |

Recommended entry point:

```bash
python 00_STARR_run_steps_01_to_05.py
```

Optional donor extent comparison:

```bash
python 00b_STARR_compare_extents.py
```

---

# 2. Repository structure

Recommended repository layout:

```text
STARR_pipeline/
├── README.md
├── 00_STARR_run_steps_01_to_05.py
├── 00b_STARR_compare_extents.py
├── 01_STARR_raster_extract.py
├── 02_STARR_matching_data_weights.py
├── 03_STARR_twin_test_selection.py
├── 04_STARR_reference_area_lock.py
├── 05_STARR_baseline_confidence_interval_UNCBSL.py
├── 05_STARR_baseline_uncertainty.py
├── STARR_pipeline.ipynb
└── docs/
    ├── expected_inputs.md
    ├── output_structure.md
    └── troubleshooting.md
```

Recommended external data layout:

```text
/data_or_drive_root/
├── covariates_project_<RUN_ID_BASE>.tif
├── covariates_donor_<RUN_ID_BASE>.tif
├── FNF18_fullBuffer.shp
├── Eligible_FNF_fullBuffer.shp
└── STARR_outputs/
    └── <effective_run_id>/
        ├── 01_extract/
        ├── 02_matching/
        ├── 03_twin_test/
        ├── 04_reference_area/
        └── 05_baseline_CI_UNCBSL/
```

Do not store heavy rasters, shapefiles or generated outputs directly in GitHub unless using Git LFS.

---

# 3. Critical naming and path alignment

This pipeline depends on strict alignment between:

1. `RUN_ID_BASE`
2. GEE raster filenames
3. `BASE_DIR`
4. `DONOR_EXTENT_KM`
5. effective output run directory
6. internal `RUN_ID` used by Step 05, if Step 05 is executed independently

If these are not aligned, the scripts may read rasters or outputs from the wrong run.

---

## 3.1 Required GEE raster naming convention

The exported GEE rasters must follow this naming structure:

```text
covariates_project_<RUN_ID_BASE>*.tif
covariates_donor_<RUN_ID_BASE>*.tif
```

Example:

```text
covariates_project_Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster.tif
covariates_donor_Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster.tif
```

Then the runner must use:

```python
RUN_ID_BASE = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster"
```

Correct:

```python
RUN_ID_BASE = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster"
```

Wrong:

```python
RUN_ID_BASE = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster_"
```

Do not add an extra underscore at the end of `RUN_ID_BASE`.

If:

```python
RUN_ID_BASE = ""
```

the script uses a wildcard and searches for any:

```text
covariates_project_*.tif
covariates_donor_*.tif
```

This is useful for testing but not recommended for reproducible production runs.

---

## 3.2 Effective run ID

The effective run ID depends on `RUN_ID_BASE` and `DONOR_EXTENT_KM`.

If:

```python
RUN_ID_BASE = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster"
DONOR_EXTENT_KM = 10
```

the effective run ID becomes:

```text
Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster_ext10km
```

For `DONOR_EXTENT_KM = "full"`, the effective run ID remains:

```text
<RUN_ID_BASE>
```

For numeric extents:

```text
<RUN_ID_BASE>_ext5km
<RUN_ID_BASE>_ext10km
<RUN_ID_BASE>_ext20km
<RUN_ID_BASE>_ext30km
```

This effective run ID becomes the output folder name:

```text
BASE_DIR/STARR_outputs/<effective_run_id>/
```

---

# 4. Main runner configuration

The main configuration is inside:

```text
00_STARR_run_steps_01_to_05.py
```

Edit only the configuration block before running:

```python
DONOR_EXTENT_KM = 10

RUN_ID_BASE = "your_run_id_base"

BASE_DIR = "/path/to/folder/containing/gee/tifs"

FNF_SHAPEFILE = "/path/to/FNF18_fullBuffer.shp"

ELIGIBLE_SHAPEFILE = "/path/to/Eligible_FNF_fullBuffer.shp"
```

---

## 4.1 `DONOR_EXTENT_KM`

`DONOR_EXTENT_KM` controls the donor search extent around the project area.

Allowed values:

```python
DONOR_EXTENT_KM = "full"
DONOR_EXTENT_KM = 5
DONOR_EXTENT_KM = 10
DONOR_EXTENT_KM = 20
DONOR_EXTENT_KM = 30
```

Meaning:

| Value | Meaning |
|---|---|
| `"full"` | Use the full donor raster |
| `5` | Clip donor raster to project bbox + 5 km |
| `10` | Clip donor raster to project bbox + 10 km |
| `20` | Clip donor raster to project bbox + 20 km |
| `30` | Clip donor raster to project bbox + 30 km |

The donor clip is applied before loading donor pixels into memory. This reduces RAM pressure and avoids loading the full donor raster when a smaller donor search window is sufficient.

---

## 4.2 `BASE_DIR`

`BASE_DIR` must point to the folder containing the GEE-exported raster files.

Example:

```python
BASE_DIR = "/content/drive/MyDrive/STARR_Idiofa_New_V2"
```

Expected files:

```text
/content/drive/MyDrive/STARR_Idiofa_New_V2/
├── covariates_project_<RUN_ID_BASE>.tif
├── covariates_donor_<RUN_ID_BASE>.tif
├── FNF18_fullBuffer.shp
└── Eligible_FNF_fullBuffer.shp
```

The same directory is also used as the root where `STARR_outputs/` is created.

---

## 4.3 `FNF_SHAPEFILE`

`FNF_SHAPEFILE` is the shapefile containing forest-at-T0 polygons.

It is used to remove forest-at-T0 pixels from both:

1. project area pixels;
2. donor pixels.

Example:

```python
FNF_SHAPEFILE = "/content/drive/MyDrive/STARR_Idiofa_New_V2/FNF18_fullBuffer.shp"
```

Set it to `None` only if this spatial filter must be disabled:

```python
FNF_SHAPEFILE = None
```

---

## 4.4 `ELIGIBLE_SHAPEFILE`

`ELIGIBLE_SHAPEFILE` is the shapefile containing donor-eligible non-forest areas.

It is applied only to donor pixels.

Example:

```python
ELIGIBLE_SHAPEFILE = "/content/drive/MyDrive/STARR_Idiofa_New_V2/Eligible_FNF_fullBuffer.shp"
```

Set it to `None` only if this spatial filter must be disabled:

```python
ELIGIBLE_SHAPEFILE = None
```

---

# 5. Output directory structure

Each run writes outputs to:

```text
BASE_DIR/STARR_outputs/<effective_run_id>/
```

Example:

```text
/content/drive/MyDrive/STARR_Idiofa_New_V2/
└── STARR_outputs/
    └── Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster_ext10km/
        ├── 01_extract/
        ├── 02_matching/
        ├── 03_twin_test/
        ├── 04_reference_area/
        └── 05_baseline_CI_UNCBSL/
```

Each step writes its outputs into a dedicated subfolder.

---

# 6. Required input raster bands

The project and donor rasters must contain consistent band names.

Minimum expected bands:

```text
NDVI_YYYY
NDVI_t0 or equivalent T0 NDVI band
SOC_g_kg
elevation
slope_deg
dist_roads_km
WRB2_CODE
pixel_area_ha
lon
lat
```

Common expected covariates:

```text
NDVI_2013
NDVI_2014
NDVI_2015
NDVI_2016
NDVI_2017
NDVI_t0
elevation
slope_deg
precip_mm_yr
SOC_g_kg
WRB2_CODE
dist_roads_km
pixel_area_ha
```

Project and donor rasters must use the same schema.

If one covariate exists in the project raster but not in the donor raster, Step 02 will fail or produce invalid matching results.

Annual NDVI bands must follow this format:

```text
NDVI_YYYY
```

Examples:

```text
NDVI_2013
NDVI_2014
NDVI_2015
NDVI_2016
NDVI_2017
```

`WRB2_CODE` is mandatory because Step 02 maps WRB soil codes to soil texture classes.

---

# 7. Step 01 — Raster extraction and spatial masking

Script:

```text
01_STARR_raster_extract.py
```

Purpose:

Step 01 reads the GEE-exported project and donor covariate rasters, converts valid raster pixels into tabular pixel records, applies spatial filters, and writes the extracted project/donor datasets.

---

## 7.1 Main operations

Step 01:

1. Searches for project and donor TIFs using:

```text
covariates_project_<RUN_ID_BASE>*.tif
covariates_donor_<RUN_ID_BASE>*.tif
```

2. Loads project raster pixels block-by-block.
3. Detects band names from raster descriptions.
4. Detects NDVI annual bands.
5. Detects the T0 year from available NDVI bands.
6. Detects continuous covariates.
7. Converts raster pixels into a tabular dataframe.
8. Adds:
   - UTM x/y coordinates;
   - lon/lat coordinates;
   - grid row/column;
   - native raster cell bounds;
   - pixel ID;
   - source tile.
9. Applies project-area edge exclusion.
10. Optionally applies spatial sampling to the project pixels if the PA is too large.
11. Applies `FNF_SHAPEFILE` to remove forest-at-T0 pixels from the project area.
12. Computes the donor clip window from project bbox + `DONOR_EXTENT_KM`.
13. Loads only the clipped donor raster window.
14. Applies donor spatial filters:
   - removes donor pixels inside `FNF_SHAPEFILE`;
   - keeps only donor pixels inside `ELIGIBLE_SHAPEFILE`.
15. Saves extracted pixel tables and diagnostic outputs.

---

## 7.2 Important Step 01 parameters

```python
RUN_ID_BASE = ""
PROJECT_NAME = "Idiofa_Lobi"

BASE_DIR_CANDIDATES = []

OUTPUT_FORMAT = "parquet"
BLOCK_HEIGHT = 2048

FNF_SHAPEFILE = None
ELIGIBLE_SHAPEFILE = None

SHP_ALL_TOUCHED = False

PA_EDGE_EXCLUSION_ENABLED = True
PA_EDGE_EXCLUSION_N_PIXELS = 1

PA_SPATIAL_SAMPLE_ENABLED = True
PA_MAX_PIXELS = 150_000
PA_SPATIAL_GRID_STEP_M = 150.0
```

---

## 7.3 Step 01 outputs

```text
01_extract/
├── project_pixels_raw.parquet
├── donor_pixels_raw.parquet
├── extraction_report.json
├── pixel_grid_diagnostics.png
└── ndvi_valid_years.png
```

The most important outputs are:

```text
project_pixels_raw.parquet
donor_pixels_raw.parquet
extraction_report.json
```

These are used directly by Step 02.

---

## 7.4 Step 01 validation checks

Before moving to Step 02, verify:

```text
[ ] project_pixels_raw.parquet exists.
[ ] donor_pixels_raw.parquet exists.
[ ] extraction_report.json exists.
[ ] project_n > 0.
[ ] donor_n > 0.
[ ] WRB2_CODE exists.
[ ] NDVI_YYYY bands were detected.
[ ] continuous_covariates is not empty.
[ ] CRS and transform are correct.
[ ] shapefile filters did not remove all valid pixels.
```

If Step 01 returns empty project or donor tables, check:

```text
[ ] wrong RUN_ID_BASE;
[ ] wrong BASE_DIR;
[ ] missing TIF files;
[ ] shapefile path incorrect;
[ ] shapefile CRS incorrect;
[ ] shapefile does not overlap raster extent;
[ ] donor extent too small;
[ ] invalid raster mask;
[ ] missing or mismatched raster bands.
```

---

# 8. Step 02 — Hard-caliper + Mahalanobis KNN matching

Script:

```text
02_STARR_matching_data_weights.py
```

Purpose:

Step 02 matches project pixels to donor/reference pixels using plain Mahalanobis distance with hard calipers.

RF-derived feature weights are intentionally removed. The implemented matching is based on plain Mahalanobis distance, standardized covariates and a Ledoit-Wolf covariance estimate.

---

## 8.1 Main operations

Step 02:

1. Loads `project_pixels_raw` and `donor_pixels_raw` from Step 01.
2. Reads covariate metadata from `extraction_report.json`.
3. Maps `WRB2_CODE` to soil texture class.
4. Removes rows with missing values in matching covariates.
5. Applies a loose donor prefilter based on project covariate ranges.
6. Fits a `StandardScaler` on a project + donor sample.
7. Fits a Ledoit-Wolf covariance matrix on a standardized sample.
8. Converts Mahalanobis distance into whitened Euclidean distance.
9. Builds a `ball_tree` KNN index.
10. Queries KNN candidates in batches to reduce memory use.
11. Applies hard calipers.
12. Enforces maximum donor reuse.
13. Exports matched and unmatched project pixels.
14. Computes SMD balance diagnostics.
15. Writes matching summary and diagnostic plots.

---

## 8.2 Main Step 02 parameters

```python
K_NEIGHBOURS = 1
KNN_QUERY_CANDIDATES = 150
N_DONOR_SAMPLE = 300_000
ALLOW_TEXTURE_FALLBACK = False
MAX_DONOR_REUSE = 1

KNN_BATCH_SIZE = 4096
KNN_N_JOBS = -1
KNN_LEAF_SIZE = 60

SCALER_FIT_MAX_ROWS = 60_000
COV_FIT_MAX_ROWS = 40_000

SMD_THRESHOLD = 0.1
RIDGE_REG = 1e-6
```

---

## 8.3 Hard calipers

The implemented hard calipers are:

```text
SOC_g_kg       ±10% of project mean
NDVI_t0        ±10% of project mean
elevation      ±200 m
slope_deg      ±10°
dist_roads_km  ±1 km
texture_class  exact match
tenure_col     exact match if available
```

Parameter block:

```python
CALIPER_SOC_FRAC_OF_PROJECT_MEAN = 0.10
CALIPER_NDVI_T0_FRAC_OF_PROJECT_MEAN = 0.10
CALIPER_ELEVATION_M = 200.0
CALIPER_SLOPE_DEG = 10.0
CALIPER_ROADS_KM = 1.0
```

---

## 8.4 Soil texture mapping

`WRB2_CODE` is mapped to texture classes:

```text
sand
sandy_loam
loam
clay_loam
clay
```

If `ALLOW_TEXTURE_FALLBACK = False`, only exact texture matches are allowed.

If `ALLOW_TEXTURE_FALLBACK = True`, fallback texture classes can be used according to the predefined fallback order.

Default recommended setting:

```python
ALLOW_TEXTURE_FALLBACK = False
```

---

## 8.5 KNN candidate logic

```python
KNN_QUERY_CANDIDATES = 150
```

This is the number of donor candidates queried before applying hard calipers.

It is not the final number of matches.

The final number of matches per project pixel is controlled by:

```python
K_NEIGHBOURS = 1
```

The workflow is:

```text
query 150 nearest donor candidates
→ apply hard calipers
→ enforce donor reuse
→ keep the best valid K_NEIGHBOURS match
```

If too many project pixels remain unmatched, increase `KNN_QUERY_CANDIDATES`.

---

## 8.6 Step 02 outputs

```text
02_matching/
├── reference_area_pixels.parquet
├── unmatched_project_pixels.parquet
├── matching_summary.json
├── smd_balance.csv
└── matching_diagnostics.png
```

Important files:

```text
reference_area_pixels.parquet
matching_summary.json
smd_balance.csv
```

---

## 8.7 Step 02 validation checks

Before moving to Step 03, verify:

```text
[ ] reference_area_pixels.parquet exists.
[ ] matching_summary.json exists.
[ ] n_matched > 0.
[ ] match_coverage_pct is acceptable.
[ ] SMD_max <= 0.10.
[ ] SMD_all_passed is True.
[ ] unmatched_project_pixels.parquet does not contain most of the PA.
```

Recommended thresholds for extent comparison:

```text
match_coverage_pct >= 90%
SMD_max <= 0.10
```

If match coverage is low, check:

```text
[ ] donor extent too small;
[ ] donor eligible mask too restrictive;
[ ] FNF mask removing too many pixels;
[ ] wrong covariate scale;
[ ] missing or wrong NDVI_t0;
[ ] texture class too restrictive;
[ ] KNN_QUERY_CANDIDATES too low;
[ ] MAX_DONOR_REUSE too strict;
[ ] project and donor rasters not comparable.
```

---

# 9. Step 03 — Parallel trend / twin-test selection

Script:

```text
03_STARR_twin_test_selection.py
```

Purpose:

Step 03 filters matched project-reference pairs using NDVI time-series behaviour. The goal is to retain only matched control pixels with sufficiently parallel pre-baseline vegetation trends.

---

## 9.1 Main operations

Step 03:

1. Loads matched pairs from Step 02.
2. Detects annual NDVI columns.
3. Extracts years from `NDVI_YYYY` band names.
4. Computes project NDVI slopes using vectorized OLS.
5. Computes reference NDVI slopes using vectorized OLS.
6. Computes pair-level slope differences.
7. Builds a long-format dataframe for aggregate interaction testing.
8. Runs an aggregate parallel trend / interaction test.
9. Selects pairs that pass the twin-test conditions.
10. Exports accepted twin-tested reference pixels.
11. Generates diagnostic plots.

---

## 9.2 Main Step 03 parameters

```python
PVALUE_THRESHOLD = 0.05
PAIR_SLOPE_DIFF_MAX = 0.005
MIN_SELECTED_FRACTION = 0.30
MIN_SELECTED_PAIRS = 30
GRID_N_EXAMPLES = 16
GRID_SEED = 42
```

---

## 9.3 Step 03 outputs

```text
03_twin_test/
├── twin_tested_pixels.parquet
├── all_pairs_with_twin_metrics.parquet
├── twin_test_report.json
└── twin_test_examples_grid.png
```

Important files:

```text
twin_tested_pixels.parquet
twin_test_report.json
```

These are used by Step 04.

---

## 9.4 Step 03 validation checks

Before moving to Step 04, verify:

```text
[ ] twin_tested_pixels.parquet exists.
[ ] twin_test_report.json exists.
[ ] selected_pairs > 0.
[ ] twin_pass_pct >= 30%, if using the comparison criteria.
[ ] aggregate_twin_passed is acceptable.
[ ] NDVI years are correct.
[ ] NDVI slopes are not dominated by missing data.
```

If the twin-test rejects too many pairs, check:

```text
[ ] annual NDVI bands are correct;
[ ] cloud masking was too strict or too weak;
[ ] pre-baseline years are wrong;
[ ] donor area is not comparable;
[ ] matching covariates do not capture vegetation behaviour;
[ ] PA and donor have different land-use dynamics;
[ ] PAIR_SLOPE_DIFF_MAX is too restrictive.
```

---

# 10. Step 04 — Reference area construction and lock

Script:

```text
04_STARR_reference_area_lock.py
```

Purpose:

Step 04 builds the final locked reference area from the twin-tested matched reference pixels.

The default approach is pixel-support based. Each accepted matched donor/reference pixel is converted into a 30 m support cell.

This avoids creating a smoothed hull that includes unvalidated donor pixels.

---

## 10.1 Main Step 04 parameters

```python
RA_SUPPORT_MODE = "pixel"
PIXEL_SIZE_M = 30.0
BLOCK_SIZE_M = 150.0

EDGE_BUFFER_M = 0.0
CLOSE_GAPS_M = 0.0
SIMPLIFY_TOL_M = 0.0
MIN_POLYGON_HA = 0.0

MONITORING_MODE = "matched_pixels"
MONITORING_SPACING_M = 500.0
MONITORING_SEED = 42
```

Recommended final-compliance setting:

```python
RA_SUPPORT_MODE = "pixel"
EDGE_BUFFER_M = 0.0
CLOSE_GAPS_M = 0.0
SIMPLIFY_TOL_M = 0.0
MIN_POLYGON_HA = 0.0
MONITORING_MODE = "matched_pixels"
```

---

## 10.2 Main operations

Step 04:

1. Loads `twin_tested_pixels` from Step 03.
2. Builds reference points from:
   - `ref_lon`
   - `ref_lat`
3. Automatically detects an appropriate projected CRS.
4. Converts accepted reference pixels into support cells.
5. Uses native raster bounds if available:
   - `ref_cell_xmin`
   - `ref_cell_ymin`
   - `ref_cell_xmax`
   - `ref_cell_ymax`
6. Builds the final reference area geometry.
7. Exports matched donor pixels.
8. Exports support cells.
9. Exports the final locked reference area.
10. Generates monitoring points.
11. Writes covariate summaries.
12. Writes the final lock manifest.
13. Writes a GEE loading helper script.

---

## 10.3 Pixel-native support cells

Step 04 prefers native raster bounds propagated from Step 01 to Step 03.

Preferred columns:

```text
ref_cell_xmin
ref_cell_ymin
ref_cell_xmax
ref_cell_ymax
ref_pixel_id
ref_grid_row
ref_grid_col
```

If these columns exist, Step 04 reconstructs exact native raster cells.

If they do not exist, Step 04 falls back to building square cells around reprojected lon/lat points.

The native-bound method is preferred because it preserves exact raster support.

---

## 10.4 Step 04 outputs

```text
04_reference_area/
├── reference_area_FINAL.gpkg
├── reference_area_FINAL.shp
├── reference_area_FINAL.geojson
├── matched_donor_pixels.gpkg
├── matched_donor_pixels.shp
├── reference_area_support_cells.gpkg
├── reference_area_support_cells.shp
├── monitoring_points_FIXED.gpkg
├── monitoring_points_FIXED.shp
├── monitoring_points_FIXED.csv
├── covariate_summary.csv
├── diagnostics.png
├── reference_area_FINAL_manifest.json
└── GEE_load_reference_area.js
```

Important files:

```text
reference_area_FINAL.gpkg
reference_area_FINAL_manifest.json
monitoring_points_FIXED.csv
GEE_load_reference_area.js
```

---

## 10.5 Step 04 validation checks

Before moving to Step 05, verify:

```text
[ ] reference_area_FINAL.gpkg exists.
[ ] reference_area_FINAL_manifest.json exists.
[ ] final reference area has non-zero area.
[ ] final reference area is based only on twin-tested matched controls.
[ ] monitoring_points_FIXED.csv exists.
[ ] support cells were generated correctly.
[ ] CRS is valid.
[ ] MultiPolygon output is accepted.
```

A MultiPolygon reference area is valid and expected when the reference area is made of disconnected matched pixel-support cells.

Avoid smoothing or generalizing the reference area unless all newly included pixels are revalidated.

---

# 11. Step 05 — Baseline confidence interval / UNCBSL

Canonical script:

```text
05_STARR_baseline_confidence_interval_UNCBSL.py
```

Purpose:

Step 05 calculates the 90% confidence interval of baseline carbon-stock change from the locked matched control/reference pixels and expresses the uncertainty as `UNCBSL`.

This step intentionally stops at baseline uncertainty. It does not calculate:

```text
activity removals
UNCAR
leakage
net verified removals
final issuable GSVERs
```

---

## 11.1 Formula

```text
SE_control = sigma_control / sqrt(N_control)
CI90_abs   = 1.645 * SE_control
UNCBSL     = CI90_abs / mean_deltaC_control
```

Where:

```text
sigma_control = standard deviation of ΔC in locked reference pixels
N_control     = number of unique locked reference/control pixels
mean_deltaC_control = mean annual carbon-stock change of locked reference pixels
```

---

## 11.2 Critical implementation rules

Step 05 must follow these rules:

```text
[ ] Use only locked matched control/reference pixels.
[ ] Do not use the full donor pool for uncertainty.
[ ] Deduplicate reference pixels before calculating N_control.
[ ] ΔC must be expressed in tC/ha/year.
[ ] If mean ΔC <= 0, report CI90 but flag UNCBSL as not applicable for baseline-removal crediting.
```

The full donor pool must never be used for `UNCBSL`.

`UNCBSL` is calculated only from the locked, twin-tested matched controls.

---

## 11.3 Accepted carbon-change schemas

Direct annual carbon-stock change columns:

```text
deltaC_control_tC_ha_yr
deltaC_ref_tC_ha_yr
deltaC_tC_ha_yr
delta_C_tC_ha_yr
dC_tC_ha_yr
```

Stock columns in `tC/ha`:

```text
C_ref_t0_tC_ha + C_ref_y_tC_ha
C_control_t0_tC_ha + C_control_y_tC_ha
carbon_t0_tC_ha + carbon_y_tC_ha
C_t0_tC_ha + C_y_tC_ha
```

AGB columns in `Mg dry matter/ha`:

```text
AGB_ref_t0_Mg_ha + AGB_ref_y_Mg_ha
AGB_control_t0_Mg_ha + AGB_control_y_Mg_ha
agb_t0_Mg_ha + agb_y_Mg_ha
AGB_t0_Mg_ha + AGB_y_Mg_ha
```

When AGB is used:

```text
C = AGB * BIOMASS_TO_CARBON_FRACTION * (1 + ROOT_TO_SHOOT_RATIO)
```

Default carbon fraction:

```python
BIOMASS_TO_CARBON_FRACTION = 0.47
```

---

## 11.4 Required carbon input file

Step 05 expects monitored carbon-stock change for the locked reference pixels.

A file named:

```text
control_carbon_stock_change.parquet
```

or:

```text
control_carbon_stock_change.csv
```

can be placed in one of these folders:

```text
STARR_outputs/<RUN_ID>/05_baseline_CI_UNCBSL/
STARR_outputs/<RUN_ID>/04_reference_area/
STARR_outputs/<RUN_ID>/03_twin_test/
STARR_outputs/<RUN_ID>/02_matching/
```

Alternatively, a dataframe can be passed directly to:

```python
run_baseline_ci_uncbsl(control_change_df=your_dataframe)
```

---

## 11.5 Step 05 outputs

```text
05_baseline_CI_UNCBSL/
├── baseline_control_deltaC_distribution.parquet
├── baseline_CI90_UNCBSL_summary.csv
├── baseline_CI90_UNCBSL_report.json
└── deltaC_control_CI90.png
```

Important files:

```text
baseline_CI90_UNCBSL_summary.csv
baseline_CI90_UNCBSL_report.json
deltaC_control_CI90.png
```

---

## 11.6 Step 05 path warning

The main runner currently calls Step 05 as:

```python
s05.run_baseline_ci_uncbsl()
```

This means Step 05 may use its internal:

```python
RUN_ID
BASE_DIR_CANDIDATES
```

Before running Step 05, make sure the internal Step 05 `RUN_ID` is aligned with the effective run ID created by Step 01.

Example:

```python
RUN_ID = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster_ext10km"
```

If this is not aligned, Step 05 may read carbon data or manifests from the wrong run directory.

Recommended improvement:

Modify the main runner so Step 05 receives the current run root explicitly.

Example:

```python
run_root = Path(out01).parent

control_pixels, ci_summary, report, fig05, out05 = s05.run_baseline_ci_uncbsl(
    base_dirs=[run_root],
    twin_df=twin_pixels,
    manifest=manifest,
)
```

This removes ambiguity and prevents Step 05 from using stale internal paths.

---

# 12. Alternative Step 05 — Extended baseline uncertainty

Alternative script:

```text
05_STARR_baseline_uncertainty.py
```

This script extends the baseline uncertainty calculation and includes additional baseline-removal related calculations such as:

```text
BR_unadj
DAF_TOTAL
BR_GOV_Y_TCO2E
PROJECT_AREA_HA
```

It also applies a no-avoided-degradation safeguard:

```python
APPLY_NO_AVOIDED_DEGRADATION_SAFEGUARD = True
```

Recommended repository handling:

```text
Keep only one canonical Step 05 in the main pipeline.
Rename the alternative script to:
05b_STARR_baseline_uncertainty_extended.py

or move it to:
archive/
```

This avoids confusion between:

```text
05_STARR_baseline_confidence_interval_UNCBSL.py
05_STARR_baseline_uncertainty.py
```

---

# 13. Optional Step 00b — Donor extent comparison

Script:

```text
00b_STARR_compare_extents.py
```

Purpose:

Runs Steps 01–04 across multiple donor extents and identifies the smallest extent that satisfies all sufficiency criteria.

Default tested extents:

```python
COMPARE_EXTENTS = ["full", 5, 10, 20, 30]
```

---

## 13.1 Sufficiency criteria

The extent is considered sufficient only if all criteria are satisfied:

```text
n_donor / n_project >= 3
match_coverage_pct >= 90%
twin_pass_pct >= 30%
smd_max <= 0.10
```

These thresholds are configured as:

```python
THRESH_RATIO_3X = 3.0
THRESH_MATCH_COV_PCT = 90.0
THRESH_TWIN_PASS_PCT = 30.0
THRESH_SMD_MAX = 0.10
```

---

## 13.2 Step 00b outputs

```text
comparison_summary.json
comparison_plots.png
recommendation.txt
```

The comparison runner reads metrics from:

```text
01_extract/extraction_report.json
02_matching/matching_summary.json
03_twin_test/twin_test_report.json
04_reference_area/reference_area_FINAL_manifest.json
```

---

## 13.3 How to interpret extent comparison

If the smallest sufficient extent is 10 km:

```python
DONOR_EXTENT_KM = 10
```

If no extent passes:

```text
[ ] check donor eligibility masks;
[ ] check FNF shapefile;
[ ] check Eligible_FNF shapefile;
[ ] check raster alignment;
[ ] check donor/project covariate overlap;
[ ] check hard calipers;
[ ] check NDVI trend quality;
[ ] consider testing extent > 30 km.
```

---

# 14. Recommended execution order

## 14.1 Single production run

Set the config in:

```text
00_STARR_run_steps_01_to_05.py
```

Then run:

```bash
python 00_STARR_run_steps_01_to_05.py
```

---

## 14.2 Donor extent comparison

Set:

```python
COMPARE_EXTENTS = ["full", 5, 10, 20, 30]
```

Then run:

```bash
python 00b_STARR_compare_extents.py
```

After reviewing the output recommendation, set the selected extent in:

```python
DONOR_EXTENT_KM = 10
```

or the selected value.

---

## 14.3 Manual step-by-step execution

Example:

```python
# Step 01
proj_df, donor_df, meta, out01 = s01.run_extraction(
    base_dirs=[BASE_DIR],
    donor_extent_km=DONOR_EXTENT_KM,
    run_id_base=RUN_ID_BASE,
    fnf_shapefile=FNF_SHAPEFILE,
    eligible_shapefile=ELIGIBLE_SHAPEFILE,
)

# Step 02
matched_df, weights, imp_df, smd_df, figs02, out02 = s02.run_matching_step(
    base_dirs=[out01],
    proj_df=proj_df,
    donor_df=None,
    meta=meta,
)

# Step 03
all_pairs, twin_pixels, twin_report, figs03, out03 = s03.run_twin_test(
    base_dirs=[out02],
    matched_df=matched_df,
    proj_df=proj_df,
    meta=meta,
)

# Step 04
bounds_gdf, mon_gdf, fig04, out04, manifest = s04.run_reference_area_lock(
    base_dirs=[out03],
    passed_df=twin_pixels,
    meta=meta,
    twin_report=twin_report,
)

# Step 05
run_root = Path(out01).parent

control_pixels, ci_summary, report, fig05, out05 = s05.run_baseline_ci_uncbsl(
    base_dirs=[run_root],
    twin_df=twin_pixels,
    manifest=manifest,
)
```

---

# 15. Pre-run checklist

Before running the pipeline:

```text
[ ] RUN_ID_BASE matches the GEE raster names.
[ ] Project raster exists: covariates_project_<RUN_ID_BASE>*.tif.
[ ] Donor raster exists: covariates_donor_<RUN_ID_BASE>*.tif.
[ ] BASE_DIR points to the folder containing the TIF files.
[ ] FNF_SHAPEFILE path is correct or explicitly set to None.
[ ] ELIGIBLE_SHAPEFILE path is correct or explicitly set to None.
[ ] FNF and Eligible shapefiles have valid CRS.
[ ] FNF and Eligible shapefiles overlap the raster extent.
[ ] Project and donor rasters have the expected CRS and resolution.
[ ] Project and donor rasters contain the same covariate bands.
[ ] WRB2_CODE exists.
[ ] Annual NDVI bands follow the format NDVI_YYYY.
[ ] DONOR_EXTENT_KM is selected intentionally.
[ ] If running Step 05, carbon-stock change data are available.
[ ] If running Step 05, RUN_ID inside Step 05 matches the effective run ID.
[ ] Output folder does not contain stale outputs from a different run.
```

---

# 16. Post-run checklist

After the run:

```text
[ ] 01_extract/project_pixels_raw.parquet exists.
[ ] 01_extract/donor_pixels_raw.parquet exists.
[ ] 01_extract/extraction_report.json exists.
[ ] 02_matching/reference_area_pixels.parquet exists.
[ ] 02_matching/matching_summary.json exists.
[ ] 02_matching/smd_balance.csv exists.
[ ] 03_twin_test/twin_tested_pixels.parquet exists.
[ ] 03_twin_test/twin_test_report.json exists.
[ ] 04_reference_area/reference_area_FINAL.gpkg exists.
[ ] 04_reference_area/reference_area_FINAL_manifest.json exists.
[ ] 04_reference_area/monitoring_points_FIXED.csv exists.
[ ] 05_baseline_CI_UNCBSL/baseline_CI90_UNCBSL_report.json exists, if Step 05 was run.
```

Recommended metric checks:

```text
[ ] n_donor / n_project >= 3.
[ ] match_coverage_pct >= 90%.
[ ] SMD_max <= 0.10.
[ ] twin_pass_pct >= 30%.
[ ] reference_area_ha > 0.
[ ] N_control is based on unique reference pixels.
```

---

# 17. Common failure points and fixes

## 17.1 No TIF found

Cause:

```text
RUN_ID_BASE does not match raster file names.
```

Fix:

Check that raster names follow:

```text
covariates_project_<RUN_ID_BASE>*.tif
covariates_donor_<RUN_ID_BASE>*.tif
```

Also check:

```text
BASE_DIR
file extension
hidden spaces
extra underscores
wrong run version
```

---

## 17.2 Project or donor dataframe is empty

Possible causes:

```text
wrong raster mask
wrong shapefile path
wrong shapefile CRS
shapefile does not overlap raster
donor extent too small
FNF mask removes everything
Eligible_FNF mask removes everything
```

Fix:

Run Step 01 without one spatial mask at a time:

```python
FNF_SHAPEFILE = None
ELIGIBLE_SHAPEFILE = None
```

Then re-enable one filter at a time.

---

## 17.3 `WRB2_CODE` missing

Cause:

```text
The GEE covariate raster does not include WRB2_CODE.
```

Fix:

Re-export the project and donor covariate rasters with the `WRB2_CODE` band.

---

## 17.4 Missing NDVI years

Cause:

```text
NDVI bands do not follow the NDVI_YYYY naming format.
```

Fix:

Rename or re-export bands using:

```text
NDVI_2013
NDVI_2014
NDVI_2015
NDVI_2016
NDVI_2017
```

---

## 17.5 Low match coverage

Possible causes:

```text
donor pool too small
donor extent too restrictive
eligible donor mask too restrictive
hard calipers too restrictive
wrong covariate scale
texture exact match too restrictive
KNN_QUERY_CANDIDATES too low
MAX_DONOR_REUSE too strict
```

Fix:

```text
[ ] increase DONOR_EXTENT_KM;
[ ] increase KNN_QUERY_CANDIDATES;
[ ] inspect FNF and Eligible_FNF masks;
[ ] check NDVI_t0 scale;
[ ] check SOC scale;
[ ] check distance-to-roads units;
[ ] check elevation/slope units;
[ ] review texture fallback only if methodologically justified.
```

---

## 17.6 SMD above 0.10

Cause:

```text
Matched controls are not balanced against project pixels.
```

Fix:

```text
[ ] improve donor extent;
[ ] check covariate transformations;
[ ] verify project/donor covariate distributions;
[ ] reduce donor reuse;
[ ] check if donor pool is ecologically comparable;
[ ] inspect outliers in continuous covariates.
```

---

## 17.7 Low twin-test pass rate

Cause:

```text
Matched controls do not have sufficiently parallel historical vegetation trends.
```

Fix:

```text
[ ] verify annual NDVI composites;
[ ] verify trend years;
[ ] check cloud/shadow masking;
[ ] check if PA and donor pool have different pre-baseline dynamics;
[ ] increase donor extent;
[ ] improve matching covariates;
[ ] review PAIR_SLOPE_DIFF_MAX only with justification.
```

---

## 17.8 Step 04 creates fragmented MultiPolygon

This is expected.

The final reference area is based on matched pixel-support cells. If matched pixels are spatially disconnected, the reference area will be a MultiPolygon.

Do not force a smoothed hull unless all included pixels are revalidated.

---

## 17.9 Step 05 reads wrong outputs

Cause:

```text
Step 05 internal RUN_ID does not match the effective run ID.
```

Fix:

Set:

```python
RUN_ID = "<RUN_ID_BASE>_ext10km"
```

or pass the run root explicitly from the main runner.

---

# 18. Methodological notes

The workflow follows these principles:

```text
[ ] The donor pool must represent eligible non-project control conditions.
[ ] Forest-at-T0 pixels are excluded from both project and donor where applicable.
[ ] Eligible donor pixels are constrained using the Eligible_FNF spatial mask.
[ ] Matching uses plain Mahalanobis distance with hard calipers.
[ ] RF-derived feature weights are intentionally not used.
[ ] Balance is checked using SMD.
[ ] The final reference area is built from matched and twin-tested donor/reference pixels only.
[ ] UNCBSL is calculated from locked matched controls, not from the full donor pool.
[ ] Duplicate control pixels are removed before uncertainty calculation.
[ ] Negative baseline carbon-stock change must not be converted into baseline removals.
```

---

# 19. GitHub implementation notes

Keep paths out of committed code where possible.

Recommended pattern:

```python
# config_local.py
# This file should be ignored by git.

RUN_ID_BASE = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster"
BASE_DIR = "/content/drive/MyDrive/STARR_Idiofa_New_V2"
DONOR_EXTENT_KM = 10
FNF_SHAPEFILE = "/content/drive/MyDrive/STARR_Idiofa_New_V2/FNF18_fullBuffer.shp"
ELIGIBLE_SHAPEFILE = "/content/drive/MyDrive/STARR_Idiofa_New_V2/Eligible_FNF_fullBuffer.shp"
```

Recommended `.gitignore`:

```text
# Local config
config_local.py
.env
*.env

# Outputs
STARR_outputs/
outputs/
logs/

# Heavy rasters
*.tif
*.tiff
*.vrt

# Vector data
*.gpkg
*.shp
*.shx
*.dbf
*.prj
*.cpg
*.qix

# Tables
*.parquet
*.csv

# Figures
*.png
*.jpg
*.jpeg

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

Commit:

```text
scripts
README
lightweight configuration templates
small example metadata files
```

Do not commit:

```text
raw rasters
generated parquet files
full shapefile outputs
large diagnostics
project-specific private paths
```

---

# 20. Minimal reproducible configuration example

Example production configuration:

```python
DONOR_EXTENT_KM = 10

RUN_ID_BASE = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster"

BASE_DIR = "/content/drive/MyDrive/STARR_Idiofa_New_V2"

FNF_SHAPEFILE = "/content/drive/MyDrive/STARR_Idiofa_New_V2/FNF18_fullBuffer.shp"

ELIGIBLE_SHAPEFILE = "/content/drive/MyDrive/STARR_Idiofa_New_V2/Eligible_FNF_fullBuffer.shp"
```

Expected raster files:

```text
/content/drive/MyDrive/STARR_Idiofa_New_V2/
├── covariates_project_Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster.tif
└── covariates_donor_Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster.tif
```

Expected output folder:

```text
/content/drive/MyDrive/STARR_Idiofa_New_V2/
└── STARR_outputs/
    └── Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster_ext10km/
```

If `DONOR_EXTENT_KM = "full"`:

```text
/content/drive/MyDrive/STARR_Idiofa_New_V2/
└── STARR_outputs/
    └── Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster/
```

---

# 21. Final operational rule

Every run must be traceable through one consistent naming chain:

```text
RUN_ID_BASE
→ covariates_project_<RUN_ID_BASE>.tif
→ covariates_donor_<RUN_ID_BASE>.tif
→ DONOR_EXTENT_KM
→ effective_run_id
→ STARR_outputs/<effective_run_id>/
→ Step 01 outputs
→ Step 02 outputs
→ Step 03 outputs
→ Step 04 locked reference area
→ Step 05 UNCBSL report
```

If this chain is broken, the result is not reproducible and may combine files from different runs.
