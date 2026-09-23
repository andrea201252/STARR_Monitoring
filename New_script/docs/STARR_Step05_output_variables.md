# STARR Step 05 — Output variables (Unadjusted Baseline + CI90 / UNCBSL)

Reference for the fields written by **Step 05**
(`05_STARR_baseline_confidence_interval_UNCBSL.py`).

Step 05 computes the **unadjusted baseline** `BL_unadj` (what the control /
reference area would remove, scaled to the project area) and its **uncertainty
(UNCBSL / CI90)**. It does **not** compute the final GSVERs: DAF, `BR_gov`
comparison (Eq-6), leakage, buffer deductions are the PM's scope.

## Where the files are

Per single run (notebook §7):
```
<OUTPUT_ROOT>/05_baseline_CI_UNCBSL.../
├── baseline_CI90_UNCBSL_report.json   # full report (summary + pm_handoff_crediting + provenance)
├── baseline_CI90_UNCBSL_summary.csv   # one-row flat summary (all summary fields)
├── control_pixels.parquet/csv         # sampled control (reference) pixels + ref_deltaC
├── project_pixels.parquet/csv         # sampled PA pixels + proj_deltaC
└── *.png                              # CI distribution + project-vs-control plots
```
Multi-source × multi-period sweep (notebook §8):
```
<COMPARISON_DIR>/batch_AGB_comparison_<timestamp>/
├── comparison_master.csv / .json      # one row per (AGB source × period) — see last table
└── <source>/<t0>_<y>/ci_summary.json  # full summary per run
```

Sign convention: values are **positive** when carbon **increases** (removal).
`tC` = tonnes of carbon; `tCO2e` = `tC × 44/12` (≈ 3.6667).

---

## 1. Identity & temporal

| Field | Unit | Meaning |
|---|---|---|
| `run_id` | – | Run identifier (from the notebook §1 `RUN_ID`). |
| `project_name` | – | Project name (from the notebook §1 `PROJECT_NAME`). |
| `timestamp_utc` | ISO | When the run was produced. |
| `methodology`, `step` | – | Fixed labels (`GS STARR Track 1 SEMDB`, step name). |
| `t0_year` | year | Baseline start year (T0). |
| `monitoring_year` | year | Monitoring/end year. |
| `monitoring_period_years` | years | `monitoring_year − t0_year`. |
| `minimum_monitoring_period_years` | years | Minimum allowed period (default 4). |
| `first_valid_monitoring_year` | year | First year at which a stock-difference endpoint is valid (T0 + min period). |
| `t0_cumulative_unadjusted_baseline_tC` | tC | Cumulative baseline at T0 (= 0 by definition). |

## 2. Sample sizes (matched rows / effective N)

| Field | Meaning |
|---|---|
| `n_rows_control_before_deduplication` | Control rows before any filtering. |
| `n_control_matched_valid_rows` | Valid matched control rows (non-null ΔC) used for the mean. |
| `n_control_unique_reference_pixels_represented` | Distinct reference pixels behind those rows. |
| `n_control_effective_for_se` | **Effective N** used for the standard error / CI (unique reference pixels — anti pseudo-replication, fix A1). |
| `n_control_spatial_blocks` / `n_spatial_blocks` | Number of spatial blocks used for the block CI. |
| `n_rows_project_before_deduplication` | Project rows before filtering. |
| `n_project_matched_valid_rows` | Valid matched PA rows (non-null ΔC). |
| `n_project_unique_pixels_represented` | Distinct PA pixels represented. |
| `control_row_selection_mode`, `project_row_selection_mode` | How rows were selected (e.g. `all_matched_rows_no_deduplication`). |

## 3. Unadjusted baseline — the core output

`BL_unadj,y = ΔC_ref,y × A_project` (GS STARR Eq-31a).

| Field | Unit | Meaning |
|---|---|---|
| `mean_deltaC_control_tC_ha_yr` | tC/ha/yr | **Raw** mean ΔC of the control area (before flooring at 0). |
| `unadjusted_baseline_mean_raw_tC_ha_yr` | tC/ha/yr | Same raw mean (diagnostic). |
| `delta_C_ref_y_tC_ha_yr` | tC/ha/yr | **ΔC_ref,y creditable** = `max(mean_control, 0)` (the baseline rate used for crediting; unless `allow_negative_baseline`). |
| `project_area_ha` / `project_area_ha_used` | ha | Full PA area the baseline is scaled to. |
| `unadjusted_baseline_total_tC_yr` | tC/yr | `ΔC_ref,y × A_project`. |
| `unadjusted_baseline_total_tC_period` | tC | × monitoring period. |
| `BL_unadj_y_tC` | tC/yr | Baseline per year (tC). |
| `BL_unadj_y_tCO2e` | tCO2e/yr | **Primary PM value, per year.** |
| `BL_unadj_period_tCO2e` | tCO2e | **Primary PM value, whole period.** |
| `unadjusted_baseline_total_raw_tC_yr` / `_period` | tC | Raw (unfloored) totals (diagnostic). |
| `unadjusted_baseline_negative_control_rule` | – | Note on how a negative control mean is handled. |
| `allow_negative_baseline` | bool | If True the baseline is kept even when the control mean ≤ 0 (PM request). |

## 4. Confidence interval & UNCBSL

| Field | Unit | Meaning |
|---|---|---|
| `ci90_z_value` | – | z used for the 90% CI (≈ 1.645). |
| `se_control_tC_ha_yr` | tC/ha/yr | Standard error on the effective N. |
| `se_control_naive_all_rows_tC_ha_yr` | tC/ha/yr | Naive SE on all rows (underestimate, diagnostic). |
| `ci90_abs_pixel_tC_ha_yr` | tC/ha/yr | Pixel-based CI90 half-width (effective N). |
| `ci90_abs_pixel_naive_all_rows_tC_ha_yr` | tC/ha/yr | Naive pixel CI (diagnostic). |
| `ci90_abs_block_tC_ha_yr` | tC/ha/yr | Block-based CI90 (spatial autocorrelation). |
| `ci90_abs_final_tC_ha_yr` | tC/ha/yr | **Final CI90 half-width** = larger of pixel/block when `use_conservative_block_ci`. |
| `ci90_final_source` | – | Which CI was chosen (`pixel_based` / `block_conservative`). |
| `ci90_lower/upper_final_tC_ha_yr` | tC/ha/yr | Final CI bounds. |
| `ci90_abs_final_total_tC_period` / `_yr` | tC | Final CI scaled to area × period. |
| `uncbsl_fraction` | – | **UNCBSL** = CI ÷ baseline (uncertainty ratio, Eq-6 factor — *not* the baseline). |
| `uncbsl_percent` | % | Same as a percentage. |
| `uncbsl_status` | – | `computed…` or `not_applicable…` (e.g. baseline ≤ 0). |
| `baseline_uncertainty_adjusted_total_tC_period` | tC | Conservative baseline = baseline + CI (period). |
| `baseline_uncertainty_adjustment_status` | – | Status of that adjustment. |
| `baseline_uncertainty_double_count_warning` | text | **B5** warning: do not apply the CI twice (once here, once via DAF). |

## 5. Project observed & net (additionality)

| Field | Unit | Meaning |
|---|---|---|
| `mean_deltaC_project_weighted_tC_ha_yr` | tC/ha/yr | Mean ΔC of the PA (project) pixels. |
| `project_observed_total_tC_period` | tC | **Project gross removals** over the period. |
| `project_observed_total_tCO2e_period` | tCO2e | Same in CO2e. |
| `project_minus_unadjusted_baseline_tC_period` | tC | **Net** = project − unadjusted baseline (positive ⇒ additional). |
| `project_minus_unadjusted_baseline_raw_tC_*` | tC | Net using the raw (unfloored) baseline (diagnostic). |
| `project_area_scaling_method` | – | How the matched-row mean is scaled to the full PA. |

## 6. PM handoff crediting (`pm_handoff_crediting` block — Eq-6 / Eq-22)

Illustrative only; the PM finalizes with `BR_gov` and the real DAF.

| Field | Unit | Meaning |
|---|---|---|
| `monitoring_period_years`, `project_area_ha` | – | Echoed for convenience. |
| `UNCBSL_fraction`, `UNCBSL_percent`, `UNCBSL_status` | – | Baseline uncertainty (Eq-6 factor). |
| `CI90_abs_final_tCO2e_ha_yr`, `CI90_source` | tCO2e/ha/yr | Final CI in CO2e. |
| `project_observed_total_tCO2e_period` | tCO2e | Project gross removals. |
| `illustrative_DAF_used` | – | DAF applied in the illustrative formula (placeholder). |
| `illustrative_BR_crediting_tCO2e_period` | tCO2e | `BL_unadj × (1 + DAF) × (1 + UNCBSL)` — the conservative crediting baseline. |
| `illustrative_note` | text | The formula + reminder to take `MAX(BR_STARR, BR_gov)`. |
| `pm_actions_required` | list | Steps the PM must still perform. |

## 7. Units, conversion & carbon chemistry (AGB→C)

| Field | Meaning |
|---|---|
| `units` / `source_units` / `target_units` | Input raster units and target (`AGB_Mg_ha` → `tC_ha`). |
| `biomass_to_carbon_fraction` (CF) | Carbon fraction of biomass (≈ 0.47). |
| `root_to_shoot_ratio` (R) | BGB via root:shoot ratio. |
| `bgb_included`, `bgb_decision_note` | Whether BGB is included and the B6 note. |
| `tC_to_tCO2e_factor` | 44/12 ≈ 3.6667. |
| `pixel_area_ha` / `pixel_area_ha_constant` | Raster pixel area. |

## 8. Provenance / status (report only)

`purpose`, `reference_area_manifest_summary`, `reference_area_definition`,
`lock_status`, `area_source`, `area_is_full_project_area`,
`block_status` / `block_ci_status` / `block_coordinate_source`,
`se_effective_n_note`, `control_deltaC_distribution`,
`project_deltaC_distribution` — descriptive metadata for the audit trail.

---

## 9. Batch comparison CSV (`comparison_master.csv`, notebook §8)

One row per **(AGB source × period)**. Curated subset of the summary:

| Column | Unit | Meaning |
|---|---|---|
| `agb_source` | – | AGB dataset key (e.g. `GEDI_embedding`). |
| `t0_year`, `monitoring_year`, `period_years` | year(s) | Period tested. |
| `mean_ctrl_raw_tC_ha_yr` | tC/ha/yr | Raw control mean ΔC. |
| `delta_C_ref_tC_ha_yr` | tC/ha/yr | Creditable baseline rate. |
| `BL_unadj_tCO2e_period` / `BL_unadj_tC_period` | tCO2e / tC | Unadjusted baseline over the period. |
| `project_tC_period` / `project_tCO2e_period` | tC / tCO2e | Project gross removals. |
| `surplus_tC_period` / `surplus_tCO2e_period` | tC / tCO2e | **Net** (project − baseline). Table is sorted by this, descending. |
| `uncbsl_percent` / `uncbsl_fraction` | % / – | Baseline uncertainty. |
| `ci90_final_tC_ha_yr` / `ci90_final_total_tC_period` | tC/ha/yr / tC | Final CI. |
| `ci90_final_source` | – | `pixel_based` / `block_conservative`. |
| `n_ctrl_rows`, `n_ctrl_eff_se`, `n_pa_unique` | count | Sample sizes (rows, effective N, unique PA pixels). |
| `project_area_ha` | ha | PA area used. |
| `baseline_status` | – | Baseline-adjustment status. |
| `allow_negative_baseline` | bool | Whether negative baselines were kept. |

---

## 10. Biomass (AGB) statistics — `biomass_statistics` block  *(new)*

Descriptive statistics of the **raw sampled stock** (biomass when
`source_units = AGB_Mg_ha`) for the **project (PA)** and the **donor/reference**
pixels, at **T0** and at the **monitoring year (Ty)**. Carbon is then
`value × CF × (1 + R)`; these are the pre-conversion biomass values.

In the JSON report under the top-level key `biomass_statistics`:

| Field | Meaning |
|---|---|
| `source_units` | Raster units of the sampled stock (e.g. `AGB_Mg_ha`). |
| `t0_year`, `monitoring_year` | The two epochs. |
| `project_t0`, `project_monitoring` | Stats of PA biomass at T0 and Ty. |
| `donor_t0`, `donor_monitoring` | Stats of donor/reference biomass at T0 and Ty. |
| `project_mean_change_t0_to_monitoring` | Mean PA biomass change T0→Ty (same units). |
| `donor_mean_change_t0_to_monitoring` | Mean donor biomass change T0→Ty. |

Each group is reported in **three units**, each with `n`, `mean`, `std`, `min`,
`p25`, `median`, `p75`, `max`, `sum`:

| Sub-key | Unit | From |
|---|---|---|
| `biomass` | source units (e.g. `AGB_Mg_ha`) | raw sampled stock |
| `carbon_tC_ha` | tC/ha | biomass × CF × (1+R) |
| `co2e_tCO2e_ha` | tCO2e/ha | carbon × 44/12 |

`*_mean_change_t0_to_monitoring` carries the T0→Ty mean change in all three units
(`biomass`, `carbon_tC_ha`, `co2e_tCO2e_ha`). `tC_to_tCO2e_factor` = 44/12.

### Flat columns in `baseline_CI90_UNCBSL_summary.csv`
For each group (`project_t0` / `project_monitoring` / `donor_t0` /
`donor_monitoring`): `agb_<group>_n`, and `<u>_<group>_{mean,std,min,median,max}`
for `<u>` in `agb` (biomass), `tc` (carbon), `tco2e` (CO2e); plus
`{agb,tc,tco2e}_project_mean_change` and `{agb,tc,tco2e}_donor_mean_change`,
and `agb_source_units`, `tC_to_tCO2e_factor`.

### Plot
`biomass_project_vs_donor.png` — overlaid histograms of project vs donor biomass
at T0 and the monitoring year (referenced in the report under
`outputs.fig_biomass_distribution`).

### Per-pixel biomass
The raw per-pixel biomass at both epochs is in the distribution files:
`*_control_deltaC_distribution.*` has `ref_stock_t0_raw`, `ref_stock_y_raw`;
`*_project_deltaC_distribution.*` has `proj_stock_t0_raw`, `proj_stock_y_raw`
(plus the derived `*_C_t0_tC_ha`, `*_C_y_tC_ha`, `*_deltaC_tC_ha_yr`).
