# =====================================================================
# §8. CONFRONTO BATCH — multi-sorgente × multi-periodo (automatico)
# =====================================================================
# Esegue Step 05 per OGNI combinazione (sorgente AGB × periodo), riusando
# la stessa Reference Area bloccata (Step 01-04 girano UNA volta: il matching
# è su covariate statiche, non sull'AGB). Salva i risultati in un folder
# dedicato, distinti per input e per anni, + un CSV/JSON master di confronto.
#
# Richieste implementate:
#   - ALLOW_NEGATIVE_BASELINE = True  (baseline calcolato anche se <= 0, richiesta PM)
#   - ROOT_TO_SHOOT_RATIO     = 0.4   (shrubland)
# =====================================================================
import json, time, traceback
from pathlib import Path
import pandas as pd

assert s05 is not None, "Step 05 script not loaded (s05 is None)."
assert 'twin_pixels' in globals(), "twin_pixels missing: run Step 01-04 first."

# Risolvi manifest/area come nella cella §7
_manifest = globals().get('manifest', None)
if _manifest is None:
    _mpath = OUTPUT_ROOT / '04_reference_area' / 'reference_area_FINAL_manifest.json'
    _manifest = json.load(open(_mpath)) if _mpath.exists() else None
_area_ha = float(globals().get('PROJECT_AREA_HA_RESOLVED',
                                globals().get('PROJECT_AREA_HA', 0.0)))
assert _area_ha > 0, "PROJECT_AREA_HA not resolved (>0)."

# ── Override espliciti richiesti ─────────────────────────────────────
s05.ALLOW_NEGATIVE_BASELINE = True       # baseline anche sotto zero (PM)
s05.ROOT_TO_SHOOT_RATIO     = 0.4        # shrubland BGB
print(f"ALLOW_NEGATIVE_BASELINE = {s05.ALLOW_NEGATIVE_BASELINE} | "
      f"ROOT_TO_SHOOT_RATIO = {s05.ROOT_TO_SHOOT_RATIO}")

DRY_RUN = False   # True = controlla solo l'esistenza dei file, non esegue

# ── Sorgenti AGB: cartella + template nome file (serializzato su {y}) ──
DRIVE = Path('/content/drive/MyDrive')
AGB_SOURCES = {
    'GEDI_embedding': {
        'dir':     DRIVE / 'STARR_AGB_maps_GEDI_embedding',
        'donor':   'AGB_DONOR_{y}_GEDI_RF.tif',
        'project': 'AGB_PROJECT_{y}_GEDI_RF.tif',
    },
    'multiyear_embedding': {
        'dir':     DRIVE / 'STARR_AGB_maps_multiyear_embedding',
        'donor':   'AGB_DONOR_{y}_RF.tif',
        'project': 'AGB_PROJECT_{y}_RF.tif',
    },
    'standard': {
        'dir':     DRIVE / 'STARR_AGB_maps',
        'donor':   'AGB_DONOR_{y}.tif',
        'project': 'AGB_PROJECT_{y}.tif',
    },
    'phenology': {
        'dir':     DRIVE / 'STARR_AGB_maps',
        'donor':   'AGB_DONOR_{y}_Phen.tif',
        'project': 'AGB_PROJECT_{y}_Phen.tif',
    },
}

# ── Periodi (start, end), tutti >= 4 anni ────────────────────────────
PERIODS = [
    (2018, 2022), (2018, 2023), (2018, 2024), (2018, 2025),
    (2019, 2023), (2019, 2024), (2019, 2025), (2020, 2024),
]

# ── Folder di output dedicato ────────────────────────────────────────
_stamp    = time.strftime('%Y%m%d_%H%M%S')
BATCH_DIR = COMPARISON_DIR / f'batch_AGB_comparison_{_stamp}'
BATCH_DIR.mkdir(parents=True, exist_ok=True)
print('BATCH_DIR :', BATCH_DIR)

def _spec(folder, template, y0, y1):
    d = AGB_SOURCES[folder]['dir']
    return {
        'delta_raster': None, 'stock_raster': None,
        'stock_t0_raster': str(d / template.format(y=y0)),
        'stock_y_raster':  str(d / template.format(y=y1)),
        'stock_t0_band': 1, 'stock_y_band': 1,
        'stock_t0_band_name': None, 'stock_y_band_name': None,
        'units': 'AGB_Mg_ha',
    }

def _curate(src_key, y0, y1, s):
    _co2e = 44.0 / 12.0
    proj_tC = s.get('project_observed_total_tC_period')
    surp_tC = s.get('project_minus_unadjusted_baseline_tC_period')
    return {
        'agb_source':            src_key,
        't0_year':               y0,
        'monitoring_year':       y1,
        'period_years':          y1 - y0,
        'mean_ctrl_raw_tC_ha_yr':   s.get('unadjusted_baseline_mean_raw_tC_ha_yr'),
        'delta_C_ref_tC_ha_yr':     s.get('delta_C_ref_y_tC_ha_yr'),
        'BL_unadj_tCO2e_period':    s.get('BL_unadj_period_tCO2e'),
        'BL_unadj_tC_period':       s.get('BL_unadj_period_tC'),
        'project_tC_period':        proj_tC,
        'project_tCO2e_period':     (proj_tC * _co2e) if proj_tC is not None else None,
        'surplus_tC_period':        surp_tC,
        'surplus_tCO2e_period':     (surp_tC * _co2e) if surp_tC is not None else None,
        'uncbsl_percent':           s.get('uncbsl_percent'),
        'uncbsl_fraction':          s.get('uncbsl_fraction'),
        'ci90_final_tC_ha_yr':      s.get('ci90_abs_final_tC_ha_yr'),
        'ci90_final_total_tC_period': s.get('ci90_abs_final_total_tC_period'),
        'ci90_final_source':        s.get('ci90_final_source'),
        'n_ctrl_rows':              s.get('n_control_matched_valid_rows'),
        'n_ctrl_eff_se':            s.get('n_control_effective_for_se'),
        'n_pa_unique':              s.get('n_project_unique_pixels_represented'),
        'project_area_ha':          s.get('project_area_ha'),
        'baseline_status':          s.get('baseline_uncertainty_adjustment_status'),
        'allow_negative_baseline':  s.get('allow_negative_baseline'),
    }

rows, skipped, errors = [], [], []
n_total = len(AGB_SOURCES) * len(PERIODS)
n_done = 0

for src_key, cfg in AGB_SOURCES.items():
    for (y0, y1) in PERIODS:
        n_done += 1
        tag = f'{src_key} | {y0}-{y1}'
        donor_spec   = _spec(src_key, cfg['donor'],   y0, y1)
        project_spec = _spec(src_key, cfg['project'], y0, y1)

        # Controllo esistenza dei 4 file
        paths = [donor_spec['stock_t0_raster'], donor_spec['stock_y_raster'],
                 project_spec['stock_t0_raster'], project_spec['stock_y_raster']]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            skipped.append({'run': tag, 'missing': missing})
            print(f'[{n_done}/{n_total}] SKIP {tag} — missing files: {len(missing)}')
            continue

        if DRY_RUN:
            print(f'[{n_done}/{n_total}] OK (dry-run) {tag}')
            continue

        out_dir = BATCH_DIR / src_key / f'{y0}_{y1}'
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            _, _, ci_summary, ci_report, _, _ = s05.run_baseline_ci_uncbsl_from_rasters(
                base_dirs                   = [OUTPUT_ROOT],
                output_dir                  = out_dir,
                donor_raster_spec           = donor_spec,
                project_raster_spec         = project_spec,
                twin_df                     = twin_pixels,
                project_df                  = None,
                project_area_ha             = _area_ha,
                use_all_matched_rows        = USE_ALL_MATCHED_ROWS,
                use_matched_project_rows    = USE_MATCHED_PROJECT_ROWS,
                manifest                    = _manifest,
                monitoring_period_years     = float(y1 - y0),
                t0_year                     = y0,
                monitoring_year             = y1,
                min_monitoring_period_years = MIN_MONITORING_PERIOD_YEARS,
                use_conservative_block_ci   = USE_CONSERVATIVE_BLOCK_CI,
                spatial_block_size_m        = SPATIAL_BLOCK_SIZE_M,
                allow_negative_baseline     = True,
                verbose                     = False,
            )
            row = _curate(src_key, y0, y1, ci_summary)
            rows.append(row)
            # salva il summary completo per-run
            with open(out_dir / 'ci_summary.json', 'w') as f:
                json.dump(ci_summary, f, indent=2, default=str)
            print(f"[{n_done}/{n_total}] OK   {tag} | "
                  f"BL={row['BL_unadj_tCO2e_period']:,.0f}  "
                  f"surplus={row['surplus_tCO2e_period']:,.0f} tCO2e")
        except Exception as e:
            errors.append({'run': tag, 'error': str(e)})
            print(f'[{n_done}/{n_total}] ERR  {tag} — {e}')
            with open(out_dir / 'ERROR.txt', 'w') as f:
                f.write(traceback.format_exc())

# ── Master CSV/JSON ──────────────────────────────────────────────────
if rows:
    master = pd.DataFrame(rows).sort_values('surplus_tCO2e_period', ascending=False)
    master.to_csv(BATCH_DIR / 'comparison_master.csv', index=False)
    master.to_json(BATCH_DIR / 'comparison_master.json', orient='records', indent=2)
    print(f'\nMaster saved: {BATCH_DIR / "comparison_master.csv"}  ({len(master)} run)')
    display(master)
else:
    print('\nNo run completed. Check skipped/errors below.')

if skipped:
    print(f'\n{len(skipped)} runs skipped (missing files):')
    for s in skipped: print('  -', s['run'])
if errors:
    print(f'\n{len(errors)} runs with error:')
    for e in errors: print('  -', e['run'], '::', e['error'][:80])

with open(BATCH_DIR / 'batch_log.json', 'w') as f:
    json.dump({'completed': len(rows), 'skipped': skipped, 'errors': errors,
               'periods': PERIODS, 'sources': list(AGB_SOURCES.keys()),
               'allow_negative_baseline': True, 'root_to_shoot_ratio': 0.4}, f, indent=2)
