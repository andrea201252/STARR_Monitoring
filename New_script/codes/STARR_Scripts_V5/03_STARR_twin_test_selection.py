# -*- coding: utf-8 -*-
"""
GS STARR – Track 1 SEMDB | 03_STARR_twin_test_selection.py
STEP 03 — Test Aggregato Trend Parallelo / Twin Test
[v2 — OLS vettorizzato + make_long veloce + singola costruzione long_df nel loop iterativo]

Modifiche rispetto a v1
───────────────────────
PERF 1  _ols_vectorized: sostituito il loop Python per-pixel con un broadcast
         numpy completo; gestisce le maschere NaN per-riga tramite somma mascherata.
         100-1000× più veloce per grandi set di coppie.
PERF 2  make_long_for_interaction: sostituito il loop Python + 20× pd.concat con
         numpy np.repeat/np.tile e un singolo costruttore DataFrame.
PERF 3  select_pairs_with_parallel_trend: costruisce valid_long_df una sola volta;
         il loop iterativo sulla frazione filtra con np.isin invece di ricostruire.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


def _is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell","Shell")
    except ImportError:
        return False

if not _is_notebook():
    matplotlib.use("Agg")


# ================================================================
# PARAMETRI
# ================================================================

PVALUE_THRESHOLD      = 0.05
PAIR_SLOPE_DIFF_MAX   = 0.005

# Frazione MINIMA di coppie che la selezione del parallel-trends test deve
# mantenere. GS STARR NON richiede una frazione minima (chiede solo trend
# paralleli p>0.05 sui pixel selezionati): quindi 0.0 = nessun pavimento sul %.
# Il solo limite rimasto è MIN_SELECTED_PAIRS (minimo ASSOLUTO di coppie, per
# validità statistica — non una percentuale).
MIN_SELECTED_FRACTION = 0.0
MIN_SELECTED_PAIRS    = 30

# Ricerca iterativa: frazioni esplorate (dal 95% giù). Generata da config, non
# più hard-coded con pavimento al 30%. Il minimo assoluto di coppie resta
# garantito da MIN_SELECTED_PAIRS.
SELECTION_FRAC_MAX  = 0.95
SELECTION_FRAC_MIN  = 0.05
SELECTION_FRAC_STEP = 0.05

GRID_N_EXAMPLES       = 16
GRID_SEED             = 42

# Sottocartella di output di questo step — EDITABILE DAL NOTEBOOK (s03.STEP_DIRNAME).
# Il livello OUTPUTS_DIRNAME/<RUN_ID> è ereditato dal path di Step 01/02.
STEP_DIRNAME = "03_twin_test"


# ================================================================
# FUNZIONI IO
# ================================================================

def load_df(path):
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"File not found: {p}")
    return pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)

def find_file(base_dirs, stem):
    for base in base_dirs:
        base = Path(base)
        for ext in ["parquet","csv"]:
            p = base/f"{stem}.{ext}"
            if p.exists(): print(f"    Found: {p.name}"); return p
    raise FileNotFoundError(f"'{stem}' not found in {base_dirs}")

def detect_ndvi_year_cols(columns):
    import re
    return sorted([c for c in columns if re.match(r"^NDVI_\d{4}$",str(c))])

def years_from_cols(ndvi_year_cols):
    return [int(str(c).split("_")[1]) for c in ndvi_year_cols]


# ================================================================
# OLS — completamente vettorizzato (PERF 1)
# ================================================================

def _ols_vectorized(Y, years, min_valid):
    """
    OLS vettorizzato: slope, p-value, valid_count per ogni riga di Y.
    Nessun loop Python per-riga. Gestisce i NaN per-riga tramite somma mascherata.

    Parametri
    ----------
    Y         : (n_pixels, n_years) array-like
    years     : (n_years,)
    min_valid : int

    Restituisce (slopes, pvalues, valid_cnt) — ciascuno ndarray (n_pixels,)
    """
    years = np.asarray(years, dtype=np.float64)
    Y     = np.asarray(Y,     dtype=np.float64)
    n_px, n_yrs = Y.shape

    finite    = np.isfinite(Y)              # (n_px, n_yrs)
    valid_cnt = finite.sum(axis=1)          # (n_px,)
    ok_px     = valid_cnt >= int(min_valid)

    # Azzera le posizioni non valide per somme vettorizzate sicure
    Y_s  = np.where(finite, Y,                  0.0)
    yr_s = np.where(finite, years[np.newaxis,:], 0.0)

    ni      = valid_cnt.astype(np.float64)
    ni_safe = np.where(ok_px, ni, 1.0)

    y_mean = Y_s.sum(axis=1) / ni_safe
    x_mean = yr_s.sum(axis=1) / ni_safe

    xc = np.where(finite, years[np.newaxis,:] - x_mean[:,np.newaxis], 0.0)
    yc = np.where(finite, Y               - y_mean[:,np.newaxis], 0.0)

    ssx  = (xc*xc).sum(axis=1)
    scov = (xc*yc).sum(axis=1)

    ssx_safe = np.where(ssx > 1e-12, ssx, 1.0)
    valid_ok  = ok_px & (ssx > 1e-12)
    slopes    = np.where(valid_ok, scov / ssx_safe, np.nan)
    icepts    = np.where(valid_ok, y_mean - slopes*x_mean, np.nan)

    y_hat = slopes[:,np.newaxis]*years[np.newaxis,:] + icepts[:,np.newaxis]
    resid = np.where(finite, Y - y_hat, 0.0)
    sse   = (resid*resid).sum(axis=1)

    dof    = np.maximum(valid_cnt - 2, 1).astype(np.float64)
    se_sq  = np.where(valid_ok, sse / dof / ssx_safe, np.nan)
    se     = np.sqrt(np.maximum(se_sq, 0.0))
    t_stat = np.where(se > 1e-15, slopes/se, 0.0)

    # scipy.stats.t.sf accetta array
    pvalues = np.where(
        valid_ok,
        2.0 * stats.t.sf(np.abs(t_stat), df=np.maximum(valid_cnt-2, 1)),
        np.nan
    )

    return slopes, pvalues, valid_cnt.astype(int)


# ================================================================
# FORMATO LONG — reshape numpy (PERF 2)
# ================================================================

def make_long_for_interaction(df, ndvi_year_cols):
    """
    Costruisce il DataFrame long [pair_id, year, group, ndvi].
    group=1 project, group=0 reference.
    Nessun loop Python; nessun pd.concat; singolo costruttore DataFrame da numpy.
    """
    if df.empty or not ndvi_year_cols:
        return pd.DataFrame(columns=["pair_id","year","group","ndvi"])

    years   = np.array([int(c.split("_")[1]) for c in ndvi_year_cols], dtype=np.float64)
    n_pairs = len(df)
    n_yrs   = len(years)

    proj_cols = [f"proj_{c}" for c in ndvi_year_cols if f"proj_{c}" in df.columns]
    ref_cols  = [f"ref_{c}"  for c in ndvi_year_cols if f"ref_{c}"  in df.columns]

    proj_vals = df[proj_cols].to_numpy(dtype=np.float64).ravel() \
                if proj_cols else np.full(n_pairs*n_yrs, np.nan)
    ref_vals  = df[ref_cols].to_numpy(dtype=np.float64).ravel()  \
                if ref_cols  else np.full(n_pairs*n_yrs, np.nan)

    pair_ids  = np.repeat(df.index.to_numpy(), n_yrs)
    years_rep = np.tile(years, n_pairs)
    N = n_pairs * n_yrs

    return pd.DataFrame({
        "pair_id": np.concatenate([pair_ids,  pair_ids]),
        "year":    np.concatenate([years_rep, years_rep]),
        "group":   np.concatenate([np.ones(N), np.zeros(N)]),
        "ndvi":    np.concatenate([proj_vals,  ref_vals]),
    })


# ================================================================
# TEST DI INTERAZIONE + PENDENZA APPAIATA
# ================================================================

def interaction_test_from_long(long_df):
    """OLS y = b0 + b1*group + b2*year_c + b3*group*year_c."""
    df = long_df[["ndvi","group","year"]].dropna().copy()
    if len(df)<8 or df["group"].nunique()<2 or df["year"].nunique()<3:
        return {"n_obs":int(len(df)),"interaction_beta":np.nan,
                "interaction_pvalue":np.nan,"passed":False}
    y  = df["ndvi"].astype(float).values
    g  = df["group"].astype(float).values
    yc = df["year"].astype(float).values; yc -= yc.mean()
    X  = np.column_stack([np.ones(len(df)), g, yc, g*yc])
    try:
        beta  = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X@beta
        dof   = len(y) - X.shape[1]
        if dof <= 0: raise ValueError("dof<=0")
        s2  = float((resid@resid)/dof)
        cov = s2 * np.linalg.inv(X.T@X)
        se  = np.sqrt(np.maximum(np.diag(cov),1e-30))
        t   = float(beta[3]/se[3])
        p   = 2.0*float(stats.t.sf(abs(t),df=dof))
        return {"n_obs":int(len(df)),"interaction_beta":float(beta[3]),
                "interaction_t":t,"interaction_pvalue":p,"passed":bool(p>PVALUE_THRESHOLD)}
    except Exception:
        return {"n_obs":int(len(df)),"interaction_beta":np.nan,
                "interaction_pvalue":np.nan,"passed":False}


def paired_slope_test(df):
    d = (df["proj_slope"]-df["ref_slope"]).replace([np.inf,-np.inf],np.nan).dropna().astype(float)
    if len(d)<3:
        return {"n_pairs":int(len(d)),"mean_slope_diff":np.nan,"pvalue":np.nan,"passed":False}
    t, p = stats.ttest_1samp(d.values, popmean=0.0, nan_policy="omit")
    return {"n_pairs":int(len(d)),"mean_slope_diff":float(d.mean()),
            "median_abs_slope_diff":float(np.median(np.abs(d.values))),
            "pvalue":float(p),
            "passed":bool(p>PVALUE_THRESHOLD and abs(float(d.mean()))<=PAIR_SLOPE_DIFF_MAX)}


# ================================================================
# PREPARAZIONE DATI
# ================================================================

def attach_project_ndvi(matched_df, proj_df, ndvi_year_cols):
    out = matched_df.copy()
    for c in ndvi_year_cols:
        if c in out.columns and f"ref_{c}" not in out.columns:
            out = out.rename(columns={c:f"ref_{c}"})
    join_cols = ["lon","lat"]+[c for c in ndvi_year_cols if c in proj_df.columns]
    pndvi = proj_df[join_cols].rename(columns={"lon":"proj_lon","lat":"proj_lat"})
    pndvi = pndvi.rename(columns={c:f"proj_{c}" for c in ndvi_year_cols if c in pndvi.columns})
    out["_plon_key"]   = out["proj_lon"].round(7)
    out["_plat_key"]   = out["proj_lat"].round(7)
    pndvi["_plon_key"] = pndvi["proj_lon"].round(7)
    pndvi["_plat_key"] = pndvi["proj_lat"].round(7)
    pndvi = pndvi.drop(columns=["proj_lon","proj_lat"],errors="ignore")
    out   = out.merge(pndvi, on=["_plon_key","_plat_key"], how="left")
    return out.drop(columns=["_plon_key","_plat_key"],errors="ignore")


def compute_pair_slopes(df, ndvi_year_cols, year_list, min_valid):
    ref_cols  = [f"ref_{c}"  for c in ndvi_year_cols]
    proj_cols = [f"proj_{c}" for c in ndvi_year_cols]
    missing   = [c for c in ref_cols+proj_cols if c not in df.columns]
    if missing: raise ValueError(f"Missing NDVI columns: {missing}")

    refY  = df[ref_cols].to_numpy(dtype=np.float64)
    projY = df[proj_cols].to_numpy(dtype=np.float64)

    ref_slope, ref_p, ref_n   = _ols_vectorized(refY,  year_list, min_valid)
    proj_slope, proj_p, proj_n = _ols_vectorized(projY, year_list, min_valid)

    out = df.copy()
    out["ref_slope"]          = ref_slope
    out["ref_pvalue_slope0"]  = ref_p
    out["ref_valid_n"]        = ref_n
    out["proj_slope"]         = proj_slope
    out["proj_pvalue_slope0"] = proj_p
    out["proj_valid_n"]       = proj_n
    out["slope_diff"]         = np.abs(proj_slope - ref_slope)
    out["slope_diff_signed"]  = proj_slope - ref_slope
    out["pair_slope_ok"]      = out["slope_diff"].le(PAIR_SLOPE_DIFF_MAX)
    out["ols_valid"]          = pd.notna(out["proj_slope"]) & pd.notna(out["ref_slope"])
    return out


# ================================================================
# SELEZIONE COPPIE — long_df costruito una sola volta (PERF 3)
# ================================================================

def select_pairs_with_parallel_trend(result_df, ndvi_year_cols):
    """
    Parte dalla soglia fissa di differenza di pendenza delle coppie; se il test
    aggregato fallisce, riduce iterativamente le coppie ordinate per |slope_diff|.

    PERF 3: valid_long_df costruito una sola volta; np.isin usato per filtrare nel loop.
    """
    min_n = max(MIN_SELECTED_PAIRS, int(len(result_df)*MIN_SELECTED_FRACTION))
    base  = result_df[result_df["ols_valid"] & result_df["pair_slope_ok"]].copy()
    if len(base) < min_n:
        base = result_df[result_df["ols_valid"]].copy()

    def _eval(sub_df, sub_long):
        it = interaction_test_from_long(sub_long)
        pt = paired_slope_test(sub_df)
        return it, pt, bool(it.get("passed",False) and pt.get("passed",False))

    base_long = make_long_for_interaction(base, ndvi_year_cols)
    it, pt, passed = _eval(base, base_long)
    if passed:
        base["twin_pass"] = True
        return base, {"selection_mode":"fixed_pair_slope_threshold",
                      "selected_fraction":float(len(base)/max(1,len(result_df))),
                      "interaction":it,"paired_slope":pt}

    # Ordina i validi una volta; costruisci long_df una volta — PERF 3
    valid               = result_df[result_df["ols_valid"]].copy().sort_values("slope_diff")
    valid_long_full     = make_long_for_interaction(valid, ndvi_year_cols)  # costruito una sola volta
    valid_pair_ids      = valid.index.to_numpy()
    valid_long_pair_arr = valid_long_full["pair_id"].to_numpy()

    best = None
    # Frazioni esplorate: generate da config (niente più pavimento hard-coded al 30%).
    search_fracs = [round(float(f), 4) for f in
                    np.arange(SELECTION_FRAC_MAX, SELECTION_FRAC_MIN - 1e-9, -SELECTION_FRAC_STEP)]
    for frac in search_fracs:
        n = max(min_n, int(len(valid)*frac))
        if n > len(valid): continue
        sub_ids  = valid_pair_ids[:n]
        mask_long = np.isin(valid_long_pair_arr, sub_ids)   # filtro veloce
        sub_long  = valid_long_full.iloc[mask_long]
        sub       = valid.iloc[:n]
        it, pt, passed = _eval(sub, sub_long)
        score = (0 if passed else 1,
                 abs(pt.get("mean_slope_diff",np.inf)),
                 -(it.get("interaction_pvalue",0) or 0))
        if best is None or score < best[0]:
            best = (score, sub, it, pt, frac, passed)
        if passed:
            sub = sub.copy(); sub["twin_pass"] = True
            return sub, {"selection_mode":"iterative_smallest_slope_diff",
                         "selected_fraction":float(frac),"interaction":it,"paired_slope":pt}

    if best is None:
        out = base.copy(); out["twin_pass"] = False
        return out, {"selection_mode":"failed_no_valid_candidate",
                     "selected_fraction":float(len(out)/max(1,len(result_df))),
                     "interaction":it,"paired_slope":pt}

    _, sub, it, pt, frac, passed = best
    sub = sub.copy(); sub["twin_pass"] = bool(passed)
    return sub, {"selection_mode":"best_available_failed_or_marginal",
                 "selected_fraction":float(frac),"interaction":it,"paired_slope":pt}


# ================================================================
# GRAFICI
# ================================================================

def plot_parallel_trends(result_df, passed_df, ndvi_year_cols, out_dir=None):
    years = years_from_cols(ndvi_year_cols)
    rows = []
    for label, df in [("all_matched",result_df),("selected",passed_df)]:
        for c, y in zip(ndvi_year_cols, years):
            pc = f"proj_{c}"; rc = f"ref_{c}"
            rows.append({"set":label,"group":"Project","year":y,
                          "mean_ndvi":df[pc].astype(float).mean(),
                          "se":df[pc].astype(float).std()/np.sqrt(max(1,df[pc].notna().sum()))})
            rows.append({"set":label,"group":"Reference","year":y,
                          "mean_ndvi":df[rc].astype(float).mean(),
                          "se":df[rc].astype(float).std()/np.sqrt(max(1,df[rc].notna().sum()))})
    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9,5))
    for label, ls, alpha in [("all_matched","--",0.45),("selected","-",0.95)]:
        for group, marker in [("Project","o"),("Reference","s")]:
            d = summary[(summary["set"]==label)&(summary["group"]==group)]
            ax.errorbar(d["year"],d["mean_ndvi"],yerr=d["se"],marker=marker,
                        linestyle=ls,alpha=alpha,label=f"{group} - {label}")
    ax.set_xlabel("Year"); ax.set_ylabel("Mean NDVI")
    ax.set_title("Parallel trend: mean NDVI Project vs Reference")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir)/"parallel_trends_mean_ndvi.png",dpi=150,bbox_inches="tight")
        summary.to_csv(Path(out_dir)/"parallel_trends_mean_ndvi_table.csv",index=False)
    return fig, summary


def plot_paired_slope_scatter(result_df, passed_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(6,6))
    failed = result_df.loc[(~result_df.get("twin_pass",pd.Series(False,index=result_df.index)).astype(bool))
                            & result_df["ols_valid"]]
    ax.scatter(failed["proj_slope"],failed["ref_slope"],s=8,alpha=0.25,label=f"Excluded ({len(failed):,})")
    ax.scatter(passed_df["proj_slope"],passed_df["ref_slope"],s=10,alpha=0.65,label=f"Selected — passed parallel test ({len(passed_df):,})")
    vals = pd.concat([result_df["proj_slope"],result_df["ref_slope"]]).dropna()
    lim  = max(float(vals.abs().quantile(0.99)*1.1) if len(vals) else 0, PAIR_SLOPE_DIFF_MAX*2)
    ax.plot([-lim,lim],[-lim,lim],"k--",lw=0.8,alpha=0.5,label="1:1 parallel")
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_xlabel("Project NDVI slope"); ax.set_ylabel("Reference NDVI slope")
    ax.set_title("Paired slope comparison"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    if out_dir: fig.savefig(Path(out_dir)/"paired_slope_scatter.png",dpi=150,bbox_inches="tight")
    return fig


def plot_example_grid(passed_df, ndvi_year_cols, out_dir=None, n=None):
    n = min(n or GRID_N_EXAMPLES, len(passed_df))
    if n == 0: return None
    rng    = np.random.default_rng(GRID_SEED)
    sample = passed_df.iloc[rng.choice(len(passed_df),size=n,replace=False)].reset_index(drop=True)
    years  = years_from_cols(ndvi_year_cols)
    side   = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side,side,figsize=(side*3.2,side*2.6))
    axes = np.array(axes).reshape(-1)
    for i, row in sample.iterrows():
        ax = axes[i]
        p = [row.get(f"proj_{c}",np.nan) for c in ndvi_year_cols]
        r = [row.get(f"ref_{c}", np.nan) for c in ndvi_year_cols]
        ax.plot(years,p,"o-",lw=1,markersize=4,label="Project")
        ax.plot(years,r,"s--",lw=1,markersize=4,label="Reference")
        ax.set_title(f"|Δslope|={row.get('slope_diff',np.nan):.4f}",fontsize=7)
        ax.tick_params(labelsize=6); ax.grid(alpha=0.2)
        if i==0: ax.legend(fontsize=6)
    for j in range(len(sample),len(axes)): axes[j].axis("off")
    fig.suptitle("Parallel-selected pairs: Project vs Reference NDVI", fontsize=10)
    plt.tight_layout(rect=[0,0,1,0.97])
    if out_dir: fig.savefig(Path(out_dir)/"twin_test_examples_grid.png",dpi=150,bbox_inches="tight")
    return fig


# ================================================================
# MAIN
# ================================================================

def run_twin_test(base_dirs=None, output_dir=None, matched_df=None,
                   proj_df=None, meta=None, verbose=True):
    if base_dirs is None: raise RuntimeError("base_dirs not provided.")
    base_dirs = [Path(b) for b in base_dirs]
    out_dir   = Path(output_dir) if output_dir else base_dirs[0].parent / STEP_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta is None:
        for cand in [base_dirs[0]/"matching_summary.json",
                     base_dirs[0].parent/"01_extract"/"extraction_report.json"]:
            if cand.exists(): meta = json.load(open(cand,encoding="utf-8")); break
    meta = meta or {}

    if matched_df is None:
        matched_df = load_df(find_file(base_dirs,"reference_area_pixels"))
    if proj_df is None:
        for cand in ["project_pixels_raw.parquet","project_pixels_raw.csv"]:
            p = base_dirs[0].parent/"01_extract"/cand
            if p.exists(): proj_df = load_df(p); break

    ndvi_year_cols = meta.get("ndvi_year_cols") or detect_ndvi_year_cols(proj_df.columns)
    year_list      = meta.get("year_list")      or years_from_cols(ndvi_year_cols)
    min_valid      = max(3, int(np.ceil(len(year_list)*0.60)))

    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 03 | p={PVALUE_THRESHOLD} | slope_tol={PAIR_SLOPE_DIFF_MAX} | "
              f"min_valid={min_valid}/{len(year_list)}")
        print(f"Output: {out_dir}\n{'='*60}")

    print("\n[1] Attach project NDVI")
    result = attach_project_ndvi(matched_df, proj_df, ndvi_year_cols)

    print("[2] Pair slopes (vectorized OLS)")
    t0 = time.time()
    result = compute_pair_slopes(result, ndvi_year_cols, year_list, min_valid)
    print(f"    Valid OLS: {int(result['ols_valid'].sum()):,}/{len(result):,} ({time.time()-t0:.1f}s)")

    print("[3] Aggregate tests")
    all_valid       = result[result["ols_valid"]]
    all_long        = make_long_for_interaction(all_valid, ndvi_year_cols)
    all_interaction = interaction_test_from_long(all_long)
    all_paired      = paired_slope_test(all_valid)
    print(f"    All interaction p={all_interaction.get('interaction_pvalue')} "
          f"slope_p={all_paired.get('pvalue')}")

    passed, selection_report = select_pairs_with_parallel_trend(result, ndvi_year_cols)
    sel_it = selection_report["interaction"]
    sel_pt = selection_report["paired_slope"]
    aggregate_passed = bool(sel_it.get("passed",False) and sel_pt.get("passed",False))

    result["twin_pass"] = result.index.isin(passed.index)
    passed = result[result["twin_pass"]].copy().reset_index(drop=True)

    # B4 fix: flag esplicito di compliance. Se il test aggregato NON passa,
    # i pixel "selezionati" sono solo il best-available e NON dovrebbero
    # alimentare il baseline senza decisione esplicita del PM.
    selection_mode = selection_report["selection_mode"]
    twin_test_compliant = bool(aggregate_passed) and selection_mode in (
        "fixed_pair_slope_threshold", "iterative_smallest_slope_diff"
    )
    proceed_recommended = twin_test_compliant

    print(f"    Mode={selection_mode} | "
          f"selected={len(passed):,}/{len(result):,} | passed={aggregate_passed}")
    if not twin_test_compliant:
        print("    " + "!" * 56)
        print("    WARNING: the parallel-trend test did NOT pass.")
        print("    The selected pixels are 'best available' and non-compliant.")
        print("    Step 04 will stop: only compliant pixels must be used.")
        print("    Review Step 02/03 and document in the PDD.")
        print("    " + "!" * 56)

    print("[4] Plots + export")
    fig_trend, trend_table = plot_parallel_trends(result, passed, ndvi_year_cols, out_dir)
    fig_slope = plot_paired_slope_scatter(result, passed, out_dir)
    fig_grid  = plot_example_grid(passed, ndvi_year_cols, out_dir)

    result.to_parquet(out_dir/"twin_test_all_pairs.parquet",  index=False)
    result.to_csv(    out_dir/"twin_test_all_pairs.csv",      index=False)
    passed.to_parquet(out_dir/"twin_tested_pixels.parquet",   index=False)
    passed.to_csv(    out_dir/"twin_tested_pixels.csv",       index=False)

    report = {
        "run_id":meta.get("run_id",""),
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "method":"Aggregate interaction test on NDVI parallel trend + paired slope test",
        "pvalue_threshold":PVALUE_THRESHOLD,"pair_slope_diff_max":PAIR_SLOPE_DIFF_MAX,
        "ndvi_year_cols":ndvi_year_cols,"year_list":year_list,"min_valid_years":int(min_valid),
        "input_pairs":int(len(result)),"ols_valid_pairs":int(result["ols_valid"].sum()),
        "selected_pairs":int(len(passed)),
        "selected_fraction":float(len(passed)/max(1,len(result))),
        "all_matched":{"interaction":all_interaction,"paired_slope":all_paired},
        "selected":{"interaction":sel_it,"paired_slope":sel_pt},
        "selection_mode":selection_report["selection_mode"],
        "aggregate_twin_passed":aggregate_passed,
        "twin_test_compliant":twin_test_compliant,
        "proceed_recommended":proceed_recommended,
        "compliance_note":(
            "If twin_test_compliant=False the selected pixels are 'best available' "
            "and did not pass the aggregate parallel-trend test. Step 04 rejects them: "
            "only compliant pixels must be used, no override is allowed."
        ),
        "next_step":"04_STARR_reference_area_lock.py",
    }
    with open(out_dir/"twin_test_report.json","w",encoding="utf-8") as f:
        json.dump(report,f,indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Parallel-selected pairs : {len(passed):,}/{len(result):,}")
        print(f"Aggregate OK            : {aggregate_passed}")
        print(f"Output                  : {out_dir}\n{'='*60}")

    return result, passed, report, {"parallel_trends":fig_trend,"slope_scatter":fig_slope,
                                     "examples":fig_grid}, out_dir


def main():
    run_twin_test()

if __name__ == "__main__":
    main()