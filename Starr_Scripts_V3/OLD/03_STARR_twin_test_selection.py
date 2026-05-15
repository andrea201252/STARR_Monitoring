# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB | 03_STARR_twin_test_selection.py
STEP 03 — Aggregate Parallel Trend / Twin Test
============================================================

Main changes vs previous version:
  - Does not test project and reference trends separately.
  - Tests whether the two trajectories diverge using an interaction model:
        NDVI ~ group + year + group:year
    The p-value of group:year must be > 0.05.
  - Adds a paired slope-difference test.
  - Keeps real annual project NDVI values, not reconstructed trend lines.
  - Uses a fixed pair-level |slope_project - slope_reference| tolerance only as
    a conservative filtering aid; the aggregate interaction test is the key test.

Output:
  twin_test_all_pairs.parquet/csv
  twin_tested_pixels.parquet/csv
  twin_test_report.json
  parallel_trends_mean_ndvi.png
  paired_slope_scatter.png
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
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except ImportError:
        return False


if not _is_notebook():
    matplotlib.use("Agg")


# ================================================================
# FIXED / USER PARAMETERS
# ================================================================

PVALUE_THRESHOLD = 0.05
PAIR_SLOPE_DIFF_MAX = 0.005  # NDVI units / year; fixed ex-ante, not derived from data
MIN_SELECTED_FRACTION = 0.30
MIN_SELECTED_PAIRS = 30
GRID_N_EXAMPLES = 16
GRID_SEED = 42


# ================================================================
# IO HELPERS
# ================================================================

def load_df(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File non trovato: {p}")
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def find_file(base_dirs, stem):
    for base in base_dirs:
        base = Path(base)
        for ext in ["parquet", "csv"]:
            p = base / f"{stem}.{ext}"
            if p.exists():
                print(f"    Trovato: {p.name}")
                return p
    raise FileNotFoundError(f"'{stem}' non trovato in {base_dirs}")


def detect_ndvi_year_cols(columns):
    import re
    return sorted([c for c in columns if re.match(r"^NDVI_\d{4}$", str(c))])


def years_from_cols(ndvi_year_cols):
    return [int(str(c).split("_")[1]) for c in ndvi_year_cols]


# ================================================================
# OLS / TESTS
# ================================================================

def _ols_vectorized(Y, years, min_valid):
    years = np.asarray(years, dtype=float)
    n_yrs = len(years)
    n_px = Y.shape[0]
    slopes = np.full(n_px, np.nan)
    pvalues = np.full(n_px, np.nan)
    valid_cnt = np.isfinite(Y).sum(axis=1)

    for i in np.where(valid_cnt >= min_valid)[0]:
        y = Y[i].astype(float)
        ok = np.isfinite(y)
        xi = years[ok]
        yi = y[ok]
        ni = len(yi)
        if ni < 3:
            continue
        xci = xi - xi.mean()
        ssx = float((xci ** 2).sum())
        if ssx <= 1e-12:
            continue
        sl = float((xci * (yi - yi.mean())).sum() / ssx)
        ic = float(yi.mean() - sl * xi.mean())
        yh = ic + sl * xi
        sse = float(((yi - yh) ** 2).sum())
        if ni <= 2:
            continue
        s2 = sse / (ni - 2)
        se = np.sqrt(max(s2 / ssx, 1e-30))
        pvalues[i] = 2.0 * float(stats.t.sf(abs(sl / se), df=ni - 2))
        slopes[i] = sl
    return slopes, pvalues, valid_cnt.astype(int)


def interaction_test_from_long(long_df):
    """OLS y = b0 + b1*group + b2*year_center + b3*group*year_center."""
    df = long_df[["ndvi", "group", "year"]].dropna().copy()
    if len(df) < 8 or df["group"].nunique() < 2 or df["year"].nunique() < 3:
        return {"n_obs": int(len(df)), "interaction_beta": np.nan, "interaction_pvalue": np.nan, "passed": False}

    y = df["ndvi"].astype(float).values
    g = df["group"].astype(float).values
    yr = df["year"].astype(float).values
    yc = yr - yr.mean()
    X = np.column_stack([np.ones(len(df)), g, yc, g * yc])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        if dof <= 0:
            raise ValueError("insufficient dof")
        sigma2 = float((resid @ resid) / dof)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
        t = float(beta[3] / se[3])
        p = 2.0 * float(stats.t.sf(abs(t), df=dof))
        return {
            "n_obs": int(len(df)),
            "interaction_beta": float(beta[3]),
            "interaction_t": t,
            "interaction_pvalue": p,
            "passed": bool(p > PVALUE_THRESHOLD),
        }
    except Exception:
        return {"n_obs": int(len(df)), "interaction_beta": np.nan, "interaction_pvalue": np.nan, "passed": False}


def paired_slope_test(df):
    d = (df["proj_slope"] - df["ref_slope"]).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(d) < 3:
        return {"n_pairs": int(len(d)), "mean_slope_diff": np.nan, "pvalue": np.nan, "passed": False}
    t, p = stats.ttest_1samp(d.values, popmean=0.0, nan_policy="omit")
    return {
        "n_pairs": int(len(d)),
        "mean_slope_diff": float(d.mean()),
        "median_abs_slope_diff": float(np.median(np.abs(d.values))),
        "pvalue": float(p),
        "passed": bool(p > PVALUE_THRESHOLD and abs(float(d.mean())) <= PAIR_SLOPE_DIFF_MAX),
    }


def make_long_for_interaction(df, ndvi_year_cols):
    rows = []
    for c in ndvi_year_cols:
        year = int(c.split("_")[1])
        pc = f"proj_{c}"
        rc = f"ref_{c}"
        if pc in df.columns:
            tmp = pd.DataFrame({"pair_id": df.index.values, "year": year, "group": 1, "ndvi": df[pc].values})
            rows.append(tmp)
        if rc in df.columns:
            tmp = pd.DataFrame({"pair_id": df.index.values, "year": year, "group": 0, "ndvi": df[rc].values})
            rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["pair_id", "year", "group", "ndvi"])
    return pd.concat(rows, ignore_index=True)


# ================================================================
# DATA PREP
# ================================================================

def attach_project_ndvi(matched_df, proj_df, ndvi_year_cols):
    """Adds proj_NDVI_YYYY and ref_NDVI_YYYY columns to matched_df."""
    out = matched_df.copy()
    # In matched_df, donor/reference annual bands are currently named NDVI_YYYY.
    for c in ndvi_year_cols:
        if c in out.columns and f"ref_{c}" not in out.columns:
            out = out.rename(columns={c: f"ref_{c}"})

    join_cols = ["lon", "lat"] + [c for c in ndvi_year_cols if c in proj_df.columns]
    pndvi = proj_df[join_cols].rename(columns={"lon": "proj_lon", "lat": "proj_lat"})
    pndvi = pndvi.rename(columns={c: f"proj_{c}" for c in ndvi_year_cols if c in pndvi.columns})

    # Avoid floating precision issues by rounded coordinate keys.
    out["_plon_key"] = out["proj_lon"].round(7)
    out["_plat_key"] = out["proj_lat"].round(7)
    pndvi["_plon_key"] = pndvi["proj_lon"].round(7)
    pndvi["_plat_key"] = pndvi["proj_lat"].round(7)
    pndvi = pndvi.drop(columns=["proj_lon", "proj_lat"], errors="ignore")
    out = out.merge(pndvi, on=["_plon_key", "_plat_key"], how="left")
    out = out.drop(columns=["_plon_key", "_plat_key"], errors="ignore")
    return out


def compute_pair_slopes(df, ndvi_year_cols, year_list, min_valid):
    ref_cols = [f"ref_{c}" for c in ndvi_year_cols]
    proj_cols = [f"proj_{c}" for c in ndvi_year_cols]
    missing = [c for c in ref_cols + proj_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne NDVI mancanti dopo join: {missing}")
    refY = df[ref_cols].values.astype(float)
    projY = df[proj_cols].values.astype(float)
    ref_slope, ref_p, ref_n = _ols_vectorized(refY, year_list, min_valid)
    proj_slope, proj_p, proj_n = _ols_vectorized(projY, year_list, min_valid)
    out = df.copy()
    out["ref_slope"] = ref_slope
    out["ref_pvalue_slope0"] = ref_p
    out["ref_valid_n"] = ref_n
    out["proj_slope"] = proj_slope
    out["proj_pvalue_slope0"] = proj_p
    out["proj_valid_n"] = proj_n
    out["slope_diff"] = np.abs(out["proj_slope"] - out["ref_slope"])
    out["slope_diff_signed"] = out["proj_slope"] - out["ref_slope"]
    out["pair_slope_ok"] = out["slope_diff"].le(PAIR_SLOPE_DIFF_MAX)
    out["ols_valid"] = out["proj_slope"].notna() & out["ref_slope"].notna()
    return out


def select_pairs_with_parallel_trend(result_df, ndvi_year_cols):
    """
    Starts from fixed pair slope tolerance. If aggregate trend still fails, iteratively
    keeps the pairs with smallest |slope diff| until the aggregate test passes.
    """
    base = result_df[result_df["ols_valid"] & result_df["pair_slope_ok"]].copy()
    if len(base) < max(MIN_SELECTED_PAIRS, int(len(result_df) * MIN_SELECTED_FRACTION)):
        base = result_df[result_df["ols_valid"]].copy()

    def evaluate(sub):
        long_df = make_long_for_interaction(sub, ndvi_year_cols)
        it = interaction_test_from_long(long_df)
        pt = paired_slope_test(sub)
        passed = bool(it.get("passed", False) and pt.get("passed", False))
        return it, pt, passed

    it, pt, passed = evaluate(base)
    if passed:
        base["twin_pass"] = True
        return base, {"selection_mode": "fixed_pair_slope_threshold", "selected_fraction": float(len(base) / len(result_df)), "interaction": it, "paired_slope": pt}

    valid = result_df[result_df["ols_valid"]].copy().sort_values("slope_diff")
    min_n = max(MIN_SELECTED_PAIRS, int(len(result_df) * MIN_SELECTED_FRACTION))
    best = None
    for frac in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]:
        n = max(min_n, int(len(valid) * frac))
        if n > len(valid):
            continue
        sub = valid.iloc[:n].copy()
        it, pt, passed = evaluate(sub)
        score = (0 if passed else 1, abs(pt.get("mean_slope_diff", np.inf)), it.get("interaction_pvalue", -np.inf) * -1)
        if best is None or score < best[0]:
            best = (score, sub, it, pt, frac, passed)
        if passed:
            sub["twin_pass"] = True
            return sub, {"selection_mode": "iterative_smallest_slope_diff", "selected_fraction": float(frac), "interaction": it, "paired_slope": pt}

    if best is None:
        out = base.copy()
        out["twin_pass"] = False
        return out, {"selection_mode": "failed_no_valid_candidate", "selected_fraction": float(len(out) / len(result_df)) if len(result_df) else 0, "interaction": it, "paired_slope": pt}

    _, sub, it, pt, frac, passed = best
    sub = sub.copy()
    sub["twin_pass"] = bool(passed)
    return sub, {"selection_mode": "best_available_failed_or_marginal", "selected_fraction": float(frac), "interaction": it, "paired_slope": pt}


# ================================================================
# PLOTS
# ================================================================

def plot_parallel_trends(result_df, passed_df, ndvi_year_cols, out_dir=None):
    years = years_from_cols(ndvi_year_cols)
    rows = []
    for label, df in [("all_matched", result_df), ("selected", passed_df)]:
        for c, y in zip(ndvi_year_cols, years):
            pc = f"proj_{c}"; rc = f"ref_{c}"
            rows.append({"set": label, "group": "Project", "year": y, "mean_ndvi": df[pc].astype(float).mean(), "se": df[pc].astype(float).std() / np.sqrt(df[pc].notna().sum())})
            rows.append({"set": label, "group": "Reference", "year": y, "mean_ndvi": df[rc].astype(float).mean(), "se": df[rc].astype(float).std() / np.sqrt(df[rc].notna().sum())})
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, ls, alpha in [("all_matched", "--", 0.45), ("selected", "-", 0.95)]:
        for group, marker in [("Project", "o"), ("Reference", "s")]:
            d = summary[(summary["set"] == label) & (summary["group"] == group)]
            ax.errorbar(d["year"], d["mean_ndvi"], yerr=d["se"], marker=marker, linestyle=ls, alpha=alpha, label=f"{group} - {label}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean NDVI")
    ax.set_title("Parallel trend test: Project vs Reference mean NDVI")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "parallel_trends_mean_ndvi.png", dpi=150, bbox_inches="tight")
        summary.to_csv(Path(out_dir) / "parallel_trends_mean_ndvi_table.csv", index=False)
    return fig, summary


def plot_paired_slope_scatter(result_df, passed_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    failed = result_df.loc[(~result_df.get("twin_pass", False).astype(bool)) & result_df["ols_valid"]]
    ax.scatter(failed["proj_slope"], failed["ref_slope"], s=8, alpha=0.25, label=f"Excluded ({len(failed):,})")
    ax.scatter(passed_df["proj_slope"], passed_df["ref_slope"], s=10, alpha=0.65, label=f"Selected ({len(passed_df):,})")
    vals = pd.concat([result_df["proj_slope"], result_df["ref_slope"]]).dropna()
    if len(vals):
        lim = float(vals.abs().quantile(0.99) * 1.1)
        lim = max(lim, PAIR_SLOPE_DIFF_MAX * 2)
    else:
        lim = PAIR_SLOPE_DIFF_MAX * 2
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5, label="1:1")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Project NDVI slope")
    ax.set_ylabel("Reference NDVI slope")
    ax.set_title("Paired slope comparison")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "paired_slope_scatter.png", dpi=150, bbox_inches="tight")
    return fig


def plot_example_grid(passed_df, ndvi_year_cols, out_dir=None, n=None):
    n = min(n or GRID_N_EXAMPLES, len(passed_df))
    if n == 0:
        return None
    rng = np.random.default_rng(GRID_SEED)
    sample = passed_df.iloc[rng.choice(len(passed_df), size=n, replace=False)].reset_index(drop=True)
    years = years_from_cols(ndvi_year_cols)
    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(side * 3.2, side * 2.6))
    axes = np.array(axes).reshape(-1)
    for i, row in sample.iterrows():
        ax = axes[i]
        p = [row.get(f"proj_{c}", np.nan) for c in ndvi_year_cols]
        r = [row.get(f"ref_{c}", np.nan) for c in ndvi_year_cols]
        ax.plot(years, p, "o-", lw=1, markersize=4, label="Project")
        ax.plot(years, r, "s--", lw=1, markersize=4, label="Reference")
        ax.set_title(f"|Δslope|={row.get('slope_diff', np.nan):.4f}", fontsize=7)
        ax.tick_params(labelsize=6); ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(len(sample), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "twin_test_examples_grid.png", dpi=150, bbox_inches="tight")
    return fig


# ================================================================
# MAIN STEP
# ================================================================

def run_twin_test(base_dirs=None, output_dir=None, matched_df=None, proj_df=None, meta=None, verbose=True):
    """Returns: result_df, passed_df, selection_report, figs, out_dir"""
    if base_dirs is None:
        candidate = Path("/content/content/MyDrive/STARR_Idiofa_New/STARR_outputs/Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v04_raster/02_matching")
        base_dirs = [candidate] if candidate.exists() else []
        if not base_dirs:
            raise RuntimeError("base_dirs non fornito e directory default non trovata.")
    base_dirs = [Path(b) for b in base_dirs]
    out_dir = Path(output_dir) if output_dir else base_dirs[0].parent / "03_twin_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta is None:
        for cand in [base_dirs[0] / "matching_summary.json", base_dirs[0].parent / "01_extract" / "extraction_report.json"]:
            if cand.exists():
                meta = json.load(open(cand, encoding="utf-8"))
                break
    meta = meta or {}

    if matched_df is None:
        matched_df = load_df(find_file(base_dirs, "reference_area_pixels"))
    if proj_df is None:
        project_file = base_dirs[0].parent / "01_extract" / "project_pixels_raw.parquet"
        if not project_file.exists():
            project_file = base_dirs[0].parent / "01_extract" / "project_pixels_raw.csv"
        proj_df = load_df(project_file)

    ndvi_year_cols = meta.get("ndvi_year_cols") or detect_ndvi_year_cols(proj_df.columns)
    year_list = meta.get("year_list") or years_from_cols(ndvi_year_cols)
    min_valid = max(3, int(np.ceil(len(year_list) * 0.60)))

    if verbose:
        print(f"\n{'=' * 60}")
        print("STEP 03 - Aggregate Parallel Trend Twin Test")
        print(f"p threshold          : > {PVALUE_THRESHOLD}")
        print(f"pair slope tolerance : <= {PAIR_SLOPE_DIFF_MAX} NDVI/year")
        print(f"min valid years      : {min_valid}/{len(year_list)}")
        print(f"Output               : {out_dir}")
        print(f"{'=' * 60}")

    print("\n[1] Attach real project NDVI series")
    result = attach_project_ndvi(matched_df, proj_df, ndvi_year_cols)

    print("[2] Compute pair slopes")
    t0 = time.time()
    result = compute_pair_slopes(result, ndvi_year_cols, year_list, min_valid)
    print(f"    OLS valid pairs: {int(result['ols_valid'].sum()):,}/{len(result):,} ({time.time() - t0:.1f}s)")

    print("[3] Aggregate interaction and paired slope tests")
    all_long = make_long_for_interaction(result[result["ols_valid"]], ndvi_year_cols)
    all_interaction = interaction_test_from_long(all_long)
    all_paired = paired_slope_test(result[result["ols_valid"]])
    print(f"    All matched interaction p : {all_interaction.get('interaction_pvalue')}")
    print(f"    All matched slope diff p  : {all_paired.get('pvalue')}")

    passed, selection_report = select_pairs_with_parallel_trend(result, ndvi_year_cols)
    selected_interaction = selection_report["interaction"]
    selected_paired = selection_report["paired_slope"]
    aggregate_passed = bool(selected_interaction.get("passed", False) and selected_paired.get("passed", False))

    # Mark selected rows in full result.
    result["twin_pass"] = result.index.isin(passed.index)
    passed = result[result["twin_pass"]].copy().reset_index(drop=True)

    print(f"    Selection mode          : {selection_report['selection_mode']}")
    print(f"    Selected pairs          : {len(passed):,}/{len(result):,}")
    print(f"    Selected interaction p  : {selected_interaction.get('interaction_pvalue')}")
    print(f"    Selected slope diff p   : {selected_paired.get('pvalue')}")
    print(f"    Aggregate twin passed   : {aggregate_passed}")

    print("[4] Plots + export")
    fig_trend, trend_table = plot_parallel_trends(result, passed, ndvi_year_cols, out_dir)
    fig_slope = plot_paired_slope_scatter(result, passed, out_dir)
    fig_grid = plot_example_grid(passed, ndvi_year_cols, out_dir)

    result.to_parquet(out_dir / "twin_test_all_pairs.parquet", index=False)
    result.to_csv(out_dir / "twin_test_all_pairs.csv", index=False)
    passed.to_parquet(out_dir / "twin_tested_pixels.parquet", index=False)
    passed.to_csv(out_dir / "twin_tested_pixels.csv", index=False)

    report = {
        "run_id": meta.get("run_id", ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Aggregate NDVI parallel trend interaction test plus paired slope test",
        "pvalue_threshold": PVALUE_THRESHOLD,
        "pair_slope_diff_max": PAIR_SLOPE_DIFF_MAX,
        "ndvi_year_cols": ndvi_year_cols,
        "year_list": year_list,
        "min_valid_years": int(min_valid),
        "input_pairs": int(len(result)),
        "ols_valid_pairs": int(result["ols_valid"].sum()),
        "selected_pairs": int(len(passed)),
        "selected_fraction": float(len(passed) / len(result)) if len(result) else 0,
        "all_matched": {"interaction": all_interaction, "paired_slope": all_paired},
        "selected": {"interaction": selected_interaction, "paired_slope": selected_paired},
        "selection_mode": selection_report["selection_mode"],
        "aggregate_twin_passed": aggregate_passed,
        "next_step": "04_STARR_reference_area_lock.py",
    }
    with open(out_dir / "twin_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    figs = {"parallel_trends": fig_trend, "slope_scatter": fig_slope, "examples": fig_grid}
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Twin selected : {len(passed):,}/{len(result):,}")
        print(f"Aggregate OK  : {aggregate_passed}")
        print(f"Output        : {out_dir}")
        print(f"{'=' * 60}")
    return result, passed, report, figs, out_dir


def main():
    run_twin_test()


if __name__ == "__main__":
    main()
