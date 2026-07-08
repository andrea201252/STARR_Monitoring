# -*- coding: utf-8 -*-
"""
GS STARR Track 1 SEMDB — 00b_STARR_compare_extents.py
Runner di confronto multi-extent.

Esegue la pipeline Steps 01-04 per ogni extent in COMPARE_EXTENTS,
aggrega le metriche dai JSON prodotti da ogni step e genera:
  - comparison_summary.json   → tutte le metriche per extent
  - comparison_plots.png      → 4 panel comparativi
  - recommendation.txt        → extent minimo sufficiente

Utilizzo
--------
  python 00b_STARR_compare_extents.py

Oppure da notebook:
  import importlib.util, pathlib
  m = importlib.util.spec_from_file_location(
        "s00b", pathlib.Path("00b_STARR_compare_extents.py"))
  s00b = importlib.util.module_from_spec(m); m.loader.exec_module(s00b)
  results = s00b.run_comparison()

Criteri di sufficienza (tutti e 4 devono essere soddisfatti)
─────────────────────────────────────────────────────────────
  n_donor / n_project ≥ 3     (linea guida STARR)
  match_coverage_pct  ≥ 90%   (Step 02)
  twin_pass_pct       ≥ 30%   (Step 03)
  smd_max             ≤ 0.10  (Step 02)
"""

import gc
import json
import importlib.util
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

# Estensioni da confrontare — "full" + km numerici
COMPARE_EXTENTS: list = ["full", 5, 10, 20, 30]

# Directory base dove trovare i TIF e dove scrivere i risultati
BASE_DIR_CANDIDATES = []


# Soglie di sufficienza
THRESH_RATIO_3X       = 3.0
THRESH_MATCH_COV_PCT  = 90.0
THRESH_TWIN_PASS_PCT  = 30.0
THRESH_SMD_MAX        = 0.10

# ══════════════════════════════════════════════════════════════════════


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and \
               shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except ImportError:
        return False


if not _is_notebook():
    matplotlib.use("Agg", force=True)


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── ESTRAZIONE METRICHE ───────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def extract_metrics(run_result: dict) -> dict:
    """
    Estrae metriche aggregate leggendo i JSON di ogni step.
    Usa i path in run_result["summary"] come punto di partenza.
    """
    s = run_result.get("summary", {})

    # Step 01 — extraction_report.json
    out01  = Path(s.get("out01", ""))
    rep01  = _load_json(out01 / "extraction_report.json") if out01.exists() else {}

    # Step 02 — matching_summary.json
    out02  = Path(s.get("out02", ""))
    rep02  = _load_json(out02 / "matching_summary.json") if out02.exists() else {}

    # Step 03 — twin_test_report.json
    out03  = Path(s.get("out03", ""))
    rep03  = _load_json(out03 / "twin_test_report.json") if out03.exists() else {}

    # Step 04 — reference_area_FINAL_manifest.json
    out04  = Path(s.get("out04", ""))
    rep04  = _load_json(out04 / "reference_area_FINAL_manifest.json") if out04.exists() else {}

    n_project   = int(rep01.get("project_n", s.get("project_n", 0)))
    n_donor     = int(rep01.get("donor_n",   s.get("donor_n",   0)))
    n_matched   = int(rep02.get("matched_n", s.get("matched_n", 0)))
    n_input     = int(rep03.get("input_pairs", max(n_matched, 1)))
    n_twin      = int(rep03.get("selected_pairs", s.get("twin_pass_n", 0)))
    smd_max     = float(rep02.get("SMD_max", s.get("smd_max") or 9.99))

    # A2: usa il rapporto 3× basato sull'AREA (corretto) se disponibile,
    # con fallback al proxy sui conteggi.
    ratio_area  = rep01.get("ratio_donor_project_area")
    ratio_count = n_donor / max(n_project, 1)
    ratio       = float(ratio_area) if ratio_area is not None else ratio_count
    ratio_is_area = ratio_area is not None
    match_cov   = n_matched / max(n_project, 1) * 100.0
    twin_pct    = n_twin    / max(n_input,   1) * 100.0
    ref_ha      = float(rep04.get("reference_area_definition", {})
                        .get("total_ha", s.get("reference_area_ha", 0)))

    # Criteri di sufficienza
    ok_ratio  = (rep01.get("meets_3x_guideline_area")
                 if rep01.get("meets_3x_guideline_area") is not None
                 else ratio >= THRESH_RATIO_3X)
    ok_ratio  = bool(ok_ratio)
    ok_match  = match_cov >= THRESH_MATCH_COV_PCT
    ok_twin   = twin_pct  >= THRESH_TWIN_PASS_PCT
    ok_smd    = smd_max   <= THRESH_SMD_MAX
    # B4: il twin deve essere conforme, non solo "selezionato".
    twin_compliant = bool(rep03.get("twin_test_compliant",
                                    rep03.get("aggregate_twin_passed", False)))
    sufficient = ok_ratio and ok_match and ok_twin and ok_smd and twin_compliant

    return {
        "extent_km":           s.get("donor_extent_km", "?"),
        "run_id":              rep01.get("run_id", s.get("run_id", "")),
        "n_project":           n_project,
        "n_donor":             n_donor,
        "ratio_donor_project": round(ratio, 2),
        "ratio_is_area_based": ratio_is_area,
        "ratio_count_proxy":   round(ratio_count, 2),
        "n_matched":           n_matched,
        "match_coverage_pct":  round(match_cov, 2),
        "smd_max":             round(smd_max, 4),
        "smd_all_passed":      bool(rep02.get("SMD_all_passed", False)),
        "n_twin_passed":       n_twin,
        "twin_pass_pct":       round(twin_pct, 2),
        "twin_aggregate_ok":   bool(rep03.get("aggregate_twin_passed", False)),
        "twin_test_compliant": twin_compliant,
        "reference_area_ha":   round(ref_ha, 2),
        "criteria": {
            "ratio_3x":        bool(ok_ratio),
            "match_cov_90pct": bool(ok_match),
            "twin_pass_30pct": bool(ok_twin),
            "twin_compliant":  twin_compliant,
            "smd_le_010":      bool(ok_smd),
            "ALL_SUFFICIENT":  bool(sufficient),
        },
        "output_dirs": {
            "01": str(s.get("out01", "")),
            "02": str(s.get("out02", "")),
            "03": str(s.get("out03", "")),
            "04": str(s.get("out04", "")),
        },
    }


# ── VERIFICA SUFFICIENZA ──────────────────────────────────────────────

def find_minimum_sufficient_extent(metrics_list: list[dict]) -> dict | None:
    """
    Restituisce il primo extent (in ordine crescente) che soddisfa
    tutti e 4 i criteri. "full" viene trattato come +∞.
    """
    def sort_key(m):
        e = m["extent_km"]
        return float("inf") if e == "full" else float(e)

    sorted_m = sorted(metrics_list, key=sort_key)
    for m in sorted_m:
        if m["criteria"]["ALL_SUFFICIENT"]:
            return m
    return None


# ── GRAFICI ───────────────────────────────────────────────────────────

def _extent_label(e) -> str:
    return "FULL" if e == "full" else f"{e}km"


def _criterion_color(ok: bool) -> str:
    return "#2e7d32" if ok else "#c62828"


def plot_comparison(metrics_list: list[dict],
                    out_path: Path | None = None) -> plt.Figure:
    """
    4 panel comparativi:
      A — n_donor + soglia ratio 3×
      B — match_coverage_pct + soglia 90%
      C — twin_pass_pct + soglia 30%
      D — smd_max + soglia 0.10
    Più una riga inferiore con la heatmap criteri per extent.
    """
    labels   = [_extent_label(m["extent_km"]) for m in metrics_list]
    x        = np.arange(len(labels))
    bar_kw   = dict(width=0.55, edgecolor="white", zorder=3)

    fig = plt.figure(figsize=(max(16, len(labels) * 3.2), 14),
                     facecolor="white")
    gs  = GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.38,
                   height_ratios=[2.5, 1])

    # ── Panel A: n_donor ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    n_proj = metrics_list[0]["n_project"]
    vals   = [m["n_donor"] for m in metrics_list]
    colors = [_criterion_color(m["criteria"]["ratio_3x"]) for m in metrics_list]
    ax.bar(x, vals, color=colors, alpha=0.78, **bar_kw)
    ax.axhline(THRESH_RATIO_3X * n_proj, color="#f5a623", ls="--", lw=1.8,
               label=f"3× ({n_proj:,} px)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_title("A — Dimensione pool donor\n(verde = soddisfa 3×)",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("n pixel donor", fontsize=8)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.35, zorder=0)
    for xi, v, r in zip(x, vals, [m["ratio_donor_project"] for m in metrics_list]):
        ax.text(xi, v, f"{r:.1f}×", ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    # ── Panel B: match_coverage ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    vals   = [m["match_coverage_pct"] for m in metrics_list]
    colors = [_criterion_color(m["criteria"]["match_cov_90pct"])
              for m in metrics_list]
    ax.bar(x, vals, color=colors, alpha=0.78, **bar_kw)
    ax.axhline(THRESH_MATCH_COV_PCT, color="#f5a623", ls="--", lw=1.8,
               label=f"Soglia {THRESH_MATCH_COV_PCT:.0f}%")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title("B — Copertura match %\n(verde = ≥90%)",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("% px progetto matchati", fontsize=8)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.35, zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    # ── Panel C: twin_pass_pct ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    vals   = [m["twin_pass_pct"] for m in metrics_list]
    colors = [_criterion_color(m["criteria"]["twin_pass_30pct"])
              for m in metrics_list]
    ax.bar(x, vals, color=colors, alpha=0.78, **bar_kw)
    ax.axhline(THRESH_TWIN_PASS_PCT, color="#f5a623", ls="--", lw=1.8,
               label=f"Soglia {THRESH_TWIN_PASS_PCT:.0f}%")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title("C — Twin test superato %\n(verde = ≥30%)",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("% coppie con trend parallelo", fontsize=8)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.35, zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    # ── Panel D: smd_max ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 3])
    vals   = [m["smd_max"] for m in metrics_list]
    colors = [_criterion_color(m["criteria"]["smd_le_010"])
              for m in metrics_list]
    ax.bar(x, vals, color=colors, alpha=0.78, **bar_kw)
    ax.axhline(THRESH_SMD_MAX, color="#f5a623", ls="--", lw=1.8,
               label=f"Soglia {THRESH_SMD_MAX}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("D — SMD max\n(verde = ≤0.10)",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("SMD massimo covariata", fontsize=8)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.35, zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # ── Riga inferiore: heatmap criteri ───────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    criteria_keys = ["ratio_3x", "match_cov_90pct",
                     "twin_pass_30pct", "twin_compliant", "smd_le_010", "ALL_SUFFICIENT"]
    criteria_labels = [
        "Ratio ≥3×", "Match ≥90%", "Twin ≥30%",
        "Twin conf.", "SMD ≤0.10", "TUTTO OK ✓"
    ]
    n_ext  = len(metrics_list)
    n_crit = len(criteria_keys)
    heat   = np.zeros((n_crit, n_ext))
    for j, m in enumerate(metrics_list):
        for i, k in enumerate(criteria_keys):
            heat[i, j] = 1.0 if m["criteria"][k] else 0.0

    im = ax2.imshow(heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(n_ext)); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_yticks(range(n_crit)); ax2.set_yticklabels(criteria_labels, fontsize=9)
    for i in range(n_crit):
        for j in range(n_ext):
            ax2.text(j, i, "✓" if heat[i, j] else "✗",
                     ha="center", va="center", fontsize=12,
                     color="white" if heat[i, j] == 0 else "black",
                     fontweight="bold")
    ax2.set_title("Heatmap criteri di sufficienza per extent",
                  fontsize=9, fontweight="bold")

    # Evidenzia extent minimo sufficiente
    min_suff = find_minimum_sufficient_extent(metrics_list)
    if min_suff:
        min_label = _extent_label(min_suff["extent_km"])
        if min_label in labels:
            j_min = labels.index(min_label)
            for i in range(n_crit):
                ax2.add_patch(
                    plt.Rectangle((j_min - 0.5, i - 0.5), 1, 1,
                                  fill=False, edgecolor="#0d47a1", lw=3))

    fig.suptitle(
        f"GS STARR — Confronto Estensione Donor\n"
        f"Progetto: {metrics_list[0]['n_project']:,} px | "
        f"Soglie: ratio≥{THRESH_RATIO_3X}× | match≥{THRESH_MATCH_COV_PCT}% | "
        f"twin≥{THRESH_TWIN_PASS_PCT}% | SMD≤{THRESH_SMD_MAX}",
        fontsize=10, fontweight="bold")

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  [00b] Plot salvato: {out_path}")

    return fig


# ── MAIN ──────────────────────────────────────────────────────────────

def run_comparison(extents: list | None = None,
                   base_dirs: list | None = None,
                   output_dir: Path | str | None = None,
                   run_step_05: bool = False,
                   run_id_base: str | None = None,
                   fnf_shapefile: str | None = None,
                   eligible_shapefile: str | None = None,
                   use_eligibility: bool = True) -> dict:
    """
    Esegue la pipeline Steps 01-04 per ogni extent e aggrega i risultati.

    Parametri
    ---------
    extents : list | None
        Lista di estensioni. Default = COMPARE_EXTENTS.
    base_dirs : list | None
        Directory base. Default = BASE_DIR_CANDIDATES.
    output_dir : path | None
        Dove salvare comparison_summary.json e comparison_plots.png.
        Default: base_dir / STARR_outputs / comparison.
    run_step_05 : bool
        Se True esegue anche Step 05 (richiede dati carbon-stock).
    run_id_base : str | None
        Prefisso dei TIF GEE (es. "Idiofa_Lobi_2018_buf50km_...").
        None / "" = wildcard (trova tutti i TIF nella cartella).
    fnf_shapefile : str | None
        Percorso shapefile FNF18. None = filtro disattivato.
    eligible_shapefile : str | None
        Percorso shapefile Eligible_FNF. None = filtro disattivato.
    use_eligibility : bool
        Se True (default) applica il filtro Eligible_FNF al donor
        (conservativo). Se False il donor usa l'intera area non-forest.

    Restituisce
    -----------
    dict {
        "metrics": [dict per ogni extent],
        "minimum_sufficient": dict | None,
        "recommendation": str,
        "output_dir": str,
    }
    """
    extents   = extents   or COMPARE_EXTENTS
    base_dirs = base_dirs or [Path(p) for p in BASE_DIR_CANDIDATES
                               if Path(p).exists()]

    if not base_dirs:
        raise FileNotFoundError(
            f"Nessuna directory trovata: {BASE_DIR_CANDIDATES}")

    out_dir = (Path(output_dir) if output_dir
               else Path(base_dirs[0]) / "STARR_outputs" / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    s00 = load_module("00_STARR_run_steps_01_to_05.py", "s00")

    print(f"\n{'═'*65}")
    print(f"00b — Confronto Estensione Donor")
    print(f"Estensioni da testare : {extents}")
    print(f"Output                : {out_dir}")
    print(f"{'═'*65}\n")

    all_results   = {}
    metrics_list  = []
    t_total       = time.time()

    for ext in extents:
        lbl = _extent_label(ext)
        print(f"\n{'─'*65}")
        print(f"▶  Extent: {lbl}")
        print(f"{'─'*65}")
        t0 = time.time()

        try:
            result = s00.main(
                run_step_05        = run_step_05,
                donor_extent_km    = ext,
                run_id_base        = run_id_base  or "",
                base_dir           = str(base_dirs[0]),
                fnf_shapefile      = fnf_shapefile,
                eligible_shapefile = eligible_shapefile,
                use_eligibility    = use_eligibility,
            )
            m = extract_metrics(result)
            all_results[lbl] = result
            metrics_list.append(m)
            ok = m["criteria"]["ALL_SUFFICIENT"]
            print(f"  ✓ Completato in {time.time()-t0:.0f}s | "
                  f"sufficient={'SI' if ok else 'NO'} | "
                  f"ratio={m['ratio_donor_project']:.1f}× | "
                  f"match={m['match_coverage_pct']:.1f}% | "
                  f"twin={m['twin_pass_pct']:.1f}% | "
                  f"smd={m['smd_max']:.3f}")
        except Exception as exc:
            print(f"  ✗ ERRORE extent={lbl}: {exc}")
            metrics_list.append({
                "extent_km": ext,
                "error": str(exc),
                "criteria": {k: False for k in
                             ["ratio_3x", "match_cov_90pct",
                              "twin_pass_30pct", "twin_compliant", "smd_le_010", "ALL_SUFFICIENT"]},
            })

        gc.collect()

    # ── Trovare extent minimo sufficiente ─────────────────────────────
    min_suff = find_minimum_sufficient_extent(metrics_list)

    if min_suff:
        rec = (
            f"Extent minimo sufficiente: {_extent_label(min_suff['extent_km'])}\n"
            f"  n_donor        = {min_suff['n_donor']:,} "
            f"({min_suff['ratio_donor_project']:.1f}× progetto)\n"
            f"  match_coverage = {min_suff['match_coverage_pct']:.1f}%\n"
            f"  twin_pass      = {min_suff['twin_pass_pct']:.1f}%\n"
            f"  smd_max        = {min_suff['smd_max']:.3f}\n"
            f"  reference_area = {min_suff['reference_area_ha']:.1f} ha"
        )
    else:
        rec = (
            "NESSUN extent soddisfa tutti i criteri.\n"
            "Azioni suggerite:\n"
            "  1. Aumentare il buffer massimo (> 30 km)\n"
            "  2. Verificare la qualità dei shapefile Eligible_FNF e FNF18\n"
            "  3. Rilassare PAIR_SLOPE_DIFF_MAX in Step 03 (0.005 → 0.01)\n"
            "  4. Verificare la copertura NDVI nel TIF donor"
        )

    print(f"\n{'═'*65}")
    print("RACCOMANDAZIONE")
    print(rec)
    print(f"Tempo totale: {time.time()-t_total:.0f}s")
    print(f"{'═'*65}")

    # ── Plot ─────────────────────────────────────────────────────────
    valid_m = [m for m in metrics_list if "error" not in m]
    fig = None
    if valid_m:
        fig = plot_comparison(
            valid_m, out_path=out_dir / "comparison_plots.png")

    # ── Salvataggio ───────────────────────────────────────────────────
    summary_out = {
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
        "extents_tested":      extents,
        "metrics":             metrics_list,
        "minimum_sufficient":  min_suff,
        "recommendation":      rec,
        "thresholds": {
            "ratio_3x":           THRESH_RATIO_3X,
            "ratio_3x_note":      "applicato su area donor eleggibile vs area PA piena (Step01 meets_3x_guideline_area)",
            "match_coverage_pct": THRESH_MATCH_COV_PCT,
            "twin_pass_pct":      THRESH_TWIN_PASS_PCT,
            "twin_compliant":     "twin_test_compliant deve essere True (Step03)",
            "smd_max":            THRESH_SMD_MAX,
        },
        "output_dir": str(out_dir),
    }
    with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    print(f"  Riepilogo JSON: {out_dir / 'comparison_summary.json'}")

    rec_path = out_dir / "recommendation.txt"
    rec_path.write_text(rec, encoding="utf-8")

    return {
        "metrics":             metrics_list,
        "minimum_sufficient":  min_suff,
        "recommendation":      rec,
        "output_dir":          str(out_dir),
        "fig":                 fig,
    }


if __name__ == "__main__":
    run_comparison()