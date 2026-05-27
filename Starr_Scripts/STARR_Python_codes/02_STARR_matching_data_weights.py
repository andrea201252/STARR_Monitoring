# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB  |  02_STARR_matching_data_weights.py
STEP 02 — Hard-Caliper + Weighted Mahalanobis KNN Matching
============================================================

PARAMETRI MANUALI:
  K_NEIGHBOURS          numero vicini KNN finali selezionati
  KNN_QUERY_CANDIDATES  candidati cercati prima dei calipers (≥ K_NEIGHBOURS)
  N_DONOR_SAMPLE        campionamento donor se None=tutti
  ALLOW_TEXTURE_FALLBACK
  MAX_DONOR_REUSE

NOVITÀ v04 rispetto alla versione precedente:
  - VECTORIZED HARD CALIPERS: il check dei calipers non usa più un loop
    Python per pixel per candidato, ma pre-estrae array numpy dal donor_df
    e fa le comparazioni in numpy → 10-50x più veloce.
  - Eliminato l'uso di pandas iloc per ogni candidato (era il bottleneck).
  - Logica KNN adattata dal QGIS PlotMatcherAlgorithm: usa la stessa metrica
    Mahalanobis (VI = inv(Σ_LedoitWolf)) ma con whitening Cholesky + ball_tree
    invece di cdist brute-force, che non scala oltre 50k pixel donor.

Hard calipers (GS STARR Annex 1 Table A.3):
  SOC_g_kg       ±10% della media di progetto
  NDVI_t0        ±10% della media di progetto
  elevation      ±200 m
  slope_deg      ±10°
  dist_roads_km  ±1 km
  texture_class  exact match (WRB2 → soil texture class)
  tenure_col     exact match se presente
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy.linalg import cholesky as scipy_cholesky
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf


def _is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except ImportError:
        return False


if not _is_notebook():
    matplotlib.use("Agg")


# ── PARAMETRI MANUALI ────────────────────────────────────────────────

K_NEIGHBOURS          = 1
KNN_QUERY_CANDIDATES  = 150   # candidati per pixel; adattativi al pool size
N_DONOR_SAMPLE        = 300_000
ALLOW_TEXTURE_FALLBACK = False
MAX_DONOR_REUSE       = 1

# ── BATCH KNN (FIX RAM) ───────────────────────────────────────────────
# zp_w NON viene mai precalcolato per tutti i project pixel.
# Lo scaling+whitening avviene on-the-fly per ogni batch.
# Memoria per batch: KNN_BATCH_SIZE × n_cov × 8 byte (es. 4096 × 20 × 8 = 640 KB)
# invece di 106K × n_cov × 8 = 17 MB tenuto in RAM permanentemente.
KNN_BATCH_SIZE        = 4096  # project pixel per query KNN (on-the-fly whitened)
KNN_N_JOBS            = -1
KNN_LEAF_SIZE         = 60
SCALER_FIT_MAX_ROWS   = 60_000   # campione per fit StandardScaler (proj+donor mix)
COV_FIT_MAX_ROWS      = 40_000   # campione per LedoitWolf (proj+donor mix)

# ── FIXED (GS STARR Annex 1 Table A.3) ───────────────────────────────

SMD_THRESHOLD                     = 0.1
RIDGE_REG                         = 1e-6
CALIPER_SOC_FRAC_OF_PROJECT_MEAN  = 0.10
CALIPER_NDVI_T0_FRAC_OF_PROJECT_MEAN = 0.10
CALIPER_ELEVATION_M               = 200.0
CALIPER_SLOPE_DEG                 = 10.0
CALIPER_ROADS_KM                  = 1.0

TENURE_COLUMN_CANDIDATES = [
    "tenure_status", "Tenure", "TENURE", "legal_status", "LegalStatus", "LEGAL_STATUS"
]

# ── WRB TEXTURE ──────────────────────────────────────────────────────

WRB_TO_TEXTURE = {
    1:"clay_loam", 2:"clay", 3:"loam", 4:"sand", 5:"loam",
    6:"clay_loam", 7:"loam", 8:"clay_loam", 9:"loam", 10:"loam",
    11:"clay", 12:"loam", 13:"clay_loam", 14:"sandy_loam", 15:"loam",
    16:"clay_loam", 17:"sandy_loam", 18:"sandy_loam", 19:"clay_loam",
    20:"clay", 21:"clay_loam", 22:"clay_loam", 23:"clay", 24:"sandy_loam",
    25:"loam", 26:"sandy_loam", 27:"clay", 28:"clay_loam", 29:"loam", 30:"clay",
}
TEXTURE_FALLBACK = {
    "sand":      ["sand","sandy_loam","loam","clay_loam","clay"],
    "sandy_loam":["sandy_loam","sand","loam","clay_loam","clay"],
    "loam":      ["loam","sandy_loam","clay_loam","sand","clay"],
    "clay_loam": ["clay_loam","loam","clay","sandy_loam","sand"],
    "clay":      ["clay","clay_loam","loam","sandy_loam","sand"],
}


# ── IO ────────────────────────────────────────────────────────────────

def format_int_or_all(v):
    return "tutti" if v is None else f"{int(v):,}"


def load_df(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File non trovato: {p}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def find_file(base_dirs, stem, fmt="parquet"):
    for base in base_dirs:
        for ext in [fmt, "parquet", "csv"]:
            p = Path(base) / f"{stem}.{ext}"
            if p.exists():
                print(f"    Trovato: {p.name}")
                return p
    raise FileNotFoundError(f"'{stem}' non trovato in {[str(b) for b in base_dirs]}")


def wrb_to_texture(code):
    try:
        return WRB_TO_TEXTURE.get(int(float(code)))
    except Exception:
        return None


def assign_texture(df):
    df = df.copy()
    if "WRB2_CODE" not in df.columns:
        raise ValueError("WRB2_CODE mancante. Verificare raster GEE Step 01.")
    df["texture_class"] = df["WRB2_CODE"].apply(wrb_to_texture)
    before = len(df)
    df = df[df["texture_class"].notna()].reset_index(drop=True)
    print(f"    WRB filter: {before:,} → {len(df):,} px validi")
    if len(df) == 0:
        raise RuntimeError("Nessun pixel valido dopo filtro WRB.")
    return df


def check_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: colonne mancanti: {missing}")


def detect_ndvi_year_cols(columns):
    import re
    return sorted([c for c in columns if re.match(r"^NDVI_\d{4}$", str(c))])


def detect_tenure_col(proj_df, donor_df):
    for c in TENURE_COLUMN_CANDIDATES:
        if c in proj_df.columns and c in donor_df.columns:
            return c
    return None


# ── PREFILTER (loose) ─────────────────────────────────────────────────

def auto_prefilter_donor(proj_df, donor_df, cont_covs):
    """Filtro largo sul donor (±20% del range). I calipers esatti vengono dopo."""
    out = donor_df.copy()
    for col in cont_covs:
        if col not in proj_df.columns or col not in out.columns:
            continue
        q01 = proj_df[col].quantile(0.01)
        q99 = proj_df[col].quantile(0.99)
        rng = q99 - q01
        if not np.isfinite(rng):
            continue
        buf = max(rng * 0.20, 1e-6)
        lo, hi = q01 - buf, q99 + buf
        before = len(out)
        out = out[(out[col] >= lo) & (out[col] <= hi)]
        print(f"    Prefilter {col:18s}: [{lo:.3f}, {hi:.3f}] {before:,} → {len(out):,}")
        if len(out) == 0:
            raise RuntimeError(f"Donor vuoto dopo prefilter '{col}'.")
    return out.reset_index(drop=True)


def build_plain_mahalanobis(z_sample, cont_covs):
    """
    Plain Mahalanobis: Sigma^-1 via LedoitWolf su un campione.

    FIX METH: rimossi i pesi RF (metodologicamente scorretti per GS STARR).
    La metrica è la plain D(i,j) = sqrt((Xi-Xj)' Sigma^-1 (Xi-Xj)).

    FIX RAM: accetta solo un campione (z_sample, al massimo COV_FIT_MAX_ROWS righe),
    non l'intero array z_proj+z_donor (che sarebbe 406K righe in RAM).

    Restituisce (metric=Sigma^-1+ridge, cov_orig=Sigma).
    """
    n_cov = len(cont_covs)
    lw = LedoitWolf(assume_centered=False)
    lw.fit(z_sample.astype(np.float64))
    cov    = lw.covariance_ + RIDGE_REG * np.eye(n_cov)
    metric = np.linalg.inv(cov) + RIDGE_REG * np.eye(n_cov)
    return metric, lw.covariance_


# ── GC CONDIZIONALE ──────────────────────────────────────────────────

try:
    import psutil as _psutil; _PSUTIL = True
except ImportError:
    _PSUTIL = False

_GC_RAM_TRESH = 0.82

def _gc(force=False):
    """GC solo se RAM > soglia o force=True. Evita overhead nel hot loop."""
    if force: gc.collect(); return
    if _PSUTIL and _psutil.virtual_memory().percent / 100.0 > _GC_RAM_TRESH:
        gc.collect()


def adaptive_k(n_donor_pool, base_k=KNN_QUERY_CANDIDATES):
    """Candidates adattativi: se il pool è piccolo, interroga tutto."""
    return max(K_NEIGHBOURS, min(base_k, int(n_donor_pool * 0.95), n_donor_pool))


def build_whitening_matrix(metric):
    """Cholesky L t.c. Mahal(a,b)² = ||aL - bL||² (whitened Euclidean)."""
    eig = np.linalg.eigvalsh(metric)
    if eig.min() < 1e-10:
        metric = metric + (abs(eig.min()) + 1e-8) * np.eye(metric.shape[0])
    return scipy_cholesky(metric, lower=True)


# ── PRE-CACHING NUMPY ARRAY PER DONOR ────────────────────────────────

def _build_donor_arrays(donor_df, cont_covs, tenure_col, ctx):
    """
    Pre-estrae le colonne rilevanti dal donor_df come array numpy.
    Questo evita la chiamata a pandas iloc per ogni candidato durante
    il check dei calipers (era il bottleneck principale).
    """
    arr = {}
    arr["texture_class"] = donor_df["texture_class"].values.astype(str)

    if tenure_col and tenure_col in donor_df.columns:
        arr["tenure"] = donor_df[tenure_col].values.astype(str)
    else:
        arr["tenure"] = None

    for col in ["SOC_g_kg", "NDVI_t0", "elevation", "slope_deg", "dist_roads_km",
                "clay_pct", "precip_mm_yr"]:
        if col in donor_df.columns:
            arr[col] = donor_df[col].values.astype(float)
        else:
            arr[col] = None

    # Precompute tolerance context
    arr["ctx"] = ctx
    return arr


# ── VECTORIZED CALIPER CHECK ──────────────────────────────────────────

def _check_calipers_vectorized(proj_row, cand_global_idx, donor_arr, ctx, tenure_col):
    """
    Controlla i calipers su TUTTI i candidati in una sola passata numpy.
    
    Input:
      proj_row:        Series (pandas row del pixel di progetto)
      cand_global_idx: array int, indici globali nel donor_df dei candidati KNN
      donor_arr:       dict di array numpy pre-estratti dal donor_df
    
    Returns:
      passes: bool array di lunghezza len(cand_global_idx)
      n_rejected: int, numero di candidati rifiutati
    """
    n      = len(cand_global_idx)
    passes = np.ones(n, dtype=bool)

    # 1. Texture exact
    proj_tex = str(proj_row.get("texture_class", ""))
    passes  &= (donor_arr["texture_class"][cand_global_idx] == proj_tex)

    # 2. Tenure exact (se disponibile)
    if donor_arr["tenure"] is not None and tenure_col:
        proj_ten = str(proj_row.get(tenure_col, ""))
        passes  &= (donor_arr["tenure"][cand_global_idx] == proj_ten)

    # 3. SOC ±10%
    if donor_arr["SOC_g_kg"] is not None and ctx.get("soc_tol"):
        pv = float(proj_row.get("SOC_g_kg", np.nan))
        if np.isfinite(pv):
            passes &= np.abs(donor_arr["SOC_g_kg"][cand_global_idx] - pv) <= ctx["soc_tol"]

    # 4. NDVI_t0 ±10%
    if donor_arr["NDVI_t0"] is not None and ctx.get("ndvi_tol"):
        pv = float(proj_row.get("NDVI_t0", np.nan))
        if np.isfinite(pv):
            passes &= np.abs(donor_arr["NDVI_t0"][cand_global_idx] - pv) <= ctx["ndvi_tol"]

    # 5. Elevation ±200 m
    if donor_arr["elevation"] is not None:
        pv = float(proj_row.get("elevation", np.nan))
        if np.isfinite(pv):
            passes &= np.abs(donor_arr["elevation"][cand_global_idx] - pv) <= CALIPER_ELEVATION_M

    # 6. Slope ±10°
    if donor_arr["slope_deg"] is not None:
        pv = float(proj_row.get("slope_deg", np.nan))
        if np.isfinite(pv):
            passes &= np.abs(donor_arr["slope_deg"][cand_global_idx] - pv) <= CALIPER_SLOPE_DEG

    # 7. Roads ±1 km
    if donor_arr["dist_roads_km"] is not None:
        pv = float(proj_row.get("dist_roads_km", np.nan))
        if np.isfinite(pv):
            passes &= np.abs(donor_arr["dist_roads_km"][cand_global_idx] - pv) <= CALIPER_ROADS_KM

    n_rejected = int((~passes).sum())
    return passes, n_rejected


# ── MATCHING ─────────────────────────────────────────────────────────

def run_matching(proj_df, donor_df, weights_dict, cont_covs, meta):
    """
    Matching stratificato per texture WRB2 con calipers hard vectorizzati.
    
    La logica KNN è adattata dal QGIS PlotMatcherAlgorithm:
      - stessa metrica Mahalanobis (VI = Σ^-1 pesata)
      - ma usa whitening + ball_tree invece di cdist brute-force
        (cdist è O(n*m) e non scala oltre 50k pixel donor)
      - i calipers hard vengono applicati vectorialmente sui candidati KNN
        invece di un loop Python per candidato (bottleneck eliminato)
    """
    if N_DONOR_SAMPLE is not None and len(donor_df) > int(N_DONOR_SAMPLE):
        print(f"    Campionamento donor: {len(donor_df):,} → {int(N_DONOR_SAMPLE):,}")
        donor_df = donor_df.sample(int(N_DONOR_SAMPLE), random_state=42).reset_index(drop=True)

    check_cols(proj_df,  cont_covs + ["lon", "lat", "texture_class"], "project_df")
    check_cols(donor_df, cont_covs + ["lon", "lat", "texture_class"], "donor_df")

    # Contesto calipers
    tenure_col = detect_tenure_col(proj_df, donor_df)
    ctx = {"tenure_col": tenure_col}
    if "SOC_g_kg" in proj_df.columns:
        ctx["soc_tol"] = max(abs(float(proj_df["SOC_g_kg"].mean())) * CALIPER_SOC_FRAC_OF_PROJECT_MEAN, 1e-9)
    if "NDVI_t0" in proj_df.columns:
        ctx["ndvi_tol"] = max(abs(float(proj_df["NDVI_t0"].mean())) * CALIPER_NDVI_T0_FRAC_OF_PROJECT_MEAN, 1e-9)

    print("    Hard calipers attivi:")
    print(f"      texture exact : YES")
    print(f"      tenure exact  : {'YES ('+tenure_col+')' if tenure_col else 'N/A'}")
    print(f"      SOC           : ±{ctx.get('soc_tol', 'N/A'):.4f}")
    print(f"      NDVI_t0       : ±{ctx.get('ndvi_tol', 'N/A'):.4f}")
    print(f"      elevation     : ±{CALIPER_ELEVATION_M:.0f} m")
    print(f"      slope         : ±{CALIPER_SLOPE_DEG:.0f}°")
    print(f"      dist_roads    : ±{CALIPER_ROADS_KM:.1f} km")

    # ── SCALING + WHITENING + KNN BATCH (FIX RAM) ───────────────────────
    #
    # PROBLEMA (OOM con 106K project pixel):
    #   z_all = scaler.fit_transform(pd.concat([proj, donor]))  -> 406K x n_cov RAM
    #   zp_w  = z_proj @ L                                      -> 106K x n_cov permanente
    #   knn.kneighbors(zp_w[proj_idx]) su tutti 106K            -> (106K x 300) x 16 byte = 508 MB output
    #
    # FIX: zp_w NON viene mai precalcolato per tutti i project pixel.
    # Scale + whitening vengono applicati on-the-fly per KNN_BATCH_SIZE pixel alla volta.
    # RAM per batch: 4096 x n_cov x 8 = 640 KB invece di 17 MB in RAM permanente.

    rng   = np.random.default_rng(42)
    n_cov = len(cont_covs)

    # 1. Fit scaler su campione (no concat full proj+donor in RAM)
    n_p_s = min(len(proj_df),  SCALER_FIT_MAX_ROWS // 2)
    n_d_s = min(len(donor_df), SCALER_FIT_MAX_ROWS // 2)
    Xp_s  = proj_df.iloc[rng.choice(len(proj_df),  n_p_s, replace=False)][cont_covs].to_numpy(dtype=np.float64)
    Xd_s  = donor_df.iloc[rng.choice(len(donor_df), n_d_s, replace=False)][cont_covs].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    scaler.fit(np.vstack([Xp_s, Xd_s]))

    # 2. LedoitWolf su campione scalato (no concat intero proj+donor)
    z_samp = np.vstack([scaler.transform(Xp_s), scaler.transform(Xd_s)]).astype(np.float64)
    del Xp_s, Xd_s
    metric, cov_orig = build_plain_mahalanobis(z_samp, cont_covs)
    del z_samp
    L = build_whitening_matrix(metric).astype(np.float64)

    # 3. Trasforma + whiten TUTTO il donor (necessario per l'indice KNN)
    #    Max 300K x n_cov x 8 = 48 MB — accettabile
    Xd_full = donor_df[cont_covs].to_numpy(dtype=np.float64)
    zdw     = (scaler.transform(Xd_full) @ L).astype(np.float64)
    del Xd_full
    gc.collect()

    # 4. KNN globale su tutto il donor whitened (costruito UNA SOLA VOLTA)
    k_global = min(max(K_NEIGHBOURS, KNN_QUERY_CANDIDATES), len(zdw))
    knn_global = NearestNeighbors(
        n_neighbors=k_global, metric="euclidean",
        algorithm="ball_tree", leaf_size=KNN_LEAF_SIZE, n_jobs=KNN_N_JOBS
    )
    knn_global.fit(zdw)
    print(f"    KNN globale: {len(zdw):,} donor | k={k_global} | n_jobs={KNN_N_JOBS}")
    print(f"    FIX RAM: zp_w on-the-fly per batch di {KNN_BATCH_SIZE} px (mai 106K in memoria)")

    donor_df   = donor_df.reset_index(drop=True)
    donor_arr  = _build_donor_arrays(donor_df, cont_covs, tenure_col, ctx)
    donor_reuse = np.zeros(len(donor_df), dtype=np.int32)

    available_set = set(donor_df["texture_class"].dropna().unique())
    print(f"    Texture groups donor: {sorted(available_set)}")

    records, tex_rows, unmatched_rows = [], [], []
    run_id   = meta.get("run_id", "")
    t0_total = time.time()

    for texture, proj_sub in proj_df.groupby("texture_class", sort=True):
        proj_idx = proj_sub.index.to_numpy(dtype=np.int64)

        if texture in available_set:
            allowed, exact = [texture], True
        elif ALLOW_TEXTURE_FALLBACK:
            allowed = [t for t in TEXTURE_FALLBACK.get(texture, []) if t in available_set]
            exact   = False
        else:
            allowed, exact = [], False

        if not allowed:
            print(f"    SKIP texture '{texture}': nessun donor.")
            for pi in proj_idx:
                unmatched_rows.append({"proj_idx": int(pi), "proj_texture": str(texture),
                                       "reason": "no_texture_donor"})
            continue

        d_mask   = np.isin(donor_arr["texture_class"], [str(a) for a in allowed])
        d_global = np.where(d_mask)[0]
        if not len(d_global):
            continue

        k_query = adaptive_k(len(d_global))

        n_matched = n_unmatched = rejected_total = 0
        dist_sum  = dist_n = 0.0

        # ── BATCH LOOP: scale+whiten on-the-fly per KNN_BATCH_SIZE pixel ──────
        for start in range(0, len(proj_idx), KNN_BATCH_SIZE):
            batch = proj_idx[start : start + KNN_BATCH_SIZE]

            # ON-THE-FLY: scala + whiten solo questo batch
            # RAM: KNN_BATCH_SIZE x n_cov x 8 = 640 KB (non 17 MB)
            Xp_b  = proj_df.loc[batch, cont_covs].to_numpy(dtype=np.float64)
            zpw_b = (scaler.transform(Xp_b) @ L).astype(np.float64)
            del Xp_b

            # Query KNN globale
            dists_g, idx_g = knn_global.kneighbors(zpw_b)
            del zpw_b

            for row_i in range(len(batch)):
                pi = int(batch[row_i])
                pr = proj_df.loc[pi]

                # Filtra per texture sul risultato globale
                cands_all   = idx_g[row_i]
                dists_all   = dists_g[row_i]
                tex_ok      = np.isin(donor_arr["texture_class"][cands_all],
                                       [str(a) for a in allowed])
                cand_global = cands_all[tex_ok][:k_query]
                cand_dists  = dists_all[tex_ok][:k_query]

                if len(cand_global) == 0:
                    n_unmatched += 1
                    unmatched_rows.append({"proj_idx": pi, "proj_texture": str(texture),
                                           "reason": "no_texture_compatible_in_knn"})
                    continue

                dist_sum += float(cand_dists[0]); dist_n += 1
                passes, n_rej = _check_calipers_vectorized(pr, cand_global, donor_arr, ctx, tenure_col)
                rejected_total += n_rej
                passing_local = np.where(passes)[0]

                if not len(passing_local):
                    n_unmatched += 1
                    unmatched_rows.append({"proj_idx": pi, "proj_texture": str(texture),
                                           "reason": "no_candidate_passed_hard_calipers",
                                           "queried_n": int(k_query)})
                    continue

                best_local  = int(passing_local[0])
                best_global = int(cand_global[best_local])
                best_dist   = float(cand_dists[best_local])
                reuse       = int(donor_reuse[best_global])
                exceeded    = bool(reuse >= MAX_DONOR_REUSE)
                donor_reuse[best_global] += 1

                dr  = donor_df.iloc[best_global]
                rec = {c: dr.get(c) for c in donor_df.columns if c != "_gidx"}
                rec.update({
                    "run_id":         run_id,
                    "ref_lon":        float(dr["lon"]),
                    "ref_lat":        float(dr["lat"]),
                    "proj_lon":       float(pr["lon"]),
                    "proj_lat":       float(pr["lat"]),
                    "match_distance": best_dist,
                    "match_rank":     int(best_local + 1),
                    "proj_idx":       pi,
                    "proj_texture":   str(texture),
                    "ref_texture":    str(dr.get("texture_class", "")),
                    "texture_exact":  bool(exact),
                    "reuse_exceeded": exceeded,
                    "all_calipers_passed": True,
                    "hard_caliper_rejected_before_selected": n_rej,
                })
                for c in ["SOC_g_kg", "NDVI_t0", "elevation", "slope_deg", "dist_roads_km"]:
                    if c in pr.index and c in dr.index:
                        rec[f"proj_{c}"] = float(pr[c])
                        rec[f"ref_{c}"]  = float(dr[c])
                        rec[f"diff_{c}"] = float(dr[c]) - float(pr[c])
                records.append(rec)
                n_matched += 1

            del dists_g, idx_g
            _gc()

        tex_rows.append({
            "texture": texture, "proj_n": int(len(proj_idx)),
            "donor_n": int(len(d_global)), "k_query": k_query,
            "exact_texture": bool(exact), "matched_n": n_matched,
            "unmatched_n": n_unmatched,
            "hard_caliper_rejected_total": int(rejected_total),
            "mean_nn_dist_unfiltered": float(dist_sum / dist_n) if dist_n else np.nan,
        })
        elapsed = time.time() - t0_total
        print(f"    Texture {texture:10s}: proj={len(proj_idx):,} donor={len(d_global):,} "
              f"matched={n_matched:,} unmatched={n_unmatched:,} "
              f"k_query={k_query} [{elapsed:.0f}s]")

    matched   = pd.DataFrame(records)
    unmatched = pd.DataFrame(unmatched_rows)
    if not matched.empty:
        matched = matched.sort_values("proj_idx").reset_index(drop=True)
        matched = matched.drop(columns=["lon", "lat"], errors="ignore")
    return matched, pd.DataFrame(tex_rows), unmatched


# ── VALIDAZIONE ───────────────────────────────────────────────────────

def compute_smd(proj_df, ref_df, covs):
    rows = []
    for cov in covs:
        if cov not in proj_df.columns or cov not in ref_df.columns:
            continue
        p   = proj_df[cov].dropna().astype(float).values
        r   = ref_df[cov].dropna().astype(float).values
        if len(p) == 0 or len(r) == 0:
            rows.append({"covariate": cov, "SMD": np.nan,
                         "proj_mean": np.nan, "ref_mean": np.nan, "passed": False})
            continue
        pooled = float(np.sqrt((np.var(p) + np.var(r)) / 2.0))
        smd    = abs(np.mean(p) - np.mean(r)) / pooled if pooled > 0 else 0.0
        rows.append({"covariate": cov, "SMD": round(smd, 4),
                     "proj_mean": round(float(p.mean()), 6),
                     "ref_mean":  round(float(r.mean()), 6),
                     "passed":    bool(smd < SMD_THRESHOLD)})
    if not rows:
        raise RuntimeError("Nessuna covariata valida per SMD.")
    return pd.DataFrame(rows).set_index("covariate")


def compute_caliper_audit(matched_df):
    flag_cols = [c for c in matched_df.columns
                 if c.endswith("_caliper") or c in
                 ("texture_exact","tenure_exact","precip_bin_exact","all_calipers_passed")]
    rows = []
    for c in flag_cols:
        v = matched_df[c].dropna().astype(bool)
        rows.append({"criterion": c, "n": int(len(v)),
                     "passed_n": int(v.sum()),
                     "passed_pct": round(float(v.mean()*100), 2) if len(v) else np.nan})
    return pd.DataFrame(rows)


# ── PLOT ─────────────────────────────────────────────────────────────

def plot_rf_weights(imp_df, out_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    max_imp = max(float(imp_df["rf_importance"].max()), 1e-12)
    max_w   = max(float(imp_df["knn_weight"].max()),   1e-12)

    ax = axes[0]
    ax.barh(imp_df["covariate"], imp_df["rf_importance"],
            color=plt.cm.RdYlGn(imp_df["rf_importance"]/max_imp))
    ax.axvline(imp_df["rf_importance"].mean(), color="gray", ls="--", alpha=0.7)
    ax.set(xlabel="RF Feature Importance",
           title="Feature importance\nproject vs donor discrimination")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    s = imp_df.sort_values("knn_weight")
    ax.barh(s["covariate"], s["knn_weight"],
            color=plt.cm.RdYlGn(s["knn_weight"]/max_w))
    ax.axvline(1.0, color="black", ls=":", alpha=0.4, label="w=1 (neutro)")
    ax.set(xlabel="KNN weight (data-driven)",
           title="Pesi Mahalanobis data-driven")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "rf_weights.png", dpi=150, bbox_inches="tight")
    return fig


def plot_smd(smd_df, out_dir=None, filename="SMD_lollipop.png", title="Covariate balance"):
    fig, ax = plt.subplots(figsize=(9, max(4, len(smd_df)*0.35)))
    colors = ["#d62728" if not p else "#2ca02c" for p in smd_df["passed"]]
    ax.hlines(smd_df.index, 0, smd_df["SMD"].fillna(0).values, lw=2, color=colors)
    ax.scatter(smd_df["SMD"].fillna(0).values, smd_df.index, s=70, color=colors, zorder=5)
    ax.axvline(SMD_THRESHOLD, color="black", ls="--", label=f"SMD={SMD_THRESHOLD}")
    ax.set(xlabel="SMD (GS STARR: < 0.1)", title=title)
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_match_distances(matched_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    d = matched_df["match_distance"].astype(float).values
    ax.hist(d, bins=60, edgecolor="white", color="#4C72B0")
    ax.axvline(d.mean(), color="red", ls="--", label=f"Media={d.mean():.3f}")
    ax.axvline(np.percentile(d, 95), color="orange", ls=":",
               label=f"P95={np.percentile(d,95):.3f}")
    ax.set(xlabel="Whitened Mahalanobis distance", ylabel="Pixels",
           title="Distribuzione distanze di match")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "match_distance_distribution.png", dpi=150, bbox_inches="tight")
    return fig


# ── MAIN ─────────────────────────────────────────────────────────────

def run_matching_step(base_dirs=None, output_dir=None,
                      proj_df=None, donor_df=None, meta=None, verbose=True):
    """
    Restituisce (matched_df, weights_dict, imp_df, smd_df, figs, out_dir).
    """
    if base_dirs is None:
        raise RuntimeError(
            "base_dirs non fornito. Passare base_dirs=[out01] dal runner "
            "oppure specificare la directory di output dello Step 01."
        )
    base_dirs = [Path(b) for b in base_dirs]
    out_dir   = Path(output_dir) if output_dir else base_dirs[0].parent / "02_matching"
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 02 — Matching | K={K_NEIGHBOURS} | "
              f"candidates={KNN_QUERY_CANDIDATES} | N_donor={format_int_or_all(N_DONOR_SAMPLE)}")
        print(f"Output: {out_dir}")
        print(f"{'='*60}")

    if proj_df is None:
        proj_df  = load_df(find_file(base_dirs, "project_pixels_raw"))
    if donor_df is None:
        donor_df = load_df(find_file(base_dirs, "donor_pixels_raw"))
    if meta is None:
        rp = base_dirs[0] / "extraction_report.json"
        meta = json.load(open(rp)) if rp.exists() else {}

    cont_covs = [c for c in (meta.get("continuous_covariates") or [])
                 if c not in ("pixel_area_ha", "ndvi_valid_years")
                 and not c.startswith("precip_bin")]
    if not cont_covs:
        raise RuntimeError("continuous_covariates vuoto. Eseguire Step 01.")

    ndvi_year_cols = meta.get("ndvi_year_cols") or detect_ndvi_year_cols(proj_df.columns)
    print(f"    Covariate Mahalanobis: {cont_covs}")
    print(f"    NDVI annuali: {ndvi_year_cols}")

    check_cols(proj_df,  cont_covs + ["WRB2_CODE", "lon", "lat"], "project_df")
    check_cols(donor_df, cont_covs + ["WRB2_CODE", "lon", "lat"], "donor_df")

    print("\n[1] Texture WRB...")
    proj_df  = assign_texture(proj_df)
    donor_df = assign_texture(donor_df)

    proj_df  = proj_df.dropna(subset=cont_covs).reset_index(drop=True)
    donor_df = donor_df.dropna(subset=cont_covs).reset_index(drop=True)
    print(f"    Project: {len(proj_df):,} | Donor: {len(donor_df):,}")
    if len(proj_df) == 0 or len(donor_df) == 0:
        raise RuntimeError("Dataset vuoto dopo dropna covariate.")

    print("\n[2] Prefilter donor (loose)...")
    donor_df = auto_prefilter_donor(proj_df, donor_df, cont_covs)

    weights_dict = {c: 1.0 for c in cont_covs}  # plain Mahalanobis: pesi uniformi (no RF)
    imp_df = None

    print("\n[3] Matching KNN plain Mahalanobis + hard calipers (batched, no RF)...")
    matched_df, tex_summary, unmatched_df = run_matching(
        proj_df, donor_df, weights_dict, cont_covs, meta)

    if matched_df.empty:
        raise RuntimeError("Nessun pixel matchato dopo hard calipers. "
                           "Aumentare KNN_QUERY_CANDIDATES o abilitare ALLOW_TEXTURE_FALLBACK.")

    n_matched  = len(matched_df)
    n_unique   = matched_df[["ref_lon","ref_lat"]].drop_duplicates().shape[0]
    n_unmatch  = len(unmatched_df)
    reuse_n    = int(matched_df.get("reuse_exceeded", pd.Series(dtype=bool)).sum()) \
                 if "reuse_exceeded" in matched_df.columns else 0

    print(f"\n    Matched   : {n_matched:,}")
    print(f"    Unmatched : {n_unmatch:,}")
    print(f"    Unique ref: {n_unique:,}")

    print(f"\n[5] SMD validation (< {SMD_THRESHOLD})...")
    smd_df = compute_smd(proj_df, matched_df, cont_covs)
    for cov, row in smd_df.iterrows():
        print(f"    {'PASS' if row['passed'] else 'FAIL':4s} {cov:20s}: SMD={row['SMD']:.4f}")

    # ndvi_smd_df = compute_smd(proj_df, matched_df, ndvi_year_cols) \
    #               if ndvi_year_cols else pd.DataFrame()
    # if not ndvi_smd_df.empty:
    #     print("\n    Annual NDVI balance:")
    #     for cov, row in ndvi_smd_df.iterrows():
    #         print(f"    {'PASS' if row['passed'] else 'FAIL':4s} {cov:20s}: SMD={row['SMD']:.4f}")

    caliper_audit = compute_caliper_audit(matched_df)

    print("\n[6] Plot + salvataggio...")
    fig_s = plot_smd(smd_df, out_dir, "SMD_lollipop.png", "Mandatory covariate balance")
    fig_d = plot_match_distances(matched_df, out_dir)
    # fig_n = plot_smd(ndvi_smd_df, out_dir, "SMD_annual_NDVI.png",
                    #  "Annual NDVI balance") if not ndvi_smd_df.empty else None

    matched_df.to_parquet(out_dir / "reference_area_pixels.parquet", index=False)
    matched_df.to_csv(out_dir / "reference_area_pixels.csv", index=False)
    unmatched_df.to_csv(out_dir / "unmatched_project_pixels.csv", index=False)
    # imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    smd_df.to_csv(out_dir / "SMD_table.csv")
    # if not ndvi_smd_df.empty:
        # ndvi_smd_df.to_csv(out_dir / "SMD_annual_NDVI_table.csv")
    tex_summary.to_csv(out_dir / "texture_summary.csv", index=False)
    caliper_audit.to_csv(out_dir / "caliper_audit.csv", index=False)

    summary = {
        "run_id":                meta.get("run_id",""),
        "timestamp_utc":         datetime.now(timezone.utc).isoformat(),
        "matching_method":       "Plain Mahalanobis KNN batched on-the-fly (no RF weights)",
        "donor_pool_rule":       "Non-Forest at T0, >5km from PA (GEE Annex A.2.2 Step A)",
        "k_neighbours":          int(K_NEIGHBOURS),
        "knn_query_candidates":  int(KNN_QUERY_CANDIDATES),
        "n_donor_sample":        None if N_DONOR_SAMPLE is None else int(N_DONOR_SAMPLE),
        "allow_texture_fallback":bool(ALLOW_TEXTURE_FALLBACK),
        "weights":               weights_dict,
        "matched_n":             int(n_matched),
        "unmatched_project_n":   int(n_unmatch),
        "unique_donor_n":        int(n_unique),
        "reuse_exceeded_n":      int(reuse_n),
        "SMD_max":               float(smd_df["SMD"].max()),
        "SMD_all_passed":        bool(smd_df["passed"].all()),
        # "NDVI_annual_SMD_max":   None if ndvi_smd_df.empty else float(ndvi_smd_df["SMD"].max()),
        "hard_calipers_all_passed": bool(matched_df["all_calipers_passed"].all())
                                    if "all_calipers_passed" in matched_df.columns else False,
        "cont_covs":             cont_covs,
        "ndvi_year_cols":        ndvi_year_cols,
        "next_step":             "03_STARR_twin_test_selection.py",
    }
    with open(out_dir / "matching_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Matched          : {n_matched:,}")
        print(f"  Unmatched        : {n_unmatch:,}")
        print(f"  SMD max          : {smd_df['SMD'].max():.4f}")
        # if not ndvi_smd_df.empty:
        #     print(f"  NDVI SMD max     : {ndvi_smd_df['SMD'].max():.4f}")
        print(f"  Output           : {out_dir}")
        print(f"{'='*60}")

    # figs = {"smd": fig_s, "distances": fig_d, "ndvi_smd": fig_n}
    figs = {"smd": fig_s, "distances": fig_d}
    return matched_df, weights_dict, imp_df, smd_df, figs, out_dir


def main():
    run_matching_step()


if __name__ == "__main__":
    main()
