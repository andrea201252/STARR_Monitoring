# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB | 02_STARR_matching_data_weights.py
STEP 02 — Streaming Plain Mahalanobis KNN + Hard Calipers + Correct SMD
[v8 — no downcast, donor streamed, SMD on matched project subset]
============================================================

Critical fixes
--------------
1. No Random Forest weights. Matching uses plain Mahalanobis distance:
       D(i,j) = sqrt((Xi - Xj)' Sigma^-1 (Xi - Xj))

2. No numeric downcast. Values remain float64 unless they were already smaller
   in the source file. KNN/scaling/covariance arrays are built as float64.

3. Donor is not loaded fully into RAM when donor_df=None. The donor parquet/csv
   is streamed in batches, filtered against project covariate ranges, and only a
   capped stratified sample is kept for KNN.

4. SMD is computed on the matched project subset versus the matched reference
   subset. The previous logic compared all eligible project pixels against the
   matched controls, which can give invalid balance diagnostics when some PA
   pixels are outside common support or unmatched.

5. Full match CSV is disabled by default. Parquet is the canonical output.
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib


def _is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


if not _is_notebook():
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from scipy.linalg import cholesky as scipy_cholesky
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf


# ================================================================
# USER PARAMETERS
# ================================================================

K_NEIGHBOURS = 1
KNN_QUERY_CANDIDATES = 80
N_DONOR_SAMPLE = 300_000
ALLOW_TEXTURE_FALLBACK = False
MAX_DONOR_REUSE = 1
ALLOW_DONOR_REUSE_OVERFLOW = False
STRICT_COMPLIANCE_STOP = True
MIN_PROJECT_MATCH_COVERAGE = 0.90

# Memory controls. No downcast is used; reduce these if RAM is still saturated.
DONOR_STREAM_BATCH_ROWS = 250_000
SCALER_FIT_MAX_DONOR = 80_000
COV_FIT_MAX_DONOR = 60_000
KNN_BATCH_SIZE = 1024
KNN_N_JOBS = 1
FLOAT_DTYPE = np.float64

WRITE_FULL_MATCH_CSV = False
MATCH_PREVIEW_CSV_ROWS = 20_000

SMD_THRESHOLD = 0.1
RIDGE_REG = 1e-6
CALIPER_SOC_FRAC_OF_PROJECT_MEAN = 0.10
CALIPER_NDVI_T0_FRAC_OF_PROJECT_MEAN = 0.10
CALIPER_ELEVATION_M = 200.0
CALIPER_SLOPE_DEG = 10.0
CALIPER_ROADS_KM = 1.0

SUPPORT_QUANTILE_LO = 0.005
SUPPORT_QUANTILE_HI = 0.995
SUPPORT_BUFFER_FRAC = 0.10

RUN_CALIPER_FEASIBILITY_SCREEN = False
FEASIBILITY_BATCH_SIZE = 128

TENURE_COLUMN_CANDIDATES = [
    "tenure_status", "Tenure", "TENURE", "legal_status", "LegalStatus", "LEGAL_STATUS"
]

WRB_TO_TEXTURE = {
    1:"clay_loam",2:"clay",3:"loam",4:"sand",5:"loam",6:"clay_loam",7:"loam",
    8:"clay_loam",9:"loam",10:"loam",11:"clay",12:"loam",13:"clay_loam",
    14:"sandy_loam",15:"loam",16:"clay_loam",17:"sandy_loam",18:"sandy_loam",
    19:"clay_loam",20:"clay",21:"clay_loam",22:"clay_loam",23:"clay",
    24:"sandy_loam",25:"loam",26:"sandy_loam",27:"clay",28:"clay_loam",
    29:"loam",30:"clay",
}
TEXTURE_ORDER = ["sand", "sandy_loam", "loam", "clay_loam", "clay"]
TEXTURE_TO_CODE = {name: i for i, name in enumerate(TEXTURE_ORDER)}
TEXTURE_FALLBACK = {
    "sand":       ["sand", "sandy_loam", "loam", "clay_loam", "clay"],
    "sandy_loam": ["sandy_loam", "sand", "loam", "clay_loam", "clay"],
    "loam":       ["loam", "sandy_loam", "clay_loam", "sand", "clay"],
    "clay_loam":  ["clay_loam", "loam", "clay", "sandy_loam", "sand"],
    "clay":       ["clay", "clay_loam", "loam", "sandy_loam", "sand"],
}

PIXEL_CELL_FIELDS = [
    "pixel_id", "source_tile",
    "grid_row", "grid_col",
    "x_utm", "y_utm", "centroid_x_utm", "centroid_y_utm",
    "lon", "lat", "centroid_lon", "centroid_lat",
    "cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax",
    "cell_width_m", "cell_height_m",
]


# ================================================================
# IO / COLUMN LOADING
# ================================================================

def format_int_or_all(v):
    return "tutti" if v is None else f"{int(v):,}"


def available_columns(path):
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(p).schema.names)
        except Exception:
            return list(pd.read_parquet(p).head(0).columns)
    return list(pd.read_csv(p, nrows=0).columns)


def load_df(path, columns=None):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File non trovato: {p}")
    if columns is not None:
        cols_available = set(available_columns(p))
        columns = [c for c in columns if c in cols_available]
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p, columns=columns)
    return pd.read_csv(p, usecols=columns)


def find_file(base_dirs, stem, fmt="parquet"):
    for base in base_dirs:
        for ext in [fmt, "parquet", "csv"]:
            p = Path(base) / f"{stem}.{ext}"
            if p.exists():
                print(f"    Trovato: {p.name}")
                return p
    raise FileNotFoundError(f"'{stem}' non trovato in {[str(b) for b in base_dirs]}")


def required_project_columns(cont_covs, ndvi_year_cols, extra_cols=None):
    cols = ["WRB2_CODE", "lon", "lat"] + list(cont_covs) + list(ndvi_year_cols or [])
    cols += PIXEL_CELL_FIELDS
    if extra_cols:
        cols += list(extra_cols)
    return list(dict.fromkeys(cols))


def required_donor_columns(cont_covs, ndvi_year_cols, extra_cols=None):
    cols = ["WRB2_CODE", "lon", "lat"] + list(cont_covs) + list(ndvi_year_cols or [])
    cols += PIXEL_CELL_FIELDS
    if extra_cols:
        cols += list(extra_cols)
    return list(dict.fromkeys(cols))


# ================================================================
# PREP
# ================================================================

def wrb_to_texture(code):
    try:
        if pd.isna(code):
            return None
        return WRB_TO_TEXTURE.get(int(float(code)))
    except Exception:
        return None


def assign_texture(df, label="df", verbose=True):
    if "WRB2_CODE" not in df.columns:
        raise ValueError(f"{label}: WRB2_CODE mancante.")
    out = df.copy()
    before = len(out)
    tex = out["WRB2_CODE"].map(wrb_to_texture)
    out["texture_class"] = pd.Categorical(tex, categories=TEXTURE_ORDER)
    out = out[out["texture_class"].notna()].reset_index(drop=True)
    out["_texture_code"] = out["texture_class"].cat.codes.astype("int8")
    if verbose:
        print(f"    {label} WRB filter: {before:,} → {len(out):,}")
    if len(out) == 0:
        raise RuntimeError(f"{label}: vuoto dopo filtro WRB.")
    return out


def enforce_numeric_no_downcast(df, numeric_cols, label="df"):
    """Validate numeric columns without downcasting to float32/int32."""
    out = df.copy()
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def dropna_required(df, cols, label):
    before = len(df)
    mask = np.ones(before, dtype=bool)
    for c in cols:
        if c in df.columns:
            mask &= pd.notna(df[c]).to_numpy()
    out = df.loc[mask].reset_index(drop=True)
    print(f"    {label} dropna: {before:,} → {len(out):,}")
    if len(out) == 0:
        raise RuntimeError(f"{label}: vuoto dopo dropna.")
    return out


def check_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: colonne mancanti: {missing}")


def detect_ndvi_year_cols(columns):
    import re
    return sorted([c for c in columns if re.match(r"^NDVI_\d{4}$", str(c))])


# ================================================================
# DONOR STREAMING / PREFILTER
# ================================================================

def project_prefilter_thresholds(proj_df, cont_covs):
    thresholds = []
    for col in cont_covs:
        if col not in proj_df.columns:
            continue
        q01 = float(proj_df[col].quantile(0.01))
        q99 = float(proj_df[col].quantile(0.99))
        rng = q99 - q01
        if not np.isfinite(rng):
            continue
        buf = max(rng * 0.20, 1e-6)
        thresholds.append({"covariate": col, "lo": q01 - buf, "hi": q99 + buf})
    return thresholds


def apply_threshold_mask(df, thresholds):
    mask = np.ones(len(df), dtype=bool)
    for t in thresholds:
        c = t["covariate"]
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        mask &= np.isfinite(v) & (v >= float(t["lo"])) & (v <= float(t["hi"]))
    return mask


def _iter_table_batches(path, columns, batch_rows=DONOR_STREAM_BATCH_ROWS):
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise ImportError("pyarrow is required for streaming parquet donor loading.") from exc
        pf = pq.ParquetFile(p)
        cols = [c for c in columns if c in pf.schema.names]
        for batch in pf.iter_batches(batch_size=int(batch_rows), columns=cols):
            yield batch.to_pandas()
    else:
        usecols = [c for c in columns if c in available_columns(p)]
        for chunk in pd.read_csv(p, usecols=usecols, chunksize=int(batch_rows)):
            yield chunk


def load_donor_streamed_filtered_sample(donor_path, donor_cols, proj_df, cont_covs,
                                        max_n=N_DONOR_SAMPLE):
    if max_n is None:
        raise RuntimeError(
            "N_DONOR_SAMPLE=None non è ammesso nella modalità no-downcast: caricherebbe tutto il donor in RAM. "
            "Impostare N_DONOR_SAMPLE a un valore finito."
        )

    max_n = int(max_n)
    per_texture_cap = max(1, int(math.ceil(max_n / max(1, len(TEXTURE_ORDER)))))
    thresholds = project_prefilter_thresholds(proj_df, cont_covs)
    samples = {t: None for t in TEXTURE_ORDER}
    seen_after_filter = {t: 0 for t in TEXTURE_ORDER}
    seen_chunks = 0
    seen_raw = 0
    seen_prefilter = 0
    rng_seed = 42

    print("    Donor streaming mode: ON")
    print(f"    Batch rows: {DONOR_STREAM_BATCH_ROWS:,} | cap total: {max_n:,} | cap/texture: {per_texture_cap:,}")
    for t in thresholds:
        print(f"    Donor prefilter {t['covariate']:18s}: [{t['lo']:.3f}, {t['hi']:.3f}]")

    for chunk in _iter_table_batches(donor_path, donor_cols, DONOR_STREAM_BATCH_ROWS):
        seen_chunks += 1
        seen_raw += len(chunk)
        if len(chunk) == 0:
            continue
        chunk = enforce_numeric_no_downcast(chunk, cont_covs + ["WRB2_CODE", "lon", "lat"], "Donor chunk")
        chunk = assign_texture(chunk, "Donor chunk", verbose=False)
        chunk = dropna_required(chunk, cont_covs + ["WRB2_CODE", "lon", "lat"], "Donor chunk")
        if len(chunk) == 0:
            continue

        m = apply_threshold_mask(chunk, thresholds)
        chunk = chunk.loc[m].reset_index(drop=True)
        seen_prefilter += len(chunk)
        if len(chunk) == 0:
            del chunk, m
            continue

        for tex in TEXTURE_ORDER:
            sub = chunk[chunk["texture_class"].astype(str) == tex]
            if sub.empty:
                continue
            seen_after_filter[tex] += len(sub)
            old = samples.get(tex)
            combo = sub if old is None else pd.concat([old, sub], ignore_index=True)
            if len(combo) > per_texture_cap:
                combo = combo.sample(per_texture_cap, random_state=rng_seed + seen_chunks).reset_index(drop=True)
            samples[tex] = combo
        del chunk, m
        if seen_chunks % 5 == 0:
            kept = sum(0 if v is None else len(v) for v in samples.values())
            print(f"      chunks={seen_chunks:03d} | raw={seen_raw:,} | prefilter={seen_prefilter:,} | kept={kept:,}")
            gc.collect()

    parts = [v for v in samples.values() if v is not None and len(v) > 0]
    if not parts:
        raise RuntimeError("Donor streamed sample vuoto dopo WRB/dropna/prefilter. Allargare buffer donor o controllare covariate.")
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_n:
        out = out.sample(max_n, random_state=42).reset_index(drop=True)

    print(f"    Donor streaming summary: raw={seen_raw:,} | after prefilter={seen_prefilter:,} | kept={len(out):,}")
    print("    Donor kept by texture:")
    for tex, n in out["texture_class"].value_counts().sort_index().items():
        print(f"      {str(tex):12s}: {int(n):,}")
    return out.reset_index(drop=True)


def build_loose_prefilter_mask(proj_df, donor_df, cont_covs):
    thresholds = project_prefilter_thresholds(proj_df, cont_covs)
    mask = np.ones(len(donor_df), dtype=bool)
    for t in thresholds:
        col = t["covariate"]
        if col not in donor_df.columns:
            continue
        v = donor_df[col].to_numpy(dtype=np.float64, copy=False)
        before = int(mask.sum())
        mask &= np.isfinite(v) & (v >= float(t["lo"])) & (v <= float(t["hi"]))
        after = int(mask.sum())
        print(f"    Donor prefilter {col:18s}: [{t['lo']:.3f},{t['hi']:.3f}] {before:,}→{after:,}")
        if after == 0:
            raise RuntimeError(f"Donor vuoto dopo prefilter '{col}'.")
    return mask


# ================================================================
# PA COMMON SUPPORT FILTER
# ================================================================

def build_project_support_mask(proj_df, donor_df, cont_covs):
    mask = np.ones(len(proj_df), dtype=bool)
    details = []
    for col in cont_covs:
        if col not in proj_df.columns or col not in donor_df.columns:
            continue
        d_lo = float(donor_df[col].quantile(SUPPORT_QUANTILE_LO))
        d_hi = float(donor_df[col].quantile(SUPPORT_QUANTILE_HI))
        rng = d_hi - d_lo
        if not np.isfinite(rng) or rng < 1e-12:
            continue
        buf = max(rng * SUPPORT_BUFFER_FRAC, 1e-6)
        lo, hi = d_lo - buf, d_hi + buf
        v = proj_df[col].to_numpy(dtype=np.float64, copy=False)
        before = int(mask.sum())
        mask &= np.isfinite(v) & (v >= lo) & (v <= hi)
        after = int(mask.sum())
        excl = before - after
        details.append({
            "covariate": col, "donor_q_lo": round(d_lo, 6), "donor_q_hi": round(d_hi, 6),
            "buffer": round(buf, 6), "support_lo": round(lo, 6), "support_hi": round(hi, 6),
            "pa_before": before, "pa_after": after, "excluded": excl,
        })
        flag = "  *** WARN" if excl > before * 0.20 else ""
        print(f"    PA support {col:18s}: [{lo:.3f},{hi:.3f}] {before:,}→{after:,}  excl={excl:,}{flag}")
        if after == 0:
            raise RuntimeError(f"proj_df vuoto dopo support filter '{col}'. Verificare donor pool.")
    return mask, details


# ================================================================
# MISC / CONTEXT
# ================================================================

def stratified_cap_donor(donor_df, max_n=N_DONOR_SAMPLE, by="texture_class"):
    if max_n is None or len(donor_df) <= int(max_n):
        return donor_df.reset_index(drop=True)
    rng = np.random.default_rng(42)
    max_n = int(max_n)
    per_texture_cap = max(1, int(math.ceil(max_n / max(1, len(TEXTURE_ORDER)))))
    parts = []
    for tex in TEXTURE_ORDER:
        sub = donor_df[donor_df[by].astype(str) == tex]
        if sub.empty:
            continue
        n = min(len(sub), per_texture_cap)
        parts.append(sub.sample(n, random_state=42).reset_index(drop=True))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_n:
        out = out.sample(max_n, random_state=42).reset_index(drop=True)
    print(f"    Donor cap: {len(donor_df):,} → {len(out):,}")
    for cls, n in out[by].value_counts().sort_index().items():
        print(f"      {str(cls):12s}: {int(n):,}")
    return out.reset_index(drop=True)


def detect_tenure_col(proj_df, donor_df):
    for c in TENURE_COLUMN_CANDIDATES:
        if c in proj_df.columns and c in donor_df.columns:
            return c
    return None


def build_ctx(proj_df, tenure_col):
    ctx = {"tenure_col": tenure_col}
    if "SOC_g_kg" in proj_df.columns:
        ctx["soc_tol"] = max(abs(float(proj_df["SOC_g_kg"].mean())) * CALIPER_SOC_FRAC_OF_PROJECT_MEAN, 1e-9)
    if "NDVI_t0" in proj_df.columns:
        ctx["ndvi_tol"] = max(abs(float(proj_df["NDVI_t0"].mean())) * CALIPER_NDVI_T0_FRAC_OF_PROJECT_MEAN, 1e-9)
    return ctx


# ================================================================
# MAHALANOBIS
# ================================================================

def build_mahalanobis_inv_cov(z_proj, z_donor, n_cov):
    rng = np.random.default_rng(42)
    if len(z_donor) > int(COV_FIT_MAX_DONOR):
        z_d_fit = z_donor[rng.choice(len(z_donor), int(COV_FIT_MAX_DONOR), replace=False)]
    else:
        z_d_fit = z_donor
    z_all = np.vstack([z_proj, z_d_fit]).astype(np.float64, copy=False)
    lw = LedoitWolf(assume_centered=False)
    lw.fit(z_all)
    cov = lw.covariance_ + RIDGE_REG * np.eye(n_cov)
    return np.linalg.inv(cov) + RIDGE_REG * np.eye(n_cov)


def build_whitening_matrix(metric):
    eig = np.linalg.eigvalsh(metric)
    if eig.min() < 1e-10:
        metric = metric + (abs(eig.min()) + 1e-8) * np.eye(metric.shape[0])
    return scipy_cholesky(metric, lower=True)


# ================================================================
# CALIPERS / MATCH RECORDS
# ================================================================

def _build_donor_arrays(donor_df, tenure_col):
    arr = {
        "texture_code": donor_df["_texture_code"].to_numpy(dtype=np.int8, copy=False),
        "tenure": donor_df[tenure_col].astype(str).values if tenure_col and tenure_col in donor_df.columns else None,
    }
    for col in ["SOC_g_kg", "NDVI_t0", "elevation", "slope_deg", "dist_roads_km", "clay_pct", "precip_mm_yr"]:
        arr[col] = donor_df[col].to_numpy(dtype=np.float64, copy=False) if col in donor_df.columns else None
    return arr


def _check_calipers_vectorized(proj_row, cand_global_idx, donor_arr, ctx, tenure_col):
    n = len(cand_global_idx)
    passes = np.ones(n, dtype=bool)
    passes &= donor_arr["texture_code"][cand_global_idx] == int(proj_row.get("_texture_code", -99))
    if donor_arr["tenure"] is not None and tenure_col:
        passes &= donor_arr["tenure"][cand_global_idx] == str(proj_row.get(tenure_col, ""))
    for field, tol_key, fixed_tol in [
        ("SOC_g_kg", "soc_tol", None),
        ("NDVI_t0", "ndvi_tol", None),
        ("elevation", None, CALIPER_ELEVATION_M),
        ("slope_deg", None, CALIPER_SLOPE_DEG),
        ("dist_roads_km", None, CALIPER_ROADS_KM),
    ]:
        if donor_arr.get(field) is None:
            continue
        tol = ctx.get(tol_key, fixed_tol) if tol_key else fixed_tol
        if tol is None:
            continue
        pv = float(proj_row.get(field, np.nan))
        if np.isfinite(pv):
            passes &= np.abs(donor_arr[field][cand_global_idx] - pv) <= float(tol)
    return passes, int((~passes).sum())


def _sf(row, col, d=np.nan):
    if col not in row.index:
        return d
    try:
        v = row[col]
        return float(v) if pd.notna(v) else d
    except Exception:
        return d


def _si(row, col, d=-1):
    if col not in row.index:
        return d
    try:
        v = row[col]
        return int(v) if pd.notna(v) else d
    except Exception:
        return d


def _attach_pf(rec, row, prefix):
    text_fields = {"pixel_id", "source_tile"}
    int_fields = {"grid_row", "grid_col"}
    for f in PIXEL_CELL_FIELDS:
        if f not in row.index:
            continue
        key = f"{prefix}_{f}"
        if f in text_fields:
            v = row[f]
            rec[key] = "" if pd.isna(v) else str(v)
        elif f in int_fields:
            rec[key] = _si(row, f)
        else:
            rec[key] = _sf(row, f)
    if not rec.get(f"{prefix}_pixel_id") and f"{prefix}_grid_row" in rec and f"{prefix}_grid_col" in rec:
        rec[f"{prefix}_pixel_id"] = f"r{rec[f'{prefix}_grid_row']}::c{rec[f'{prefix}_grid_col']}"
    return rec


# ================================================================
# MATCHING CORE
# ================================================================

def run_matching(proj_df, donor_df, cont_covs, ndvi_year_cols, meta, ctx=None, donor_arr=None):
    check_cols(proj_df, cont_covs + ["lon", "lat", "texture_class", "_texture_code"], "project_df")
    check_cols(donor_df, cont_covs + ["lon", "lat", "texture_class", "_texture_code"], "donor_df")

    if ctx is None:
        tenure_col = detect_tenure_col(proj_df, donor_df)
        ctx = build_ctx(proj_df, tenure_col)
    else:
        tenure_col = ctx.get("tenure_col")
    if donor_arr is None:
        donor_arr = _build_donor_arrays(donor_df, tenure_col)

    print("    Hard calipers: texture=EXACT | "
          f"SOC=±{ctx.get('soc_tol', 'N/A')} | NDVI_t0=±{ctx.get('ndvi_tol', 'N/A')} | "
          f"elev=±{CALIPER_ELEVATION_M:.0f}m | slope=±{CALIPER_SLOPE_DEG:.0f}° | roads=±{CALIPER_ROADS_KM:.1f}km")

    rng = np.random.default_rng(42)
    Xp = proj_df[cont_covs].to_numpy(dtype=FLOAT_DTYPE, copy=True)
    Xd = donor_df[cont_covs].to_numpy(dtype=FLOAT_DTYPE, copy=True)
    if len(Xd) > int(SCALER_FIT_MAX_DONOR):
        idx = rng.choice(len(Xd), int(SCALER_FIT_MAX_DONOR), replace=False)
        X_fit = np.vstack([Xp, Xd[idx]])
    else:
        X_fit = np.vstack([Xp, Xd])
    scaler = StandardScaler()
    scaler.fit(X_fit)
    z_proj = scaler.transform(Xp).astype(FLOAT_DTYPE, copy=False)
    z_donor = scaler.transform(Xd).astype(FLOAT_DTYPE, copy=False)
    del Xp, Xd, X_fit
    gc.collect()

    metric = build_mahalanobis_inv_cov(z_proj, z_donor, len(cont_covs))
    L = build_whitening_matrix(metric).astype(FLOAT_DTYPE, copy=False)
    zp_w = (z_proj @ L).astype(FLOAT_DTYPE, copy=False)
    zd_w = (z_donor @ L).astype(FLOAT_DTYPE, copy=False)
    del z_proj, z_donor
    gc.collect()
    print(f"    Whitening OK | z_proj={zp_w.shape} | z_donor={zd_w.shape}")

    donor_df = donor_df.reset_index(drop=True)
    donor_reuse = np.zeros(len(donor_df), dtype=np.int16)
    avail_codes = set(donor_arr["texture_code"].astype(int).tolist())
    records, tex_rows, unmatched_rows = [], [], []
    run_id = meta.get("run_id", "")
    t0_total = time.time()

    for texture, proj_sub in proj_df.groupby("texture_class", sort=True, observed=True):
        if pd.isna(texture):
            continue
        proj_idx = proj_sub.index.to_numpy(dtype=np.int64)
        tex_code = TEXTURE_TO_CODE.get(str(texture), -99)

        if tex_code in avail_codes:
            allowed_codes, exact = [tex_code], True
        elif ALLOW_TEXTURE_FALLBACK:
            allowed_codes = [TEXTURE_TO_CODE[t] for t in TEXTURE_FALLBACK.get(str(texture), [])
                             if TEXTURE_TO_CODE.get(t) in avail_codes]
            exact = False
        else:
            allowed_codes, exact = [], False

        if not allowed_codes:
            print(f"    SKIP texture '{texture}': nessun donor.")
            for pi in proj_idx:
                unmatched_rows.append({"proj_idx": int(pi), "proj_texture": str(texture), "reason": "no_texture_donor"})
            continue

        d_global = np.where(np.isin(donor_arr["texture_code"], allowed_codes))[0]
        if not len(d_global):
            continue

        k_query = min(int(max(K_NEIGHBOURS, KNN_QUERY_CANDIDATES)), len(d_global))
        knn = NearestNeighbors(n_neighbors=k_query, metric="euclidean", algorithm="ball_tree",
                               leaf_size=40, n_jobs=KNN_N_JOBS)
        knn.fit(zd_w[d_global])

        n_matched = n_unmatched = rejected_total = 0
        dist_sum = dist_n = 0.0
        for start in range(0, len(proj_idx), int(KNN_BATCH_SIZE)):
            batch = proj_idx[start:start + int(KNN_BATCH_SIZE)]
            dists, loc_idx = knn.kneighbors(zp_w[batch])
            for row_i in range(len(batch)):
                pi = int(batch[row_i])
                pr = proj_df.loc[pi]
                cand_global = d_global[loc_idx[row_i]]
                dist_sum += float(dists[row_i, 0])
                dist_n += 1
                passes, n_rej = _check_calipers_vectorized(pr, cand_global, donor_arr, ctx, tenure_col)
                rejected_total += n_rej
                passing_local = np.where(passes)[0]
                if not len(passing_local):
                    n_unmatched += 1
                    unmatched_rows.append({
                        "proj_idx": pi,
                        "proj_lon": _sf(pr, "lon"), "proj_lat": _sf(pr, "lat"),
                        "proj_grid_row": _si(pr, "grid_row"), "proj_grid_col": _si(pr, "grid_col"),
                        "proj_texture": str(texture),
                        "reason": "no_candidate_passed_hard_calipers", "queried_n": int(k_query),
                    })
                    continue

                chosen_local = None
                for loc in passing_local:
                    gi = int(cand_global[int(loc)])
                    if donor_reuse[gi] < int(MAX_DONOR_REUSE):
                        chosen_local = int(loc)
                        break
                if chosen_local is None and not ALLOW_DONOR_REUSE_OVERFLOW:
                    n_unmatched += 1
                    unmatched_rows.append({
                        "proj_idx": pi,
                        "proj_lon": _sf(pr, "lon"), "proj_lat": _sf(pr, "lat"),
                        "proj_grid_row": _si(pr, "grid_row"), "proj_grid_col": _si(pr, "grid_col"),
                        "proj_texture": str(texture),
                        "reason": "all_passing_candidates_reuse_saturated",
                        "queried_n": int(k_query), "passing_candidates_n": int(len(passing_local)),
                    })
                    continue
                if chosen_local is None:
                    chosen_local = int(passing_local[0])

                best_global = int(cand_global[chosen_local])
                best_dist = float(dists[row_i, chosen_local])
                reuse = int(donor_reuse[best_global])
                donor_reuse[best_global] += 1
                dr = donor_df.iloc[best_global]

                rec = {
                    "run_id": run_id,
                    "proj_idx": pi,
                    "ref_idx": best_global,
                    "ref_lon": _sf(dr, "lon"), "ref_lat": _sf(dr, "lat"),
                    "proj_lon": _sf(pr, "lon"), "proj_lat": _sf(pr, "lat"),
                    "match_distance": best_dist,
                    "match_rank": int(chosen_local + 1),
                    "proj_texture": str(texture), "ref_texture": str(dr.get("texture_class", "")),
                    "texture_exact": bool(exact),
                    "reuse_exceeded": bool(reuse >= MAX_DONOR_REUSE),
                    "all_calipers_passed": True,
                    "queried_n": int(k_query),
                    "passing_candidates_n": int(len(passing_local)),
                    "donor_reuse_before": int(reuse),
                    "hard_caliper_rejected_total": int(n_rej),
                }
                # Reference values kept under original covariate names for Step 03 compatibility.
                for c in cont_covs + list(ndvi_year_cols or []):
                    if c in dr.index:
                        rec[c] = _sf(dr, c)
                    if c in pr.index:
                        rec[f"proj_{c}"] = _sf(pr, c)
                for c in cont_covs + list(ndvi_year_cols or []):
                    if c in pr.index and c in dr.index:
                        rec[f"diff_{c}"] = _sf(dr, c) - _sf(pr, c)
                _attach_pf(rec, dr, "ref")
                _attach_pf(rec, pr, "proj")
                records.append(rec)
                n_matched += 1

            del dists, loc_idx
            if (start // int(KNN_BATCH_SIZE)) % 10 == 0:
                gc.collect()

        tex_rows.append({
            "texture": str(texture), "proj_n": int(len(proj_idx)), "donor_n": int(len(d_global)),
            "k_query": int(k_query), "exact_texture": bool(exact),
            "matched_n": int(n_matched), "unmatched_n": int(n_unmatched),
            "hard_caliper_rejected_total": int(rejected_total),
            "mean_nn_dist_unfiltered": float(dist_sum / dist_n) if dist_n else np.nan,
        })
        print(f"    Texture {str(texture):10s}: proj={len(proj_idx):,} donor={len(d_global):,} "
              f"matched={n_matched:,} unmatched={n_unmatched:,} k={k_query} [{time.time()-t0_total:.0f}s]")
        del knn
        gc.collect()

    matched = pd.DataFrame(records)
    unmatched = pd.DataFrame(unmatched_rows)
    if not matched.empty:
        matched = matched.sort_values("proj_idx").reset_index(drop=True)
    return matched, pd.DataFrame(tex_rows), unmatched, metric


# ================================================================
# VALIDATION / PLOTS
# ================================================================

def _project_values_for_smd(proj_df, matched_df, cov):
    # Preferred: exact project value stored in matched_df.
    pc = f"proj_{cov}"
    if pc in matched_df.columns:
        return pd.to_numeric(matched_df[pc], errors="coerce")
    # Fallback: reconstruct from reference minus difference.
    dc = f"diff_{cov}"
    if cov in matched_df.columns and dc in matched_df.columns:
        return pd.to_numeric(matched_df[cov], errors="coerce") - pd.to_numeric(matched_df[dc], errors="coerce")
    # Last fallback: index lookup.
    if "proj_idx" in matched_df.columns and cov in proj_df.columns:
        idx = pd.to_numeric(matched_df["proj_idx"], errors="coerce").dropna().astype(int)
        idx = idx[(idx >= 0) & (idx < len(proj_df))]
        return pd.to_numeric(proj_df.loc[idx, cov], errors="coerce")
    return pd.Series(dtype=float)


def compute_smd(proj_df, ref_df, covs):
    """
    Computes SMD on matched project pixels vs matched reference pixels.
    This is the correct balance check for the selected synthetic-control cohort.
    """
    rows = []
    for cov in covs:
        if cov not in ref_df.columns:
            continue
        p = _project_values_for_smd(proj_df, ref_df, cov).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        r = pd.to_numeric(ref_df[cov], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        n = min(len(p), len(r))
        if n == 0:
            smd = np.nan
            rows.append({"covariate": cov, "project_mean": np.nan, "reference_mean": np.nan,
                         "pooled_sd": np.nan, "SMD": np.nan, "n_project_matched": 0,
                         "n_reference_matched": int(len(r)), "passed": False})
            continue
        p = p.iloc[:n]
        r = r.iloc[:n]
        pooled = np.sqrt((p.var(ddof=1) + r.var(ddof=1)) / 2.0) if n > 1 else np.nan
        smd = abs(p.mean() - r.mean()) / pooled if pooled and np.isfinite(pooled) and pooled > 0 else 0.0
        rows.append({
            "covariate": cov,
            "project_mean": float(p.mean()),
            "reference_mean": float(r.mean()),
            "pooled_sd": float(pooled) if np.isfinite(pooled) else np.nan,
            "SMD": float(smd),
            "n_project_matched": int(len(p)),
            "n_reference_matched": int(len(r)),
            "passed": bool(np.isfinite(smd) and smd < SMD_THRESHOLD),
        })
    if not rows:
        return pd.DataFrame(columns=["project_mean", "reference_mean", "pooled_sd", "SMD", "passed"])
    return pd.DataFrame(rows).set_index("covariate")


def compute_caliper_audit(matched_df):
    rows = []
    specs = [
        ("SOC_g_kg", "diff_SOC_g_kg"),
        ("NDVI_t0", "diff_NDVI_t0"),
        ("elevation", "diff_elevation"),
        ("slope_deg", "diff_slope_deg"),
        ("dist_roads_km", "diff_dist_roads_km"),
    ]
    for name, col in specs:
        if col not in matched_df.columns:
            continue
        d = matched_df[col].astype(float).abs().replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"caliper": name, "n": int(len(d)), "mean_abs_diff": float(d.mean()),
                     "p95_abs_diff": float(d.quantile(0.95)), "max_abs_diff": float(d.max())})
    return pd.DataFrame(rows)


def plot_smd(smd_df, out_dir=None, filename="SMD_lollipop.png", title="Covariate balance"):
    fig, ax = plt.subplots(figsize=(9, max(4, len(smd_df) * 0.35)))
    if smd_df.empty:
        ax.text(0.5, 0.5, "No SMD data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        colors = ["#d62728" if not bool(p) else "#2ca02c" for p in smd_df["passed"]]
        ax.hlines(smd_df.index, 0, smd_df["SMD"].fillna(0).values, lw=2, color=colors)
        ax.scatter(smd_df["SMD"].fillna(0).values, smd_df.index, s=70, color=colors, zorder=5)
        ax.axvline(SMD_THRESHOLD, color="black", ls="--", label=f"SMD={SMD_THRESHOLD}")
        ax.set(xlabel="SMD on matched project subset vs matched reference", title=title)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / filename, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_match_distances(matched_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    d = matched_df["match_distance"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
    if len(d) == 0:
        ax.text(0.5, 0.5, "No match distances", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        ax.hist(d, bins=60, edgecolor="white")
        ax.axvline(d.mean(), ls="--", label=f"Mean={d.mean():.3f}")
        ax.axvline(np.percentile(d, 95), ls=":", label=f"P95={np.percentile(d, 95):.3f}")
        ax.set(xlabel="Whitened Mahalanobis distance", ylabel="Pixels", title="Match-distance distribution")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "match_distance_distribution.png", dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_mahalanobis_metric(metric, cont_covs, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = float(np.abs(metric).max()) if np.size(metric) else 1.0
    im = ax.imshow(metric, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cont_covs)))
    ax.set_yticks(range(len(cont_covs)))
    ax.set_xticklabels(cont_covs, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cont_covs, fontsize=9)
    plt.colorbar(im, ax=ax, label="Sigma^-1 entry")
    ax.set_title("Plain Mahalanobis inverse covariance (no RF weights)")
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "mahalanobis_inv_cov.png", dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_support_overlap(proj_df_tagged, donor_df, cont_covs, support_details, out_dir=None):
    n_cols = min(4, max(1, len(cont_covs)))
    n_rows = max(1, (len(cont_covs) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
    axes = np.array(axes).flatten()
    detail_map = {d["covariate"]: d for d in support_details}
    elig_mask = proj_df_tagged["_in_support"].to_numpy(dtype=bool)
    for i, cov in enumerate(cont_covs):
        ax = axes[i]
        if cov not in proj_df_tagged.columns or cov not in donor_df.columns:
            ax.axis("off")
            continue
        dv = pd.to_numeric(donor_df[cov], errors="coerce").dropna().astype(float).values
        pv_elig = pd.to_numeric(proj_df_tagged.loc[elig_mask, cov], errors="coerce").dropna().astype(float).values
        pv_excl = pd.to_numeric(proj_df_tagged.loc[~elig_mask, cov], errors="coerce").dropna().astype(float).values
        all_v = np.concatenate([dv, pv_elig, pv_excl]) if len(dv) + len(pv_elig) + len(pv_excl) else np.array([])
        if not len(all_v):
            ax.axis("off")
            continue
        lo = float(np.nanpercentile(all_v, 1))
        hi = float(np.nanpercentile(all_v, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            ax.axis("off")
            continue
        bins = np.linspace(lo, hi, 40)
        ax.hist(dv, bins=bins, alpha=0.45, density=True, label=f"Donor sample ({len(dv):,})")
        if len(pv_elig):
            ax.hist(pv_elig, bins=bins, alpha=0.65, density=True, label=f"PA elig ({len(pv_elig):,})")
        if len(pv_excl):
            ax.hist(pv_excl, bins=bins, alpha=0.70, density=True, label=f"PA OOS ({len(pv_excl):,})")
        d = detail_map.get(cov)
        if d:
            ax.axvline(d["support_lo"], lw=1.4, ls="--")
            ax.axvline(d["support_hi"], lw=1.4, ls="--")
        ax.set_title(cov, fontsize=9, fontweight="bold")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for j in range(len(cont_covs), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Common support diagnostics — donor streamed sample vs PA", fontsize=10)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "support_overlap_diagnostics.png", dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# ================================================================
# OPTIONAL FEASIBILITY SCREEN
# ================================================================

def fast_caliper_feasibility_screen(proj_df, donor_arr, ctx, tenure_col):
    n = len(proj_df)
    feasible_mask = np.zeros(n, dtype=bool)
    infeasible_reason = np.full(n, "not_checked", dtype=object)
    B = int(FEASIBILITY_BATCH_SIZE)
    for tex_code in sorted(proj_df["_texture_code"].unique()):
        if tex_code < 0:
            continue
        pa_pos = np.where(proj_df["_texture_code"].to_numpy() == tex_code)[0]
        d_pos = np.where(donor_arr["texture_code"] == tex_code)[0]
        if len(d_pos) == 0:
            infeasible_reason[pa_pos] = "no_texture_donor"
            continue
        # This is diagnostic only. Keep B small because it creates B x n_donor boolean matrices.
        for start in range(0, len(pa_pos), B):
            batch = pa_pos[start:start+B]
            passes = np.ones((len(batch), len(d_pos)), dtype=bool)
            for field, tol_key, fixed_tol in [
                ("SOC_g_kg", "soc_tol", None),
                ("NDVI_t0", "ndvi_tol", None),
                ("elevation", None, CALIPER_ELEVATION_M),
                ("slope_deg", None, CALIPER_SLOPE_DEG),
                ("dist_roads_km", None, CALIPER_ROADS_KM),
            ]:
                if donor_arr.get(field) is None or field not in proj_df.columns:
                    continue
                tol = ctx.get(tol_key, fixed_tol) if tol_key else fixed_tol
                if tol is None:
                    continue
                pa_vals = proj_df.iloc[batch][field].to_numpy(dtype=np.float64)
                d_vals = donor_arr[field][d_pos]
                passes &= np.abs(pa_vals[:, None] - d_vals[None, :]) <= float(tol)
            has_match = passes.any(axis=1)
            feasible_mask[batch] = has_match
            infeasible_reason[batch] = np.where(has_match, "feasible", "no_donor_passes_all_calipers")
            del passes
    return feasible_mask, infeasible_reason


# ================================================================
# MAIN
# ================================================================

def run_matching_step(base_dirs=None, output_dir=None, proj_df=None, donor_df=None, meta=None, verbose=True):
    if base_dirs is None:
        raise RuntimeError("base_dirs non fornito.")
    base_dirs = [Path(b) for b in base_dirs]
    out_dir = Path(output_dir) if output_dir else base_dirs[0].parent / "02_matching"
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta is None:
        rp = base_dirs[0] / "extraction_report.json"
        meta = json.load(open(rp, encoding="utf-8")) if rp.exists() else {}

    cont_covs = [c for c in (meta.get("continuous_covariates") or [])
                 if c not in ("pixel_area_ha", "ndvi_valid_years") and not str(c).startswith("precip_bin")]
    if not cont_covs:
        raise RuntimeError("continuous_covariates vuoto.")
    ndvi_year_cols = meta.get("ndvi_year_cols") or []

    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 02 | Plain Mahalanobis | K={K_NEIGHBOURS} | candidates={KNN_QUERY_CANDIDATES}")
        print(f"NO DOWNCAST | donor streamed={donor_df is None} | donor cap={format_int_or_all(N_DONOR_SAMPLE)}")
        print(f"Output: {out_dir}\n{'='*60}")

    project_cols = required_project_columns(cont_covs, ndvi_year_cols)
    donor_cols = required_donor_columns(cont_covs, ndvi_year_cols)

    if proj_df is None:
        proj_df = load_df(find_file(base_dirs, "project_pixels_raw"), columns=project_cols)
    else:
        proj_df = proj_df[[c for c in project_cols if c in proj_df.columns]].copy()

    check_cols(proj_df, cont_covs + ["WRB2_CODE", "lon", "lat"], "project_df")

    print("\n[1] Project texture + numeric validation (no downcast)")
    proj_df = assign_texture(proj_df, "Project")
    proj_df = enforce_numeric_no_downcast(proj_df, cont_covs + ["WRB2_CODE", "lon", "lat"], "Project")
    proj_df = dropna_required(proj_df, cont_covs + ["WRB2_CODE", "lon", "lat"], "Project")
    print(f"    Project: {len(proj_df):,}")

    if donor_df is None:
        print("\n[2] Donor streaming + loose prefilter + stratified cap")
        donor_path = find_file(base_dirs, "donor_pixels_raw")
        donor_df = load_donor_streamed_filtered_sample(donor_path, donor_cols, proj_df, cont_covs, N_DONOR_SAMPLE)
    else:
        print("\n[2] Donor in-memory texture + numeric validation + loose prefilter")
        donor_df = donor_df[[c for c in donor_cols if c in donor_df.columns]].copy()
        check_cols(donor_df, cont_covs + ["WRB2_CODE", "lon", "lat"], "donor_df")
        donor_df = assign_texture(donor_df, "Donor")
        donor_df = enforce_numeric_no_downcast(donor_df, cont_covs + ["WRB2_CODE", "lon", "lat"], "Donor")
        donor_df = dropna_required(donor_df, cont_covs + ["WRB2_CODE", "lon", "lat"], "Donor")
        dmask = build_loose_prefilter_mask(proj_df, donor_df, cont_covs)
        donor_df = donor_df.loc[dmask].reset_index(drop=True)
        del dmask
        if N_DONOR_SAMPLE is not None and len(donor_df) > int(N_DONOR_SAMPLE):
            donor_df = stratified_cap_donor(donor_df, N_DONOR_SAMPLE, "texture_class")
        gc.collect()
    print(f"    Donor retained for KNN: {len(donor_df):,}")

    print("\n[2b] PA common support filter")
    n_pa_total = len(proj_df)
    support_mask, support_details = build_project_support_mask(proj_df, donor_df, cont_covs)
    proj_df["_in_support"] = support_mask
    n_ineligible = int((~support_mask).sum())
    if n_ineligible > 0:
        proj_df.loc[~support_mask].drop(columns=["_in_support"], errors="ignore").to_csv(
            out_dir / "ineligible_project_pixels_out_of_support.csv", index=False)
        print(f"    → ineligible_project_pixels_out_of_support.csv ({n_ineligible:,} px)")
    fig_support = plot_support_overlap(proj_df, donor_df, cont_covs, support_details, out_dir=out_dir)
    if support_details:
        pd.DataFrame(support_details).to_csv(out_dir / "support_filter_audit.csv", index=False)
    proj_df = proj_df.loc[support_mask].drop(columns=["_in_support"]).reset_index(drop=True)
    del support_mask
    gc.collect()
    n_eligible = len(proj_df)
    ineligible_frac = n_ineligible / max(1, n_pa_total)
    print(f"    PA total / eligible / OOS: {n_pa_total:,} / {n_eligible:,} / {n_ineligible:,}")

    print("\n[2c] Context + donor arrays")
    tenure_col = detect_tenure_col(proj_df, donor_df)
    ctx = build_ctx(proj_df, tenure_col)
    donor_arr = _build_donor_arrays(donor_df, tenure_col)

    n_caliper_infeasible = 0
    if RUN_CALIPER_FEASIBILITY_SCREEN:
        print("\n[2d] Caliper feasibility screen")
        feasible_mask, infeasible_reasons = fast_caliper_feasibility_screen(proj_df, donor_arr, ctx, tenure_col)
        n_caliper_infeasible = int((~feasible_mask).sum())
        if n_caliper_infeasible > 0:
            infeas_df = proj_df.loc[~feasible_mask].copy()
            infeas_df["infeasibility_reason"] = infeasible_reasons[~feasible_mask]
            infeas_df.to_csv(out_dir / "ineligible_project_pixels_caliper_infeasible.csv", index=False)
        del feasible_mask, infeasible_reasons
        gc.collect()

    print("\n[3] Matching KNN + hard calipers + plain Mahalanobis")
    matched_df, tex_summary, unmatched_df, metric = run_matching(
        proj_df, donor_df, cont_covs, ndvi_year_cols, meta, ctx=ctx, donor_arr=donor_arr)
    del donor_df, donor_arr
    gc.collect()

    if matched_df.empty:
        raise RuntimeError("Nessun pixel matchato. Aumentare KNN_QUERY_CANDIDATES o N_DONOR_SAMPLE.")

    n_matched = int(len(matched_df))
    n_unique = int(matched_df[["ref_lon", "ref_lat"]].drop_duplicates().shape[0])
    matched_project_n = int(matched_df["proj_idx"].nunique()) if "proj_idx" in matched_df.columns else 0
    unmatched_n = int(unmatched_df["proj_idx"].nunique()) if (not unmatched_df.empty and "proj_idx" in unmatched_df.columns) else 0
    cov_elig = float(matched_project_n / n_eligible) if n_eligible > 0 else 0.0
    cov_total = float(matched_project_n / n_pa_total) if n_pa_total > 0 else 0.0
    reuse_n = int(matched_df.get("reuse_exceeded", pd.Series(dtype=bool)).sum()) if "reuse_exceeded" in matched_df.columns else 0

    print(f"\n    Matched/eligible : {matched_project_n:,}/{n_eligible:,} ({cov_elig*100:.2f}%)")
    print(f"    Matched/total PA : {matched_project_n:,}/{n_pa_total:,} ({cov_total*100:.2f}%)")
    print(f"    Unique ref       : {n_unique:,} | Unmatched: {unmatched_n:,}")

    print(f"\n[4] SMD on matched project subset vs matched references (< {SMD_THRESHOLD})")
    smd_df = compute_smd(proj_df, matched_df, cont_covs)
    ndvi_smd_df = compute_smd(proj_df, matched_df, ndvi_year_cols) if ndvi_year_cols else pd.DataFrame()
    for cov, row in smd_df.iterrows():
        print(f"    {'PASS' if row['passed'] else 'FAIL':4s} {cov:20s}: SMD={row['SMD']:.4f}")
    if not ndvi_smd_df.empty:
        print("    Annual NDVI balance:")
        for cov, row in ndvi_smd_df.iterrows():
            print(f"    {'PASS' if row['passed'] else 'FAIL':4s} {cov:20s}: SMD={row['SMD']:.4f}")

    caliper_audit = compute_caliper_audit(matched_df)
    metric_df = pd.DataFrame(metric, index=cont_covs, columns=cont_covs)

    print("\n[5] Plot + export")
    fig_m = plot_mahalanobis_metric(metric, cont_covs, out_dir)
    fig_s = plot_smd(smd_df, out_dir, "SMD_lollipop.png", "Mandatory covariate balance")
    fig_d = plot_match_distances(matched_df, out_dir)
    fig_n = plot_smd(ndvi_smd_df, out_dir, "SMD_annual_NDVI.png", "Annual NDVI balance") if not ndvi_smd_df.empty else None

    matched_df.to_parquet(out_dir / "reference_area_pixels.parquet", index=False)
    if WRITE_FULL_MATCH_CSV:
        matched_df.to_csv(out_dir / "reference_area_pixels.csv", index=False)
    else:
        matched_df.head(int(MATCH_PREVIEW_CSV_ROWS)).to_csv(out_dir / "reference_area_pixels_preview.csv", index=False)
    unmatched_df.to_csv(out_dir / "unmatched_project_pixels.csv", index=False)
    smd_df.to_csv(out_dir / "SMD_table.csv")
    if not ndvi_smd_df.empty:
        ndvi_smd_df.to_csv(out_dir / "SMD_annual_NDVI_table.csv")
    tex_summary.to_csv(out_dir / "texture_summary.csv", index=False)
    caliper_audit.to_csv(out_dir / "caliper_audit.csv", index=False)
    metric_df.to_csv(out_dir / "mahalanobis_inv_cov.csv")

    summary = {
        "run_id": meta.get("run_id", ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "matching_method": "Streaming hard-caliper plain Mahalanobis KNN; no RF weights; no downcast",
        "k_neighbours": int(K_NEIGHBOURS),
        "knn_query_candidates": int(KNN_QUERY_CANDIDATES),
        "n_donor_sample": None if N_DONOR_SAMPLE is None else int(N_DONOR_SAMPLE),
        "donor_stream_batch_rows": int(DONOR_STREAM_BATCH_ROWS),
        "pa_total_n": int(n_pa_total),
        "pa_eligible_n": int(n_eligible),
        "pa_ineligible_out_of_support_n": int(n_ineligible),
        "pa_ineligible_caliper_infeasible_n": int(n_caliper_infeasible),
        "ineligible_out_of_support_frac": round(ineligible_frac, 6),
        "support_quantile_lo": float(SUPPORT_QUANTILE_LO),
        "support_quantile_hi": float(SUPPORT_QUANTILE_HI),
        "support_buffer_frac": float(SUPPORT_BUFFER_FRAC),
        "matched_rows_n": int(n_matched),
        "matched_project_n": int(matched_project_n),
        "unmatched_project_n": int(unmatched_n),
        "unique_donor_n": int(n_unique),
        "reuse_exceeded_n": int(reuse_n),
        "project_match_coverage_of_eligible": round(cov_elig, 6),
        "project_match_coverage_of_total": round(cov_total, 6),
        "coverage_denominator": "eligible_pa_pixels_inside_donor_support",
        "smd_method": "matched project subset vs matched reference subset",
        "smd_pass_all": bool(smd_df["passed"].all()),
        "ndvi_smd_pass_all": bool(ndvi_smd_df["passed"].all()) if not ndvi_smd_df.empty else None,
        "next_step": "03_STARR_twin_test_selection.py",
    }
    with open(out_dir / "matching_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if STRICT_COMPLIANCE_STOP:
        errors = []
        if not bool(smd_df["passed"].all()):
            errors.append("mandatory covariate SMD failed")
        if cov_elig < float(MIN_PROJECT_MATCH_COVERAGE):
            errors.append(f"coverage of eligible {cov_elig:.3f} < {MIN_PROJECT_MATCH_COVERAGE:.3f}")
        if reuse_n > 0 and not ALLOW_DONOR_REUSE_OVERFLOW:
            errors.append("donor reuse exceeded")
        if errors:
            raise RuntimeError("STEP 02 non compliant: " + "; ".join(errors) + f". Diagnostics in {out_dir}")

    figs = {"mahalanobis_metric": fig_m, "smd": fig_s, "distances": fig_d,
            "ndvi_smd": fig_n, "support_overlap": fig_support}
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 02 COMPLETE")
        print(f"PA total/eligible/OOS : {n_pa_total:,}/{n_eligible:,}/{n_ineligible:,}")
        print(f"Matched/eligible      : {matched_project_n:,}/{n_eligible:,} ({cov_elig*100:.2f}%)")
        print(f"Unique donor          : {n_unique:,} | SMD pass: {bool(smd_df['passed'].all())}")
        print(f"Output                : {out_dir}\n{'='*60}")
    return matched_df, None, metric_df, smd_df, figs, out_dir


def main():
    run_matching_step()


if __name__ == "__main__":
    main()
