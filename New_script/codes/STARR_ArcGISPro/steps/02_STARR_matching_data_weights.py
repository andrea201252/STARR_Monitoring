# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB  |  02_STARR_matching_data_weights.py
STEP 02 — Hard-Caliper + Weighted Mahalanobis KNN Matching
============================================================

MANUAL PARAMETERS:
  K_NEIGHBOURS          number of final selected KNN neighbours
  KNN_QUERY_CANDIDATES  candidates searched before the calipers (≥ K_NEIGHBOURS)
  N_DONOR_SAMPLE        donor sampling; if None=all
  ALLOW_TEXTURE_FALLBACK
  MAX_DONOR_REUSE

WHAT'S NEW in v04 compared to the previous version:
  - VECTORIZED HARD CALIPERS: the caliper check no longer uses a Python loop
    per pixel per candidate, but pre-extracts numpy arrays from donor_df
    and performs the comparisons in numpy → 10-50x faster.
  - Removed the use of pandas iloc for each candidate (it was the bottleneck).
  - KNN logic adapted from the QGIS PlotMatcherAlgorithm: uses the same
    Mahalanobis metric (VI = inv(Σ_LedoitWolf)) but with Cholesky whitening + ball_tree
    instead of brute-force cdist, which does not scale beyond 50k donor pixels.

Hard calipers (GS STARR Annex 1 Table A.3):
  SOC_g_kg       ±10% of the project mean
  NDVI_t0        ±10% of the project mean
  elevation      ±200 m
  slope_deg      ±10°
  dist_roads_km  ±1 km
  texture_class  exact match (WRB2 → soil texture class)
  tenure_col     exact match if present
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy.linalg import cholesky as scipy_cholesky
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import LedoitWolf
except ImportError:
    # ArcGIS Pro Python without scikit-learn: numpy/scipy drop-ins that reproduce
    # StandardScaler / LedoitWolf / NearestNeighbors identically (validated against
    # scikit-learn). Requires the package folder (steps/) on sys.path — the
    # toolbox adds it before loading this module.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _sklearn_fallback import NearestNeighbors, StandardScaler, LedoitWolf


def _is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except ImportError:
        return False


if not _is_notebook():
    matplotlib.use("Agg")


# ── MANUAL PARAMETERS ────────────────────────────────────────────────
#
# ALL editable from the notebook §1. They are read from environment variables
# so the value SURVIVES the internal module reloads done by 00b/00 (setting
# s02.K_NEIGHBOURS = ... as a module attribute does NOT reach the fresh copy
# that 00b re-imports; an env var does). The notebook sets:
#   os.environ["STARR_K_NEIGHBOURS"], ["STARR_KNN_QUERY_CANDIDATES"],
#   ["STARR_N_DONOR_SAMPLE"], ["STARR_KNN_N_JOBS"].
def _env_int(name, default):
    v = os.environ.get(name)
    try:
        return int(float(v)) if v not in (None, "") else int(default)
    except Exception:
        return int(default)

def _env_int_or_none(name, default):
    """Returns None when the env var is 'none'/'all'/'0'/'' (→ use all donors)."""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if str(v).strip().lower() in ("none", "all", "0"):
        return None
    try:
        return int(float(v))
    except Exception:
        return default

K_NEIGHBOURS          = _env_int("STARR_K_NEIGHBOURS", 1)
KNN_QUERY_CANDIDATES  = _env_int("STARR_KNN_QUERY_CANDIDATES", 150)   # candidates per pixel; adaptive to the pool size
N_DONOR_SAMPLE        = _env_int_or_none("STARR_N_DONOR_SAMPLE", 300_000)
ALLOW_TEXTURE_FALLBACK = os.environ.get("STARR_ALLOW_TEXTURE_FALLBACK", "0").strip().lower() in ("1", "true", "yes")
MAX_DONOR_REUSE       = _env_int("STARR_MAX_DONOR_REUSE", 1)

# Covariates to EXCLUDE from VALUE-based matching (prefilter + Mahalanobis + SMD).
#
# precip_mm_yr: in GS Annex 1 Table A.3 Precipitation (MAP) is mandatory but
# its tolerance is "Same Isohyet / Ecoregion" — i.e. a CATEGORICAL zone
# constraint, NOT a value caliper (±mm). It must therefore be satisfied upstream,
# not as a continuous covariate: the donor pool is already clipped to the
# project's ECOREGION in GEE (rawDonorSearchGeom = buffer ∩ ecoregion), so the
# "same ecoregion" requirement is met by construction.
# Using it as a continuous value was wrong (and it decimated the pool: e.g. Muraca_Caia
# 938k → 79k, -92%). If the "same isohyet" stringency is needed, add a
# categorical match on rainfall bands (isohyets), not a tight caliper.
# Set [] to (re)use all value-based covariates.
EXCLUDE_COVARIATES_FROM_MATCHING = ["precip_mm_yr"]

# Output subfolder of this step — EDITABLE FROM THE NOTEBOOK (s02.STEP_DIRNAME).
# The OUTPUTS_DIRNAME/<RUN_ID> level is inherited from the Step 01 path
# (out_dir = base_dirs[0].parent / STEP_DIRNAME).
STEP_DIRNAME = "02_matching"

# ── BATCH KNN (RAM FIX) ───────────────────────────────────────────────
# zp_w is NEVER precomputed for all project pixels.
# Scaling+whitening happens on-the-fly for each batch.
# Memory per batch: KNN_BATCH_SIZE × n_cov × 8 bytes (e.g. 4096 × 20 × 8 = 640 KB)
# instead of 106K × n_cov × 8 = 17 MB kept permanently in RAM.
KNN_BATCH_SIZE        = 4096  # project pixels per KNN query (whitened on-the-fly)
KNN_N_JOBS            = _env_int("STARR_KNN_N_JOBS", -1)
KNN_LEAF_SIZE         = 60
SCALER_FIT_MAX_ROWS   = 60_000   # sample for the StandardScaler fit (proj+donor mix)
COV_FIT_MAX_ROWS      = 40_000   # sample for LedoitWolf (proj+donor mix)

# ── FIXED (GS STARR Annex 1 Table A.3) ───────────────────────────────

SMD_THRESHOLD                     = 0.1
RIDGE_REG                         = 1e-6
CALIPER_SOC_FRAC_OF_PROJECT_MEAN  = 0.10
CALIPER_NDVI_T0_FRAC_OF_PROJECT_MEAN = 0.10
CALIPER_ELEVATION_M               = 200.0   # methodological maximum (GS Table A.3)
CALIPER_SLOPE_DEG                 = 10.0
CALIPER_ROADS_KM                  = 1.0

# ADAPTIVE elevation caliper. On an almost flat PA (little elevation
# variability) the ±200 m does not constrain anything → elevation stays unbalanced
# (high SMD). If active, the effective caliper =
#   clip(CALIPER_ELEVATION_SD_MULT × sd_elevation_PA,
#        CALIPER_ELEVATION_MIN_M, CALIPER_ELEVATION_M)
# → tighter (more conservative) for flat PAs, unchanged (=200 m) for PAs
# with strong relief. Tightening is always allowed (more conservative than the GS max).
CALIPER_ELEVATION_ADAPTIVE        = False   # False → fixed caliper ±CALIPER_ELEVATION_M (200 m, GS Table A.3 value)
CALIPER_ELEVATION_SD_MULT         = 4.0
CALIPER_ELEVATION_MIN_M           = 5.0

# Removal of rows with physically invalid / nodata covariates before
# matching: SOC_g_kg <= MIN_VALID_SOC (0 = nodata HWSD/SoilGrids; SOC here
# is in t/ha) and NDVI_t0 outside [-1, 1]. Protects the pool from spurious values
# (e.g. SOC=0 that was entering the donor).
DROP_INVALID_COVARIATES           = os.environ.get("STARR_DROP_INVALID_COVARIATES", "1").strip().lower() in ("1", "true", "yes")
MIN_VALID_SOC                     = float(os.environ.get("STARR_MIN_VALID_SOC", "0.0") or 0.0)

# ── MANDATORY HARD CALIPERS (GS STARR Annex 1 Table A.3) ─────────────
# B1 fix: these columns MUST exist in proj_df and donor_df.
# If even a single band is missing, Step 02 stops with an explicit error
# (no longer a silent skip). Set to False only with justification
# documented in the PDD for a specific caliper absent in the dataset.
REQUIRE_MANDATORY_CALIPERS = True
MANDATORY_CALIPER_COLUMNS = [
    "WRB2_CODE",      # → texture_class (exact match)
    "SOC_g_kg",       # ±10% PA mean
    "NDVI_t0",        # ±10% PA mean
    "elevation",      # ±200 m
    "slope_deg",      # ±10°
    "dist_roads_km",  # ±1 km
]

TENURE_COLUMN_CANDIDATES = [
    "tenure_status", "Tenure", "TENURE", "legal_status", "LegalStatus", "LEGAL_STATUS"
]

# ── WRB2_CODE → soil texture (HWSD2 v2.0 legend) ─────────────────────
# REALIGNED to the REAL HWSD2 numbering (table D_WRB2code of the
# HWSD2.mdb database) and to the DOMINANT USDA texture per soil group,
# derived data-driven from HWSD2_SMU.TEXTURE_USDA (dominant weighted by SHARE
# over the 29,539 components of the database).
#
# Replaces the previous 1–30 table which did NOT follow the HWSD2
# numbering: e.g. the old code 12 was "loam" but in HWSD2 the 12 = Glaciers;
# the 16 was "clay_loam" but it is Islands (non-soils). The texture caliper was
# therefore systematically wrong for HWSD2 data.
#
# USDA classes(13) → 5 STARR buckets:
#   clay        ← Clay heavy, Silty clay, Clay light, Sandy clay   {1,2,3,8}
#   clay_loam   ← Silty clay loam, Clay loam, Sandy clay loam      {4,5,10}
#   loam        ← Silt, Silt loam, Loam                            {6,7,9}
#   sandy_loam  ← Sandy loam, Loamy sand                           {11,12}
#   sand        ← Sand                                             {13}
#
# Non-soil codes (12 Glaciers, 16 Islands, 34 Open Water, 35 No Data) and
# Technosols (31, lacking texture data in HWSD2) are NOT mapped → the
# corresponding pixels are discarded in Step 02 (with a diagnostic warning).
WRB_TO_TEXTURE = {
    1:"sandy_loam",   # Acrisols
    2:"loam",         # Alisols
    3:"loam",         # Andosols
    4:"sandy_loam",   # Arenosols
    5:"loam",         # Anthrosols
    6:"loam",         # Chernozems
    7:"loam",         # Calcisols
    8:"clay_loam",    # Cambisols
    9:"loam",         # Cryosols
    10:"loam",        # Fluvisols
    11:"clay",        # Ferralsols
    13:"clay_loam",   # Gleysols
    14:"loam",        # Gypsisols
    15:"clay_loam",   # Histosols
    17:"loam",        # Kastanozems
    18:"loam",        # Leptosols
    19:"loam",        # Luvisols
    20:"sandy_loam",  # Lixisols
    21:"clay",        # Nitisols
    22:"loam",        # Phaeozems
    23:"loam",        # Planosols
    24:"sandy_loam",  # Plinthosols
    25:"sandy_loam",  # Podzols
    26:"sandy_loam",  # Regosols
    27:"loam",        # Retisols
    28:"loam",        # Solonchaks
    29:"loam",        # Solonetz
    30:"loam",        # Stagnosols
    32:"loam",        # Umbrisols
    33:"clay",        # Vertisols
}

# HWSD2 codes discarded ON PURPOSE from the donor pool (not an error):
#   - non-soils: 12 Glaciers, 16 Islands, 34 Open Water, 35 No Data
#   - anthropic: 31 Technosols (not a natural analogue, and lacking texture in HWSD2)
# If they appear in the donor they are removed in Step 02; the diagnostics flags them
# as "excluded (expected)" and NOT as codes to add to WRB_TO_TEXTURE.
WRB_NONDONOR_CODES = {
    12: "Glaciers", 16: "Islands", 31: "Technosols(anthropic)",
    34: "OpenWater", 35: "NoData",
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
    return "all" if v is None else f"{int(v):,}"


def load_df(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def find_file(base_dirs, stem, fmt="parquet"):
    for base in base_dirs:
        for ext in [fmt, "parquet", "csv"]:
            p = Path(base) / f"{stem}.{ext}"
            if p.exists():
                print(f"    Found: {p.name}")
                return p
    raise FileNotFoundError(f"'{stem}' not found in {[str(b) for b in base_dirs]}")


def wrb_to_texture(code):
    try:
        return WRB_TO_TEXTURE.get(int(float(code)))
    except Exception:
        return None


def assign_texture(df):
    df = df.copy()
    if "WRB2_CODE" not in df.columns:
        raise ValueError("WRB2_CODE missing. Check GEE raster Step 01.")
    df["texture_class"] = df["WRB2_CODE"].apply(wrb_to_texture)
    before = len(df)
    # Diagnostics: which WRB2_CODE remain unmapped in WRB_TO_TEXTURE
    # (→ discarded). Serves to immediately highlight missing codes (e.g. 33) instead
    # of letting the error propagate downstream as "Donor empty after prefilter".
    unmapped = df.loc[df["texture_class"].isna(), "WRB2_CODE"]
    df = df[df["texture_class"].notna()].reset_index(drop=True)
    print(f"    WRB filter: {before:,} → {len(df):,} valid px")
    if len(unmapped) > 0:
        vc = unmapped.round().astype("Int64").value_counts().sort_values(ascending=False)
        frac = len(unmapped) / max(before, 1) * 100
        # Separate the codes excluded on purpose (non-soils + Technosols) from those
        # truly unexpected (which would deserve to be mapped).
        known = [(c, n) for c, n in vc.items() if int(c) in WRB_NONDONOR_CODES]
        other = [(c, n) for c, n in vc.items() if int(c) not in WRB_NONDONOR_CODES]
        if known:
            kk = ", ".join(f"{WRB_NONDONOR_CODES[int(c)]}(code{int(c)})×{int(n):,}"
                           for c, n in known)
            print(f"    WRB2_CODE non-donor excluded (expected, {frac:.2f}%): {kk}")
        if other:
            oo = ", ".join(f"code{int(c)}×{int(n):,}" for c, n in other)
            print(f"    ⚠ UNEXPECTED WRB2_CODE not mapped: {oo}")
            print(f"      → if they are real soils, add them to WRB_TO_TEXTURE.")
    if len(df) == 0:
        raise RuntimeError(
            "No valid pixel after WRB filter: all WRB2_CODE are outside "
            "WRB_TO_TEXTURE. See the unmapped codes listed above and "
            "add them to the WRB_TO_TEXTURE table.")
    return df


def check_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")


def drop_invalid_covariate_rows(df, name, verbose=True):
    """Removes rows with physically impossible / nodata covariates.

    - SOC_g_kg <= MIN_VALID_SOC  (0 = nodata; SOC is in t/ha)
    - NDVI_t0 outside [-1, 1]

    Protects the pool from spurious values (e.g. SOC=0 that was polluting the donor).
    """
    if not DROP_INVALID_COVARIATES:
        return df
    n0   = len(df)
    keep = pd.Series(True, index=df.index)
    if "SOC_g_kg" in df.columns:
        keep &= (df["SOC_g_kg"] > MIN_VALID_SOC)
    if "NDVI_t0" in df.columns:
        keep &= df["NDVI_t0"].between(-1.0, 1.0)
    out   = df[keep].reset_index(drop=True)
    n_rem = n0 - len(out)
    if verbose and n_rem > 0:
        print(f"    Outlier/nodata {name}: {n_rem:,} rows removed "
              f"(SOC<=0 or NDVI outside [-1,1]) → {len(out):,}")
    return out


def detect_ndvi_year_cols(columns):
    import re
    return sorted([c for c in columns if re.match(r"^NDVI_\d{4}$", str(c))])


def detect_tenure_col(proj_df, donor_df):
    for c in TENURE_COLUMN_CANDIDATES:
        if c in proj_df.columns and c in donor_df.columns:
            return c
    return None


# ── PREFILTER (wide) ──────────────────────────────────────────────────

def auto_prefilter_donor(proj_df, donor_df, cont_covs):
    """Wide filter on the donor (±20% of the range). The exact calipers come afterwards."""
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
            raise RuntimeError(f"Donor empty after prefilter '{col}'.")
    return out.reset_index(drop=True)


def build_plain_mahalanobis(z_sample, cont_covs):
    """
    Plain Mahalanobis: Sigma^-1 via LedoitWolf on a sample.

    METH FIX: removed the RF weights (methodologically incorrect for GS STARR).
    The metric is the plain D(i,j) = sqrt((Xi-Xj)' Sigma^-1 (Xi-Xj)).

    RAM FIX: accepts only a sample (z_sample, at most COV_FIT_MAX_ROWS rows),
    not the entire z_proj+z_donor array (which would be 406K rows in RAM).

    Returns (metric=Sigma^-1+ridge, cov_orig=Sigma).
    """
    n_cov = len(cont_covs)
    lw = LedoitWolf(assume_centered=False)
    lw.fit(z_sample.astype(np.float64))
    cov    = lw.covariance_ + RIDGE_REG * np.eye(n_cov)
    metric = np.linalg.inv(cov) + RIDGE_REG * np.eye(n_cov)
    return metric, lw.covariance_


# ── CONDITIONAL GC ───────────────────────────────────────────────────

try:
    import psutil as _psutil; _PSUTIL = True
except ImportError:
    _PSUTIL = False

_GC_RAM_TRESH = 0.82

def _gc(force=False):
    """GC only if RAM > threshold or force=True. Avoids overhead in the hot loop."""
    if force: gc.collect(); return
    if _PSUTIL and _psutil.virtual_memory().percent / 100.0 > _GC_RAM_TRESH:
        gc.collect()


def adaptive_k(n_donor_pool, base_k=KNN_QUERY_CANDIDATES):
    """Adaptive candidates: if the pool is small, query everything."""
    return max(K_NEIGHBOURS, min(base_k, int(n_donor_pool * 0.95), n_donor_pool))


def build_whitening_matrix(metric):
    """Cholesky L t.c. Mahal(a,b)² = ||aL - bL||² (whitened Euclidean)."""
    eig = np.linalg.eigvalsh(metric)
    if eig.min() < 1e-10:
        metric = metric + (abs(eig.min()) + 1e-8) * np.eye(metric.shape[0])
    return scipy_cholesky(metric, lower=True)


# ── PRE-CACHING NUMPY ARRAYS FOR DONOR ───────────────────────────────

def _build_donor_arrays(donor_df, cont_covs, tenure_col, ctx):
    """
    Pre-extracts the relevant columns from donor_df as numpy arrays.
    This avoids the pandas iloc call for each candidate during
    the caliper check (it was the main bottleneck).
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

    # Precompute the tolerance context
    arr["ctx"] = ctx
    return arr


# ── VECTORIZED CALIPER CHECK ─────────────────────────────────────────

def _check_calipers_vectorized(proj_row, cand_global_idx, donor_arr, ctx, tenure_col):
    """
    Checks the calipers on ALL candidates in a single numpy pass.

    Input:
      proj_row:        Series (pandas row of the project pixel)
      cand_global_idx: int array, global indices in donor_df of the KNN candidates
      donor_arr:       dict of numpy arrays pre-extracted from donor_df

    Returns:
      passes: bool array of length len(cand_global_idx)
      n_rejected: int, number of rejected candidates
    """
    n      = len(cand_global_idx)
    passes = np.ones(n, dtype=bool)

    # 1. Exact texture
    proj_tex = str(proj_row.get("texture_class", ""))
    passes  &= (donor_arr["texture_class"][cand_global_idx] == proj_tex)

    # 2. Exact tenure (if available)
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

    # 5. Elevation (adaptive caliper: ctx["elev_tol"], default ±200 m)
    if donor_arr["elevation"] is not None:
        pv = float(proj_row.get("elevation", np.nan))
        if np.isfinite(pv):
            _elev_tol = ctx.get("elev_tol", CALIPER_ELEVATION_M)
            passes &= np.abs(donor_arr["elevation"][cand_global_idx] - pv) <= _elev_tol

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

def validate_mandatory_calipers(proj_df, donor_df):
    """
    B1 fix: explicit fail if a mandatory caliper band is missing.

    Checks that every column in MANDATORY_CALIPER_COLUMNS is present in
    BOTH proj_df and donor_df. Without these bands the corresponding caliper
    would be silently skipped and the match would pass on fewer constraints than
    Annex 1 Table A.3 requires.
    """
    if not REQUIRE_MANDATORY_CALIPERS:
        return
    def _ok(df, c):
        # WRB2_CODE may already have been converted to texture_class.
        if c == "WRB2_CODE":
            return ("WRB2_CODE" in df.columns) or ("texture_class" in df.columns)
        return c in df.columns
    missing_proj  = [c for c in MANDATORY_CALIPER_COLUMNS if not _ok(proj_df, c)]
    missing_donor = [c for c in MANDATORY_CALIPER_COLUMNS if not _ok(donor_df, c)]
    if missing_proj or missing_donor:
        raise RuntimeError(
            "STEP 02 blocked: mandatory caliper bands missing "
            "(GS STARR Annex 1 Table A.3).\n"
            f"  Missing in project_df : {missing_proj}\n"
            f"  Missing in donor_df   : {missing_donor}\n"
            "These columns are required for the hard calipers. Export them from GEE "
            "in Step 01, or (only with justification documented in the PDD) "
            "remove the band from MANDATORY_CALIPER_COLUMNS or set "
            "REQUIRE_MANDATORY_CALIPERS=False.\n"
            "NB: NDVI_t0 in particular is NOT generated by Step 01: it must be "
            "exported directly from the GEE raster."
        )


def stratified_donor_cap(donor_df, sample_n, strat_col="texture_class", seed=42):
    """
    B2 fix: donor cap STRATIFIED by texture (not a random sample).

    Keeps the proportion of each texture group in the sample, so that rare
    textures are not underrepresented. If strat_col is missing, falls back to a
    random sample (with warning).
    """
    n_total = len(donor_df)
    if sample_n is None or n_total <= int(sample_n):
        return donor_df.reset_index(drop=True)
    sample_n = int(sample_n)

    if strat_col not in donor_df.columns:
        print(f"    WARN: '{strat_col}' absent — random donor cap (not stratified).")
        return donor_df.sample(sample_n, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    frac = sample_n / n_total
    parts = []
    for _, grp in donor_df.groupby(strat_col, sort=False):
        # at least 1 row per non-empty group; round down elsewhere
        n_grp = max(1, int(round(len(grp) * frac)))
        n_grp = min(n_grp, len(grp))
        idx = rng.choice(grp.index.to_numpy(), size=n_grp, replace=False)
        parts.append(donor_df.loc[idx])
    out = pd.concat(parts).reset_index(drop=True)
    # rebalance if rounding exceeded/missed the target
    if len(out) > sample_n:
        out = out.sample(sample_n, random_state=seed).reset_index(drop=True)
    print(f"    STRATIFIED donor cap by {strat_col}: "
          f"{n_total:,} → {len(out):,} (target {sample_n:,})")
    return out


def run_matching(proj_df, donor_df, weights_dict, cont_covs, meta):
    """
    Matching stratified by WRB2 texture with vectorized hard calipers.

    The KNN logic is adapted from the QGIS PlotMatcherAlgorithm:
      - same Mahalanobis metric (VI = weighted Σ^-1)
      - but uses whitening + ball_tree instead of brute-force cdist
        (cdist is O(n*m) and does not scale beyond 50k donor pixels)
      - the hard calipers are applied vectorially on the KNN candidates
        instead of a Python loop per candidate (bottleneck eliminated)
    """
    # B1: explicit fail on missing caliper bands BEFORE any computation.
    validate_mandatory_calipers(proj_df, donor_df)

    # B2: donor cap stratified by texture (it was a random sample).
    if N_DONOR_SAMPLE is not None and len(donor_df) > int(N_DONOR_SAMPLE):
        donor_df = stratified_donor_cap(donor_df, N_DONOR_SAMPLE, strat_col="texture_class")

    check_cols(proj_df,  cont_covs + ["lon", "lat", "texture_class"], "project_df")
    check_cols(donor_df, cont_covs + ["lon", "lat", "texture_class"], "donor_df")

    # Caliper context — columns now guaranteed present (validate_mandatory_calipers)
    tenure_col = detect_tenure_col(proj_df, donor_df)
    ctx = {"tenure_col": tenure_col}
    _soc_mean = float(proj_df["SOC_g_kg"].mean()) if "SOC_g_kg" in proj_df.columns else np.nan
    _ndvi_mean = float(proj_df["NDVI_t0"].mean()) if "NDVI_t0" in proj_df.columns else np.nan
    if not np.isfinite(_soc_mean) or not np.isfinite(_ndvi_mean):
        raise RuntimeError(
            "Caliper tolerance not computable: PA mean NaN for "
            f"SOC_g_kg ({_soc_mean}) or NDVI_t0 ({_ndvi_mean}). "
            "The caliper bands are present but have no valid values in the PA."
        )
    ctx["soc_tol"]  = max(abs(_soc_mean)  * CALIPER_SOC_FRAC_OF_PROJECT_MEAN, 1e-9)
    ctx["ndvi_tol"] = max(abs(_ndvi_mean) * CALIPER_NDVI_T0_FRAC_OF_PROJECT_MEAN, 1e-9)

    # Elevation caliper adaptive to the PA's elevation variability.
    _elev_std = float(proj_df["elevation"].std()) if "elevation" in proj_df.columns else np.nan
    if CALIPER_ELEVATION_ADAPTIVE and np.isfinite(_elev_std) and _elev_std > 0:
        ctx["elev_tol"] = float(np.clip(CALIPER_ELEVATION_SD_MULT * _elev_std,
                                        CALIPER_ELEVATION_MIN_M, CALIPER_ELEVATION_M))
    else:
        ctx["elev_tol"] = CALIPER_ELEVATION_M

    print("    Active hard calipers (all mandatory present):")
    print(f"      exact texture : YES")
    print(f"      exact tenure  : {'YES ('+tenure_col+')' if tenure_col else 'N/A (optional)'}")
    print(f"      SOC           : ±{ctx['soc_tol']:.4f}")
    print(f"      NDVI_t0       : ±{ctx['ndvi_tol']:.4f}")
    print(f"      elevation     : ±{ctx['elev_tol']:.1f} m"
          + (f"  (adaptive: sd_PA={_elev_std:.1f} m × {CALIPER_ELEVATION_SD_MULT:.0f}, "
             f"max {CALIPER_ELEVATION_M:.0f})"
             if CALIPER_ELEVATION_ADAPTIVE and np.isfinite(_elev_std) else ""))
    print(f"      slope         : ±{CALIPER_SLOPE_DEG:.0f}°")
    print(f"      dist_roads    : ±{CALIPER_ROADS_KM:.1f} km")

    # ── SCALING + WHITENING + KNN BATCH (RAM FIX) ───────────────────────
    #
    # PROBLEM (OOM with 106K project pixels):
    #   z_all = scaler.fit_transform(pd.concat([proj, donor]))  -> 406K x n_cov RAM
    #   zp_w  = z_proj @ L                                      -> 106K x n_cov permanent
    #   knn.kneighbors(zp_w[proj_idx]) on all 106K             -> (106K x 300) x 16 bytes = 508 MB output
    #
    # FIX: zp_w is NEVER precomputed for all project pixels.
    # Scale + whitening are applied on-the-fly for KNN_BATCH_SIZE pixels at a time.
    # RAM per batch: 4096 x n_cov x 8 = 640 KB instead of 17 MB in permanent RAM.

    rng   = np.random.default_rng(42)
    n_cov = len(cont_covs)

    # 1. Fit scaler on a sample (no full proj+donor concat in RAM)
    n_p_s = min(len(proj_df),  SCALER_FIT_MAX_ROWS // 2)
    n_d_s = min(len(donor_df), SCALER_FIT_MAX_ROWS // 2)
    Xp_s  = proj_df.iloc[rng.choice(len(proj_df),  n_p_s, replace=False)][cont_covs].to_numpy(dtype=np.float64)
    Xd_s  = donor_df.iloc[rng.choice(len(donor_df), n_d_s, replace=False)][cont_covs].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    scaler.fit(np.vstack([Xp_s, Xd_s]))

    # 2. LedoitWolf on the scaled sample (no full proj+donor concat)
    z_samp = np.vstack([scaler.transform(Xp_s), scaler.transform(Xd_s)]).astype(np.float64)
    del Xp_s, Xd_s
    metric, cov_orig = build_plain_mahalanobis(z_samp, cont_covs)
    del z_samp
    L = build_whitening_matrix(metric).astype(np.float64)

    # 3. Transform + whiten the WHOLE donor (needed for the KNN index)
    #    Max 300K x n_cov x 8 = 48 MB — acceptable
    Xd_full = donor_df[cont_covs].to_numpy(dtype=np.float64)
    zdw     = (scaler.transform(Xd_full) @ L).astype(np.float64)
    del Xd_full
    gc.collect()

    # 4. KNN is built PER-TEXTURE GROUP inside the loop below (NOT one global
    #    index). This is the fix for the texture-starvation bug: a single global
    #    KNN returns the k_global nearest donors across ALL textures, and the
    #    same-texture ones are then filtered out AFTERWARDS. If the project's
    #    texture is a minority in the donor pool, few (or zero) of those global
    #    neighbours share the texture → the pixel is left unmatched even though
    #    plenty of same-texture donors exist. Building the KNN on the
    #    same-texture donor subset guarantees every project pixel receives up to
    #    KNN_QUERY_CANDIDATES SAME-TEXTURE candidates before the hard calipers.
    print(f"    Per-texture KNN (whitened Mahalanobis) | candidates={KNN_QUERY_CANDIDATES} "
          f"| n_jobs={KNN_N_JOBS}")
    print(f"    RAM FIX: zp_w on-the-fly per batch of {KNN_BATCH_SIZE} px (never 106K in memory)")

    donor_df   = donor_df.reset_index(drop=True)
    donor_arr  = _build_donor_arrays(donor_df, cont_covs, tenure_col, ctx)
    donor_reuse = np.zeros(len(donor_df), dtype=np.int32)

    available_set = set(donor_df["texture_class"].dropna().unique())
    print(f"    Donor texture groups: {sorted(available_set)}")

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
            print(f"    SKIP texture '{texture}': no donor.")
            for pi in proj_idx:
                unmatched_rows.append({"proj_idx": int(pi), "proj_texture": str(texture),
                                       "reason": "no_texture_donor"})
            continue

        d_mask   = np.isin(donor_arr["texture_class"], [str(a) for a in allowed])
        d_global = np.where(d_mask)[0]
        if not len(d_global):
            continue

        # Per-texture KNN index: built on the SAME-TEXTURE whitened donor subset
        # only, so every candidate returned already matches the texture caliper.
        k_query   = min(max(K_NEIGHBOURS, KNN_QUERY_CANDIDATES), len(d_global))
        zdw_tex   = zdw[d_global]
        knn_tex   = NearestNeighbors(
            n_neighbors=k_query, metric="euclidean",
            algorithm="ball_tree", leaf_size=KNN_LEAF_SIZE, n_jobs=KNN_N_JOBS
        )
        knn_tex.fit(zdw_tex)

        n_matched = n_unmatched = rejected_total = 0
        dist_sum  = dist_n = 0.0

        # ── BATCH LOOP: scale+whiten on-the-fly for KNN_BATCH_SIZE pixels ─────
        for start in range(0, len(proj_idx), KNN_BATCH_SIZE):
            batch = proj_idx[start : start + KNN_BATCH_SIZE]

            # ON-THE-FLY: scale + whiten only this batch
            # RAM: KNN_BATCH_SIZE x n_cov x 8 = 640 KB (not 17 MB)
            Xp_b  = proj_df.loc[batch, cont_covs].to_numpy(dtype=np.float64)
            zpw_b = (scaler.transform(Xp_b) @ L).astype(np.float64)
            del Xp_b

            # Per-texture KNN query (every candidate is already same-texture)
            dists_t, idx_t = knn_tex.kneighbors(zpw_b)
            del zpw_b

            for row_i in range(len(batch)):
                pi = int(batch[row_i])
                pr = proj_df.loc[pi]

                # Map local indices (into the same-texture subset) back to the
                # global donor_df indices. No texture post-filter is needed
                # because knn_tex was fitted on same-texture donors only.
                cand_local  = idx_t[row_i]
                cand_global = d_global[cand_local]
                cand_dists  = dists_t[row_i]

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

                # ── K:1 SELECTION ─────────────────────────────────────────────
                # Select up to K_NEIGHBOURS best passing donors (already ordered
                # by ascending match distance); each becomes one reference record
                # for this project pixel. K_NEIGHBOURS=1 → classic 1:1 matching.
                # MAX_DONOR_REUSE is FLAGGED (reuse_exceeded) but not blocked, so
                # a donor may serve several project pixels; Step 05 already uses
                # the count of UNIQUE reference pixels as the effective N for the CI.
                sel_locals = passing_local[:max(1, K_NEIGHBOURS)]
                for knn_order, sel_local in enumerate(sel_locals, start=1):
                    sel_local   = int(sel_local)
                    sel_global  = int(cand_global[sel_local])
                    sel_dist    = float(cand_dists[sel_local])
                    reuse       = int(donor_reuse[sel_global])
                    exceeded    = bool(reuse >= MAX_DONOR_REUSE)
                    donor_reuse[sel_global] += 1

                    dr  = donor_df.iloc[sel_global]
                    rec = {c: dr.get(c) for c in donor_df.columns if c != "_gidx"}
                    rec.update({
                        "run_id":         run_id,
                        "ref_lon":        float(dr["lon"]),
                        "ref_lat":        float(dr["lat"]),
                        "proj_lon":       float(pr["lon"]),
                        "proj_lat":       float(pr["lat"]),
                        "match_distance": sel_dist,
                        "match_rank":     int(sel_local + 1),
                        "match_knn_order": int(knn_order),   # 1..K_NEIGHBOURS (K:1 matching)
                        "proj_idx":       pi,
                        "proj_texture":   str(texture),
                        "ref_texture":    str(dr.get("texture_class", "")),
                        "texture_exact":  bool(exact),
                        "reuse_exceeded": exceeded,
                        "all_calipers_passed": True,
                        "hard_caliper_rejected_before_selected": n_rej,
                    })
                    # B3 fix: propagate the donor pixel's native bounds with ref_ prefix
                    # so Step 04 uses the real raster footprint (no centroid fallback).
                    for _src, _dst in [
                        ("cell_xmin", "ref_cell_xmin"), ("cell_ymin", "ref_cell_ymin"),
                        ("cell_xmax", "ref_cell_xmax"), ("cell_ymax", "ref_cell_ymax"),
                        ("pixel_id",  "ref_pixel_id"),
                        ("grid_row",  "ref_grid_row"), ("grid_col",  "ref_grid_col"),
                    ]:
                        if _src in dr.index:
                            rec[_dst] = dr[_src]
                    # proj_/ref_/diff_ for EVERY matching covariate (not just the 5
                    # mandatory ones) so the per-covariate diagnostics are self-contained.
                    _pair_cols = list(dict.fromkeys(
                        ["SOC_g_kg", "NDVI_t0", "elevation", "slope_deg", "dist_roads_km"]
                        + list(cont_covs)))
                    for c in _pair_cols:
                        if c in pr.index and c in dr.index:
                            rec[f"proj_{c}"] = float(pr[c])
                            rec[f"ref_{c}"]  = float(dr[c])
                            rec[f"diff_{c}"] = float(dr[c]) - float(pr[c])
                    records.append(rec)
                    n_matched += 1

            del dists_t, idx_t
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


# ── VALIDATION ────────────────────────────────────────────────────────

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
        raise RuntimeError("No valid covariate for SMD.")
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
    ax.set(xlabel="RF feature importance",
           title="Feature importance\nproject vs donor discrimination")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    s = imp_df.sort_values("knn_weight")
    ax.barh(s["covariate"], s["knn_weight"],
            color=plt.cm.RdYlGn(s["knn_weight"]/max_w))
    ax.axvline(1.0, color="black", ls=":", alpha=0.4, label="w=1 (neutral)")
    ax.set(xlabel="KNN weight (data-driven)",
           title="Data-driven Mahalanobis weights")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "rf_weights.png", dpi=150, bbox_inches="tight")
    return fig


def plot_smd(smd_df, out_dir=None, filename="SMD_lollipop.png", title="Covariate balance"):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9, max(4, len(smd_df)*0.35)))
    c_pass, c_fail = "#2ca02c", "#d62728"
    colors = [c_fail if not p else c_pass for p in smd_df["passed"]]
    ax.hlines(smd_df.index, 0, smd_df["SMD"].fillna(0).values, lw=2, color=colors)
    ax.scatter(smd_df["SMD"].fillna(0).values, smd_df.index, s=70, color=colors, zorder=5)
    ax.axvline(SMD_THRESHOLD, color="black", ls="--", lw=1.5)
    # Explicit legend: what each dot/colour means + the labelled threshold line
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c_pass, markersize=9,
               label=f"PASS  (SMD < {SMD_THRESHOLD})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c_fail, markersize=9,
               label=f"FAIL  (SMD ≥ {SMD_THRESHOLD})"),
        Line2D([0], [0], color="black", ls="--", lw=1.5,
               label=f"Balance threshold = {SMD_THRESHOLD}"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="best", framealpha=0.9)
    ax.set(xlabel="Standardized Mean Difference (SMD)  —  GS STARR target < 0.1",
           ylabel="Covariate",
           title=title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_match_distances(matched_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    d = matched_df["match_distance"].astype(float).values
    ax.hist(d, bins=60, edgecolor="white", color="#4C72B0",
            label="Matched pixels (project → reference)")
    ax.axvline(d.mean(), color="red", ls="--", lw=1.5,
               label=f"Mean = {d.mean():.3f}")
    ax.axvline(np.percentile(d, 95), color="orange", ls=":", lw=1.5,
               label=f"P95 = {np.percentile(d,95):.3f}")
    ax.set(xlabel="Whitened Mahalanobis distance (lower = closer match)",
           ylabel="Number of pixels",
           title="Match distance distribution")
    ax.legend(fontsize=8, framealpha=0.9); ax.grid(alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "match_distance_distribution.png", dpi=150, bbox_inches="tight")
    return fig


def plot_covariate_pairs(proj_df, matched_df, smd_df, cont_covs,
                         caliper_tol=None, out_dir=None):
    """Per-covariate hexbin of the matched pairs: project pixel value (x) vs its
    matched donor value (y), with the 1:1 line, the caliper band (when in range)
    and the SMD annotation (full PA vs matched donors).

    Uses the paired columns proj_<cov>/ref_<cov> written by run_matching for the
    mandatory calipers, and falls back to proj_idx (join to proj_df) for the
    other covariates (e.g. NDVI_slope_5yr).
    """
    caliper_tol = caliper_tol or {}
    covs = [c for c in cont_covs if (c in matched_df.columns or f"ref_{c}" in matched_df.columns)]
    if not covs:
        return None
    has_pidx = "proj_idx" in matched_df.columns
    ncol = 3
    nrow = int(np.ceil(len(covs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.3, nrow * 3.6), squeeze=False)
    axes = axes.ravel()

    for i, cov in enumerate(covs):
        ax = axes[i]
        # donor (y) and project (x) values, paired per matched row
        y = (matched_df[f"ref_{cov}"] if f"ref_{cov}" in matched_df.columns
             else matched_df[cov]).astype(float).to_numpy()
        if f"proj_{cov}" in matched_df.columns:
            x = matched_df[f"proj_{cov}"].astype(float).to_numpy()
        elif has_pidx and cov in proj_df.columns:
            x = proj_df[cov].reindex(matched_df["proj_idx"].to_numpy()).astype(float).to_numpy()
        else:
            ax.set_visible(False)
            continue
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) == 0:
            ax.set_visible(False)
            continue

        ax.hexbin(x, y, gridsize=30, cmap="viridis", mincnt=1, linewidths=0.2)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], "k-", lw=1.0)                     # 1:1 line
        tol = caliper_tol.get(cov)
        if tol is not None and 0 < tol < (hi - lo):                  # band only if visible in range
            ax.plot([lo, hi], [lo + tol, hi + tol], "r--", lw=0.8, alpha=0.7)
            ax.plot([lo, hi], [lo - tol, hi - tol], "r--", lw=0.8, alpha=0.7)

        if cov in smd_df.index:
            smd = smd_df.loc[cov, "SMD"]
            tag = "PASS" if bool(smd_df.loc[cov, "passed"]) else "FAIL"
            ax.set_title(f"{cov} | SMD={smd:.4f} {tag}", fontsize=9)
        else:
            ax.set_title(cov, fontsize=9)
        ax.set_xlabel("Project Area")
        ax.set_ylabel("Matched donor")
        ax.grid(alpha=0.25)

    for j in range(len(covs), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Matched pairs; SMD annotated = full PA vs matched donors", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if out_dir:
        fig.savefig(Path(out_dir) / "covariate_match_pairs.png", dpi=150, bbox_inches="tight")
    return fig


# ── MAIN ─────────────────────────────────────────────────────────────

def run_matching_step(base_dirs=None, output_dir=None,
                      proj_df=None, donor_df=None, meta=None, verbose=True):
    """
    Returns (matched_df, weights_dict, imp_df, smd_df, figs, out_dir).
    """
    if base_dirs is None:
        raise RuntimeError(
            "base_dirs not provided. Pass base_dirs=[out01] from the runner "
            "or specify the Step 01 output directory."
        )
    base_dirs = [Path(b) for b in base_dirs]
    out_dir   = Path(output_dir) if output_dir else base_dirs[0].parent / STEP_DIRNAME
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
                 and c not in EXCLUDE_COVARIATES_FROM_MATCHING
                 and not c.startswith("precip_bin")]
    if not cont_covs:
        raise RuntimeError("continuous_covariates empty. Run Step 01.")
    if EXCLUDE_COVARIATES_FROM_MATCHING:
        _excl = [c for c in EXCLUDE_COVARIATES_FROM_MATCHING
                 if c in (meta.get("continuous_covariates") or [])]
        if _excl:
            print(f"    Covariates EXCLUDED from matching (config): {_excl}")

    ndvi_year_cols = meta.get("ndvi_year_cols") or detect_ndvi_year_cols(proj_df.columns)
    print(f"    Mahalanobis covariates: {cont_covs}")
    print(f"    Annual NDVI: {ndvi_year_cols}")

    check_cols(proj_df,  cont_covs + ["WRB2_CODE", "lon", "lat"], "project_df")
    check_cols(donor_df, cont_covs + ["WRB2_CODE", "lon", "lat"], "donor_df")

    print("\n[1] Texture WRB...")
    # B1: explicit fail on mandatory caliper bands BEFORE transforming the data.
    validate_mandatory_calipers(proj_df, donor_df)
    proj_df  = assign_texture(proj_df)
    donor_df = assign_texture(donor_df)

    proj_df  = proj_df.dropna(subset=cont_covs).reset_index(drop=True)
    donor_df = donor_df.dropna(subset=cont_covs).reset_index(drop=True)
    # Removal of physically invalid outliers/nodata (SOC<=0, NDVI out of range)
    proj_df  = drop_invalid_covariate_rows(proj_df,  "PROJECT")
    donor_df = drop_invalid_covariate_rows(donor_df, "DONOR")
    print(f"    Project: {len(proj_df):,} | Donor: {len(donor_df):,}")
    if len(proj_df) == 0 or len(donor_df) == 0:
        raise RuntimeError("Empty dataset after dropna/covariate outlier removal.")

    print("\n[2] Donor prefilter (wide)...")
    donor_df = auto_prefilter_donor(proj_df, donor_df, cont_covs)

    weights_dict = {c: 1.0 for c in cont_covs}  # plain Mahalanobis: uniform weights (no RF)
    imp_df = None

    print("\n[3] KNN plain Mahalanobis matching + hard calipers (batched, without RF)...")
    matched_df, tex_summary, unmatched_df = run_matching(
        proj_df, donor_df, weights_dict, cont_covs, meta)

    if matched_df.empty:
        raise RuntimeError("No pixel matched after hard calipers. "
                           "Increase KNN_QUERY_CANDIDATES or enable ALLOW_TEXTURE_FALLBACK.")

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

    print("\n[6] Plot + saving...")
    fig_s = plot_smd(smd_df, out_dir, "SMD_lollipop.png", "Mandatory covariate balance")
    fig_d = plot_match_distances(matched_df, out_dir)
    # Per-covariate hexbin of the matched pairs (project vs matched donor)
    caliper_tol = {
        "SOC_g_kg":      (abs(float(proj_df["SOC_g_kg"].mean())) * CALIPER_SOC_FRAC_OF_PROJECT_MEAN
                          if "SOC_g_kg" in proj_df.columns else None),
        "NDVI_t0":       (abs(float(proj_df["NDVI_t0"].mean())) * CALIPER_NDVI_T0_FRAC_OF_PROJECT_MEAN
                          if "NDVI_t0" in proj_df.columns else None),
        "elevation":     CALIPER_ELEVATION_M,
        "slope_deg":     CALIPER_SLOPE_DEG,
        "dist_roads_km": CALIPER_ROADS_KM,
    }
    fig_p = plot_covariate_pairs(proj_df, matched_df, smd_df, cont_covs, caliper_tol, out_dir)
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
        "matching_method":       "Plain Mahalanobis KNN, on-the-fly batches (no RF weights)",
        "donor_pool_rule":       "Non-Forest at T0, >5km from the PA (GEE Annex A.2.2 Step A)",
        "donor_cap_method":      "stratified_by_texture_class" if N_DONOR_SAMPLE is not None else "no_cap",
        "donor_loading_note":    ("Donor fully loaded in RAM from parquet, then "
                                  "stratified cap by texture. It is NOT batch streaming "
                                  "(the runner comment was inaccurate)."),
        "mandatory_calipers_enforced": bool(REQUIRE_MANDATORY_CALIPERS),
        "mandatory_caliper_columns":   list(MANDATORY_CALIPER_COLUMNS),
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
