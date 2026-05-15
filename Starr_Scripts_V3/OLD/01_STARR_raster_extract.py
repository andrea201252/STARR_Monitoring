# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB  |  01_STARR_raster_extract.py
STEP 01 — Raster Extraction + FAST Aligned Pixel Grid Plot
============================================================

PARAMETRI MANUALI: nessuno.

Logica di campionamento adattata da QGIS RasterSamplerCore:
  - Reprojection batched in C (un'unica chiamata pyproj per tutti i pixel)
  - Snap dei centroidi al grid raster usando rasterio transform
    (origin_x, origin_y, pixel_size) → garantita perfetta allineamento
  - Bounds delle celle come array numpy (non shapely list)
  - Rendering con matplotlib PatchCollection (C-level), non geopandas.plot
  - Filtraggio zoom via numpy boolean indexing sul bbox, non sort O(n log n)

Donor pool (GS STARR Annex A.2.2 Step A): pixel non-forest a T0,
>5 km dalla PA, filtrati in GEE prima dell'export del raster.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.windows import Window

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def _is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except ImportError:
        return False


if not _is_notebook():
    matplotlib.use("Agg")


# ── CONFIG ────────────────────────────────────────────────────────────

RUN_ID       = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v06_raster"
PROJECT_NAME = "Idiofa_Lobi"

BASE_DIR_CANDIDATES = [
    "/content/drive/MyDrive/STARR_Idiofa_New_V2",
    "/content/content/MyDrive/STARR_Idiofa_New_V2",
]

OUTPUT_DIR          = None
PROJECT_TIF_PATTERN = f"covariates_project_{RUN_ID}*.tif"
DONOR_TIF_PATTERN   = f"covariates_donor_{RUN_ID}*.tif"
OUTPUT_FORMAT       = "parquet"
BLOCK_HEIGHT        = None   # None = intero raster; 2048 per > 4 GB


# ── AUTO-DETECT ───────────────────────────────────────────────────────

def detect_band_names(src):
    descs = src.descriptions
    if descs and all(d is not None and d.strip() != "" for d in descs):
        return list(descs)
    return [f"band_{i+1}" for i in range(src.count)]


def detect_ndvi_year_cols(band_names):
    pat  = re.compile(r"^NDVI_(\d{4})$")
    hits = [(int(pat.match(b).group(1)), b) for b in band_names if pat.match(b)]
    hits.sort()
    return [b for _, b in hits], [y for y, _ in hits]


def detect_t0_year(band_names, year_list):
    return max(year_list) + 1 if year_list else None


def detect_continuous_covariates(band_names):
    skip = {"is_project", "is_donor", "ndvi_valid_years", "pixel_area_ha",
            "lon", "lat", "precip_bin", "precip_bin_100mm", "precip_bin_250mm"}
    skip_pat = re.compile(r"^(NDVI_\d{4}|band_\d+)$")
    return [b for b in band_names
            if b not in skip and not skip_pat.match(b) and b != "WRB2_CODE"]


# ── RASTER READING ────────────────────────────────────────────────────

def _process_block(data, transform, transformer, band_names, row_offset):
    H, W = data.shape[1], data.shape[2]
    m    = data.mask

    if np.ndim(m) == 0:
        invalid_2d  = np.full((H, W), bool(m), dtype=bool)
        invalid_2d |= ~np.isfinite(data.data).all(axis=0)
    elif np.ndim(m) == 2:
        invalid_2d  = m.astype(bool)
        invalid_2d |= ~np.isfinite(data.data).all(axis=0)
    else:
        invalid_2d  = np.any(m, axis=0)
        invalid_2d |= ~np.isfinite(data.data).all(axis=0)

    valid_2d = ~invalid_2d
    if not valid_2d.any():
        return pd.DataFrame(columns=band_names + ["lon", "lat"])

    rows, cols  = np.where(valid_2d)
    global_rows = rows + row_offset
    xs, ys      = rasterio.transform.xy(transform, global_rows, cols)
    lons, lats  = transformer.transform(np.asarray(xs, float), np.asarray(ys, float))

    df = pd.DataFrame(data.data[:, rows, cols].T, columns=band_names)
    df["lon"] = lons
    df["lat"] = lats
    return df


def extract_tile(tif_path, transformer):
    with rasterio.open(tif_path) as src:
        band_names = detect_band_names(src)
        epsg       = src.crs.to_epsg() if src.crs else "?"
        transform  = src.transform
        pixel_size = abs(transform.a)
        print(f"    {tif_path.name}")
        print(f"      {src.width}×{src.height}px | EPSG:{epsg} | {src.count} bande")
        print(f"      Origin: ({transform.c:.2f}, {transform.f:.2f}) | "
              f"Pixel: {pixel_size:.1f}×{abs(transform.e):.1f} m")

        if BLOCK_HEIGHT is None:
            data = src.read(masked=True)
            dfs  = [_process_block(data, transform, transformer, band_names, 0)]
        else:
            dfs = []
            for r0 in range(0, src.height, BLOCK_HEIGHT):
                h   = min(BLOCK_HEIGHT, src.height - r0)
                win = Window(0, r0, src.width, h)
                dfs.append(_process_block(src.read(window=win, masked=True),
                                          transform, transformer, band_names, r0))

    return pd.concat([d for d in dfs if len(d) > 0], ignore_index=True), band_names, transform


def load_raster_to_dataframe(tif_files, label):
    if not tif_files:
        raise FileNotFoundError(f"Nessun TIF trovato per {label}.")

    with rasterio.open(tif_files[0]) as src:
        crs_str    = src.crs.to_string() if src.crs else "EPSG:32734"
        pixel_size = abs(src.transform.a)
    transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
    print(f"    CRS: {crs_str} | pixel: {pixel_size:.1f} m")

    frames, band_names, first_transform = [], None, None
    for i, f in enumerate(tif_files):
        t0 = time.time()
        df, bn, tr = extract_tile(f, transformer)
        if band_names is None:
            band_names, first_transform = bn, tr
        print(f"      → {len(df):,} px ({time.time()-t0:.1f}s)")
        if len(df) > 0:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            f"Nessun pixel valido da {label}. "
            "Controllare diagnostics GEE sezione 9: [6] deve essere > 0 ha."
        )

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    if before > len(merged):
        print(f"    Rimossi {before-len(merged):,} duplicati lon/lat")

    ndvi_year_cols, year_list = detect_ndvi_year_cols(list(merged.columns))
    t0_year   = detect_t0_year(list(merged.columns), year_list)
    cont_covs = detect_continuous_covariates(band_names or [])

    print(f"    {label}: {len(merged):,} px | T0={t0_year} | NDVI={year_list}")
    return merged, band_names, ndvi_year_cols, year_list, t0_year, cont_covs, crs_str, pixel_size, first_transform


def find_tif_tiles(base_dirs, pattern, label):
    files, seen = [], set()
    for base in base_dirs:
        for p in Path(base).rglob(pattern):
            k = str(p.resolve())
            if k not in seen:
                seen.add(k); files.append(p.resolve())
    files = sorted(files)
    print(f"    {label}: {len(files)} tile trovate")
    for f in files:
        print(f"      {f.name}")
    return files


def save_df(df, path):
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    out = p.with_suffix(f".{OUTPUT_FORMAT}")
    if OUTPUT_FORMAT == "parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    return out


# ══════════════════════════════════════════════════════════════════════
# FAST VECTORIZED PIXEL GRID PLOT
# (logica adattata da QGIS RasterSamplerCore)
# ══════════════════════════════════════════════════════════════════════

def _snap_centers_to_grid(xs, ys, origin_x, origin_y, pix, half):
    """
    Snap pixel centers a un grid regolare definito da (origin_x, origin_y, pix).
    
    Conformemente a rasterio:
      pixel(r, c).center = (origin_x + (c+0.5)*pix,  origin_y - (r+0.5)*pix)
    
    Snapping di (x, y) al centro più vicino:
      c = round((x - origin_x - half) / pix)
      r = round((origin_y - half - y) / pix)
    """
    cs = np.round((xs - origin_x - half) / pix).astype(np.int64)
    rs = np.round((origin_y - half - ys) / pix).astype(np.int64)
    xs_snap = origin_x + cs * pix + half
    ys_snap = origin_y - rs * pix - half
    return xs_snap, ys_snap


def _build_rect_patches(x_mins, y_mins, widths, heights):
    """Costruisce una lista di Rectangle in batch (più veloce di shapely.box)."""
    # zip in C tramite numpy iter; Rectangle constructor è leggero
    return [Rectangle((x0, y0), w, h)
            for x0, y0, w, h in zip(x_mins, y_mins, widths, heights)]


def plot_aligned_pixel_grids(proj_df, donor_df, meta, out_dir=None,
                              zoom_size_pix_panel_c=30,
                              zoom_size_pix_fig2=20):
    """
    Costruisce due figure inline che mostrano le griglie pixel allineate.

    Pipeline (tutto vettorializzato):
      1. ONE pyproj batch reproject WGS84→UTM
      2. Snap dei centroidi al grid raster (numpy)
      3. Calcolo bounds delle celle (numpy array)
      4. Filter via numpy boolean indexing (per zoom)
      5. Rendering via matplotlib PatchCollection (C-level)

    Allineamento garantito perché PA e donor pool sono esportati da GEE
    con identica `crsTransform` (stesso origin, stesso scale=30m).
    """
    t_total = time.time()
    crs_utm = meta.get("crs_src",  "EPSG:32734")
    pix     = float(meta.get("pixel_size_m", 30.0))
    half    = pix / 2.0
    tr_dict = meta.get("raster_transform") or {}
    origin_x = tr_dict.get("c")
    origin_y = tr_dict.get("f")
    run_id   = meta.get("run_id", "")

    # ── (1) Batched reproject ────────────────────────────────────────
    t0 = time.time()
    tr_fwd = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    proj_xs_raw,  proj_ys_raw  = tr_fwd.transform(proj_df["lon"].values,  proj_df["lat"].values)
    donor_xs_raw, donor_ys_raw = tr_fwd.transform(donor_df["lon"].values, donor_df["lat"].values)
    proj_xs_raw  = np.asarray(proj_xs_raw,  dtype=np.float64)
    proj_ys_raw  = np.asarray(proj_ys_raw,  dtype=np.float64)
    donor_xs_raw = np.asarray(donor_xs_raw, dtype=np.float64)
    donor_ys_raw = np.asarray(donor_ys_raw, dtype=np.float64)
    print(f"    [1] Batched reproject: {time.time()-t0:.2f}s "
          f"(PA={len(proj_df):,} + donor={len(donor_df):,})")

    # ── (2) Snap to raster grid ──────────────────────────────────────
    t0 = time.time()
    if origin_x is not None and origin_y is not None:
        proj_xs,  proj_ys  = _snap_centers_to_grid(proj_xs_raw,  proj_ys_raw,
                                                    origin_x, origin_y, pix, half)
        donor_xs, donor_ys = _snap_centers_to_grid(donor_xs_raw, donor_ys_raw,
                                                    origin_x, origin_y, pix, half)
        snap_msg = "snapped to raster grid"
    else:
        proj_xs,  proj_ys  = proj_xs_raw,  proj_ys_raw
        donor_xs, donor_ys = donor_xs_raw, donor_ys_raw
        snap_msg = "raw centroids (no transform)"
    print(f"    [2] Snap to grid: {time.time()-t0:.2f}s — {snap_msg}")

    # ── (3) Cell bounds as numpy arrays ──────────────────────────────
    p_x0 = proj_xs - half;  p_x1 = proj_xs + half
    p_y0 = proj_ys - half;  p_y1 = proj_ys + half
    d_x0 = donor_xs - half; d_x1 = donor_xs + half
    d_y0 = donor_ys - half; d_y1 = donor_ys + half

    pa_minx, pa_maxx = float(proj_xs.min()), float(proj_xs.max())
    pa_miny, pa_maxy = float(proj_ys.min()), float(proj_ys.max())
    pa_cx, pa_cy     = (pa_minx + pa_maxx)/2, (pa_miny + pa_maxy)/2

    # ── (4) FIGURE 1: 3 panels ───────────────────────────────────────
    t0 = time.time()
    fig1, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel A — overview WGS84 (scatter, no boxes)
    ax = axes[0]
    n_d_show = min(15000, len(donor_df))
    rng = np.random.default_rng(42)
    samp_idx = rng.choice(len(donor_df), n_d_show, replace=False)
    ax.scatter(donor_df["lon"].values[samp_idx], donor_df["lat"].values[samp_idx],
               s=0.4, alpha=0.18, color="steelblue",
               label=f"Donor ({len(donor_df):,}, mostrati {n_d_show:,})")
    ax.scatter(proj_df["lon"], proj_df["lat"], s=3, alpha=0.75, color="tomato",
               label=f"PA ({len(proj_df):,})")
    ax.set(xlabel="Longitudine (WGS84)", ylabel="Latitudine",
           title="Overview centroidi\n(WGS84 — scatter, no boxes)")
    ax.legend(fontsize=8, markerscale=4); ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # Panel B — PA grid completa (PatchCollection vettorializzato)
    ax = axes[1]
    pa_widths  = np.full(len(proj_df), pix, dtype=np.float64)
    pa_heights = np.full(len(proj_df), pix, dtype=np.float64)
    pa_rects   = _build_rect_patches(p_x0, p_y0, pa_widths, pa_heights)
    pa_coll = PatchCollection(pa_rects, facecolor="tomato",
                               edgecolor="darkred", linewidth=0.25, alpha=0.75)
    ax.add_collection(pa_coll)
    margin = pix * 3
    ax.set_xlim(pa_minx - margin, pa_maxx + margin)
    ax.set_ylim(pa_miny - margin, pa_maxy + margin)
    ax.set(xlabel=f"Easting UTM ({crs_utm})", ylabel="Northing UTM",
           title=f"PA grid completa — {len(proj_df):,} celle {pix:.0f}×{pix:.0f} m\n"
                 f"PatchCollection vettorializzato")
    ax.set_aspect("equal")

    # Righello pixel
    rx = pa_minx - margin*0.6
    ry = pa_miny - margin*0.6
    ax.annotate("", xy=(rx + pix, ry), xytext=(rx, ry),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(rx + pix/2, ry + pix*0.5, f"{pix:.0f} m",
            ha="center", fontsize=9, color="darkred",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    # Panel C — zoom allineamento PA + donor
    ax = axes[2]
    zw  = pix * zoom_size_pix_panel_c     # default 30 px → 900m
    z_minx, z_maxx = pa_cx - zw/2, pa_cx + zw/2
    z_miny, z_maxy = pa_cy - zw/2, pa_cy + zw/2

    # Numpy boolean masks (veloce, no geopandas)
    pa_mask = ((proj_xs  >= z_minx) & (proj_xs  <= z_maxx) &
               (proj_ys  >= z_miny) & (proj_ys  <= z_maxy))
    d_mask  = ((donor_xs >= z_minx) & (donor_xs <= z_maxx) &
               (donor_ys >= z_miny) & (donor_ys <= z_maxy))

    if d_mask.any():
        d_w = np.full(d_mask.sum(), pix); d_h = d_w.copy()
        d_rects = _build_rect_patches(d_x0[d_mask], d_y0[d_mask], d_w, d_h)
        ax.add_collection(PatchCollection(d_rects, facecolor="#5b9bd5",
                                            edgecolor="navy", linewidth=0.4, alpha=0.55))
    if pa_mask.any():
        p_w = np.full(pa_mask.sum(), pix); p_h = p_w.copy()
        p_rects = _build_rect_patches(p_x0[pa_mask], p_y0[pa_mask], p_w, p_h)
        ax.add_collection(PatchCollection(p_rects, facecolor="tomato",
                                            edgecolor="darkred", linewidth=0.4, alpha=0.80))

    # Grid lines ausiliarie ogni pixel (mostrano l'allineamento)
    for xg in np.arange(z_minx, z_maxx + pix, pix):
        ax.axvline(xg, color="gray", lw=0.25, alpha=0.4)
    for yg in np.arange(z_miny, z_maxy + pix, pix):
        ax.axhline(yg, color="gray", lw=0.25, alpha=0.4)

    ax.set_xlim(z_minx, z_maxx); ax.set_ylim(z_miny, z_maxy)
    ax.set(xlabel=f"Easting UTM ({crs_utm})",
           title=f"Zoom allineamento {zw:.0f}×{zw:.0f} m\n"
                 f"PA ({pa_mask.sum()}) + Donor ({d_mask.sum()}) — stesso grid")
    ax.set_aspect("equal")
    ph = mpatches.Patch(facecolor="tomato",  edgecolor="darkred", label="PA")
    dh = mpatches.Patch(facecolor="#5b9bd5", edgecolor="navy",    label="Donor")
    ax.legend(handles=[ph, dh], fontsize=9, loc="upper right")

    plt.suptitle(
        f"GS STARR — Pixel grid allineato all'UTM | {crs_utm} | {pix:.0f} m\n"
        f"{snap_msg}  |  RUN_ID: {run_id}", fontsize=10)
    plt.tight_layout()
    print(f"    [3] Fig1 ({len(proj_df):,} PA boxes + zoom): {time.time()-t0:.2f}s")
    if out_dir:
        fig1.savefig(Path(out_dir) / "pixel_grids_aligned.png", dpi=160, bbox_inches="tight")
        print(f"        → pixel_grids_aligned.png")

    # ── (5) FIGURE 2: dettaglio UTM con coordinate ───────────────────
    t0 = time.time()
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    zw2 = pix * zoom_size_pix_fig2
    z2_minx, z2_maxx = pa_cx - zw2/2, pa_cx + zw2/2
    z2_miny, z2_maxy = pa_cy - zw2/2, pa_cy + zw2/2

    pa2 = ((proj_xs  >= z2_minx) & (proj_xs  <= z2_maxx) &
           (proj_ys  >= z2_miny) & (proj_ys  <= z2_maxy))
    d2  = ((donor_xs >= z2_minx) & (donor_xs <= z2_maxx) &
           (donor_ys >= z2_miny) & (donor_ys <= z2_maxy))

    if d2.any():
        w = np.full(d2.sum(), pix)
        d_rects = _build_rect_patches(d_x0[d2], d_y0[d2], w, w)
        ax2.add_collection(PatchCollection(d_rects, facecolor="#5b9bd5",
                                            edgecolor="navy", linewidth=0.5, alpha=0.55))
    if pa2.any():
        w = np.full(pa2.sum(), pix)
        p_rects = _build_rect_patches(p_x0[pa2], p_y0[pa2], w, w)
        ax2.add_collection(PatchCollection(p_rects, facecolor="tomato",
                                            edgecolor="darkred", linewidth=0.6, alpha=0.85))

    # Grid lines ogni pixel
    for xg in np.arange(z2_minx, z2_maxx + pix, pix):
        ax2.axvline(xg, color="gray", lw=0.4, alpha=0.5)
    for yg in np.arange(z2_miny, z2_maxy + pix, pix):
        ax2.axhline(yg, color="gray", lw=0.4, alpha=0.5)

    ax2.set_xlim(z2_minx, z2_maxx); ax2.set_ylim(z2_miny, z2_maxy)
    ax2.set(xlabel=f"Easting UTM ({crs_utm}) — m", ylabel="Northing UTM — m",
            title=f"Dettaglio griglia UTM (zoom {zw2:.0f}×{zw2:.0f} m)\n"
                  f"Celle {pix:.0f}×{pix:.0f} m | grid aux ogni {pix:.0f} m")
    ax2.set_aspect("equal")

    ax2.text(0.02, 0.98,
             f"PA       : {len(proj_df):,} px  ({pa2.sum()} in zoom)\n"
             f"Donor    : {len(donor_df):,} px  ({d2.sum()} in zoom)\n"
             f"CRS      : {crs_utm}\n"
             f"Pixel    : {pix:.0f} m\n"
             f"Snap     : {snap_msg}\n"
             f"Origin X : {origin_x:.2f}\n" if origin_x is not None else
             f"PA       : {len(proj_df):,} px\nDonor    : {len(donor_df):,} px\n"
             f"CRS      : {crs_utm}\nPixel    : {pix:.0f} m",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))
    ph2 = mpatches.Patch(facecolor="tomato",  edgecolor="darkred", label="PA")
    dh2 = mpatches.Patch(facecolor="#5b9bd5", edgecolor="navy",    label="Donor")
    ax2.legend(handles=[ph2, dh2], fontsize=9, loc="upper right")
    plt.tight_layout()
    print(f"    [4] Fig2 ({pa2.sum()+d2.sum()} boxes in zoom): {time.time()-t0:.2f}s")

    if out_dir:
        fig2.savefig(Path(out_dir) / "pixel_grid_detail_UTM.png", dpi=160, bbox_inches="tight")
        print(f"        → pixel_grid_detail_UTM.png")

    print(f"    TOTAL grid plotting: {time.time()-t_total:.2f}s")
    return fig1, fig2


# ── PLOT STANDARD COVARIATE ───────────────────────────────────────────

def plot_covariate_distributions(proj_df, donor_df, cont_covs, out_dir=None):
    n, ncols = len(cont_covs), 4
    nrows = (n + ncols - 1) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.5, nrows*3.5))
    axes = axes.flatten()
    for i, cov in enumerate(cont_covs):
        ax = axes[i]
        if cov not in proj_df.columns or cov not in donor_df.columns:
            ax.axis("off"); continue
        pv = proj_df[cov].dropna(); dv = donor_df[cov].dropna()
        lo = min(pv.quantile(0.01), dv.quantile(0.01))
        hi = max(pv.quantile(0.99), dv.quantile(0.99))
        bins = np.linspace(lo, hi, 40)
        ax.hist(dv, bins=bins, alpha=0.5, color="steelblue", density=True,
                label=f"Donor ({len(dv):,})")
        ax.hist(pv, bins=bins, alpha=0.7, color="tomato", density=True,
                label=f"Project ({len(pv):,})")
        ax.axvline(pv.median(), color="darkred", lw=1.5, ls="--", alpha=0.7)
        ax.axvline(dv.median(), color="navy",    lw=1.5, ls="--", alpha=0.7)
        ax.set_title(cov, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    for j in range(len(cont_covs), len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"Distribuzioni covariate — Project vs Donor\n{RUN_ID}", fontsize=11)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "covariate_distributions.png", dpi=150, bbox_inches="tight")
    return fig


def plot_ndvi_valid_years(proj_df, donor_df, out_dir=None):
    if "ndvi_valid_years" not in proj_df.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, df, label, color in [
        (axes[0], proj_df,  "Project", "tomato"),
        (axes[1], donor_df, "Donor",   "steelblue"),
    ]:
        cnt  = df["ndvi_valid_years"].value_counts().sort_index()
        bars = ax.bar(cnt.index.astype(str), cnt.values, color=color, edgecolor="white", width=0.6)
        ax.set(xlabel="Anni NDVI validi", ylabel="Pixel",
               title=f"{label}\nDistribuzione anni NDVI validi")
        ax.grid(alpha=0.3, axis="y")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(cnt.values)*0.01,
                    f"{int(b.get_height()):,}", ha="center", fontsize=8)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "ndvi_valid_years.png", dpi=150, bbox_inches="tight")
    return fig


# ── MAIN ─────────────────────────────────────────────────────────────

def run_extraction(base_dirs=None, output_dir=None, verbose=True):
    """
    Restituisce (proj_df, donor_df, meta, out_dir).
    meta["_grid_figs"] contiene le figure delle griglie (tuple di 2 Fig).
    """
    if base_dirs is None:
        base_dirs = [Path(p) for p in BASE_DIR_CANDIDATES if Path(p).exists()]
    if not base_dirs:
        raise FileNotFoundError(f"Nessuna directory trovata: {BASE_DIR_CANDIDATES}")
    base_dirs = [Path(b) for b in base_dirs]

    out_dir = (Path(output_dir) if output_dir
               else base_dirs[0] / "STARR_outputs" / RUN_ID / "01_extract")
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 01 — Raster Extraction | {RUN_ID}")
        print(f"Output: {out_dir}")
        print(f"{'='*60}")

    print("\n[1] Ricerca tiles...")
    proj_tiles  = find_tif_tiles(base_dirs, PROJECT_TIF_PATTERN, "Project")
    donor_tiles = find_tif_tiles(base_dirs, DONOR_TIF_PATTERN,   "Donor")

    print("\n[2] Estrazione PROJECT...")
    (proj_df, band_names, ndvi_year_cols, year_list,
     t0_year, cont_covs, crs_src, pixel_size, proj_transform) = \
        load_raster_to_dataframe(proj_tiles, "Project")

    print("\n[3] Estrazione DONOR...")
    (donor_df, *_, donor_transform) = load_raster_to_dataframe(donor_tiles, "Donor")

    tr_dict = None
    if proj_transform is not None:
        tr_dict = {"a": proj_transform.a, "b": proj_transform.b,
                   "c": proj_transform.c, "d": proj_transform.d,
                   "e": proj_transform.e, "f": proj_transform.f}

    meta = {
        "run_id":                RUN_ID,
        "project_name":          PROJECT_NAME,
        "band_names":            band_names,
        "ndvi_year_cols":        ndvi_year_cols,
        "year_list":             year_list,
        "t0_year":               t0_year,
        "trend_years":           len(year_list),
        "continuous_covariates": cont_covs,
        "output_format":         OUTPUT_FORMAT,
        "crs_src":               crs_src,
        "pixel_size_m":          pixel_size,
        "raster_transform":      tr_dict,
    }

    print("\n[4] Salvataggio...")
    p1 = save_df(proj_df,  out_dir / f"project_pixels_raw.{OUTPUT_FORMAT}")
    p2 = save_df(donor_df, out_dir / f"donor_pixels_raw.{OUTPUT_FORMAT}")

    report = {**{k: v for k, v in meta.items() if k != "raster_transform"},
              "project_n":       int(len(proj_df)),
              "donor_n":         int(len(donor_df)),
              "raster_transform": tr_dict,
              "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
              "output_dir":      str(out_dir)}
    with open(out_dir / "extraction_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n[5] Griglie pixel allineate (vettorializzato)...")
    grid_figs = plot_aligned_pixel_grids(proj_df, donor_df, meta, out_dir=out_dir)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Project   : {len(proj_df):,} pixel  → {p1.name}")
        print(f"  Donor     : {len(donor_df):,} pixel  → {p2.name}")
        print(f"  T0 (auto) : {t0_year}")
        print(f"  NDVI anni : {year_list}")
        print(f"  Covariate : {cont_covs}")
        print(f"  CRS       : {crs_src} | pixel {pixel_size:.0f} m")
        print(f"  Output    : {out_dir}")
        print(f"{'='*60}")

    meta["_grid_figs"] = grid_figs
    return proj_df, donor_df, meta, out_dir


def main():
    run_extraction()


if __name__ == "__main__":
    main()