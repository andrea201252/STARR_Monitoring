# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB  |  01_STARR_raster_extract.py
STEP 01 — Raster Extraction + NATIVE Raster Pixel Cells
============================================================

PARAMETRI MANUALI: RUN_ID, PROJECT_NAME, BASE_DIR_CANDIDATES, BLOCK_HEIGHT.

Logica corretta di campionamento:
  - I pixel non vengono ricostruiti da lon/lat e poi "snappati".
  - La griglia viene letta direttamente dal raster: row, col, transform, bounds.
  - Ogni record del dataframe è una cella raster valida già clipped dal mask export PA/DP.
  - Ogni cella mantiene footprint 30×30 m, row/col, pixel_id e centroide.
  - I valori delle bande sono estratti dalla stessa cella raster.
  - lon/lat sono le coordinate WGS84 del centroide della cella, non geometrie.
  - Rendering con matplotlib PatchCollection (C-level), non geopandas.plot.

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


def _is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except ImportError:
        return False


if not _is_notebook():
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


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
BLOCK_HEIGHT        = 2048   # block-read obbligatorio: evita lettura intero donor raster in RAM

# Vector export of clipped raster pixel grids.
# DISABLED BY DEFAULT. Full or sampled GPKG/SHP generation can dominate RAM because
# every pixel becomes a Shapely polygon. Use 01b_STARR_export_grid_vectors.py
# after Step 01 for sampled audit vectors.
EXPORT_GRID_VECTORS        = False   # default OFF: GPKG/SHP grids are generated only by the standalone exporter
EXPORT_GRID_GPKG           = True
EXPORT_GRID_SHP            = False   # SHP is heavy and field-limited; enable only for small sampled exports
EXPORT_PROJECT_GRID_FULL   = False
EXPORT_DONOR_GRID_SAMPLE   = False
EXPORT_DONOR_GRID_SAMPLE_N = 25_000
EXPORT_DONOR_GRID_FULL     = False
SHAPEFILE_MAX_FEATURES     = 50_000
GRID_VECTOR_DIRNAME        = "grid_vectors"

# Plot guards. Large PatchCollections can silently produce blank PNGs in
# Colab/Jupyter backends. Overview panels are therefore sampled, while the
# cell-by-cell detail panel remains true raster footprints.
PLOT_MAX_DONOR_OVERVIEW_CELLS   = 50_000
PLOT_MAX_DONOR_COMPLETE_CELLS   = 120_000  # visualization cap only; full donor extent is always used
PLOT_MAX_PROJECT_OVERVIEW_CELLS = 80_000
PLOT_MAX_PROJECT_FULL_CELLS     = 80_000
PLOT_DETAIL_ZOOM_PIXELS         = 40


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

def _cell_bounds_from_transform(transform, rows, cols, pix):
    """
    Bounds nativi delle celle raster.
    Per raster north-up senza rotazione usa il transform esatto.
    Fallback: centro ± pixel/2.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    if abs(float(transform.b)) < 1e-12 and abs(float(transform.d)) < 1e-12:
        x_left  = transform.c + cols * transform.a
        x_right = x_left + transform.a
        y_top   = transform.f + rows * transform.e
        y_bot   = y_top + transform.e
        xmin = np.minimum(x_left, x_right).astype(np.float64)
        xmax = np.maximum(x_left, x_right).astype(np.float64)
        ymin = np.minimum(y_top, y_bot).astype(np.float64)
        ymax = np.maximum(y_top, y_bot).astype(np.float64)
        xcen = (xmin + xmax) / 2.0
        ycen = (ymin + ymax) / 2.0
        return xcen, ycen, xmin, ymin, xmax, ymax

    # Rotated rasters: not expected for these GEE exports.
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xcen = np.asarray(xs, dtype=np.float64)
    ycen = np.asarray(ys, dtype=np.float64)
    half = pix / 2.0
    return xcen, ycen, xcen - half, ycen - half, xcen + half, ycen + half


def _process_block(data, transform, transformer, band_names, row_offset, tile_name=""):
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
        return pd.DataFrame(columns=band_names + [
            "lon", "lat", "x_utm", "y_utm", "centroid_x_utm", "centroid_y_utm",
            "centroid_lon", "centroid_lat", "grid_row", "grid_col", "pixel_id",
            "cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax", "cell_width_m",
            "cell_height_m", "source_tile"
        ])

    rows, cols  = np.where(valid_2d)
    global_rows = rows + row_offset
    pix = float(abs(transform.a))

    # La griglia vera è questa: row/col + transform del raster.
    # Non si ricostruisce la griglia da lon/lat.
    xs, ys, xmin, ymin, xmax, ymax = _cell_bounds_from_transform(transform, global_rows, cols, pix)
    lons, lats = transformer.transform(xs, ys)

    df = pd.DataFrame(data.data[:, rows, cols].T, columns=band_names)
    df["lon"] = np.asarray(lons, dtype=np.float64)
    df["lat"] = np.asarray(lats, dtype=np.float64)
    df["x_utm"] = xs.astype(np.float64)
    df["y_utm"] = ys.astype(np.float64)
    df["centroid_x_utm"] = df["x_utm"]
    df["centroid_y_utm"] = df["y_utm"]
    df["centroid_lon"] = df["lon"]
    df["centroid_lat"] = df["lat"]
    df["grid_row"] = global_rows.astype(np.int64)
    df["grid_col"] = cols.astype(np.int64)
    df["pixel_id"] = [f"{tile_name}::r{int(r)}::c{int(c)}" for r, c in zip(global_rows, cols)]
    df["cell_xmin"] = xmin.astype(np.float64)
    df["cell_ymin"] = ymin.astype(np.float64)
    df["cell_xmax"] = xmax.astype(np.float64)
    df["cell_ymax"] = ymax.astype(np.float64)
    df["cell_width_m"] = (df["cell_xmax"] - df["cell_xmin"]).astype(np.float64)
    df["cell_height_m"] = (df["cell_ymax"] - df["cell_ymin"]).astype(np.float64)
    df["source_tile"] = tile_name
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
            dfs  = [_process_block(data, transform, transformer, band_names, 0, tif_path.name)]
        else:
            dfs = []
            for r0 in range(0, src.height, BLOCK_HEIGHT):
                h   = min(BLOCK_HEIGHT, src.height - r0)
                win = Window(0, r0, src.width, h)
                dfs.append(_process_block(src.read(window=win, masked=True),
                                          transform, transformer, band_names, r0, tif_path.name))

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
    merged = merged.drop_duplicates(subset=["x_utm", "y_utm"]).reset_index(drop=True)
    if before > len(merged):
        print(f"    Rimossi {before-len(merged):,} duplicati x_utm/y_utm")

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


# ── VECTOR EXPORT: GRID CELLS ────────────────────────────────────────

def _safe_remove_vector(path):
    """Remove an existing vector file. Handles shapefile sidecars."""
    path = Path(path)
    if path.suffix.lower() == ".shp":
        stem = path.with_suffix("")
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
            q = stem.with_suffix(ext)
            if q.exists():
                q.unlink()
    elif path.exists() and path.is_file():
        path.unlink()


def _grid_cells_gdf(df, crs_utm, label, max_features=None, random_state=42):
    """
    Build a GeoDataFrame of the *actual raster cell footprints*.
    Geometry is created from cell_xmin/cell_ymin/cell_xmax/cell_ymax.
    Raster values remain in parquet; vector export keeps only audit attributes.
    """
    required = {"cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax"}
    if not required.issubset(df.columns):
        raise ValueError(f"{label}: missing native cell bounds: {sorted(required - set(df.columns))}")

    work = df
    sampled = False
    if max_features is not None and len(work) > int(max_features):
        work = work.sample(int(max_features), random_state=random_state).reset_index(drop=True)
        sampled = True
    else:
        work = work.reset_index(drop=True)

    try:
        import geopandas as gpd
        from shapely.geometry import box
    except Exception as exc:
        raise ImportError(
            "geopandas/shapely are required to export GPKG/SHP grid cells. "
            "Install them in the notebook environment before running Step 01."
        ) from exc

    geoms = [box(float(x0), float(y0), float(x1), float(y1))
             for x0, y0, x1, y1 in zip(work["cell_xmin"], work["cell_ymin"], work["cell_xmax"], work["cell_ymax"])]

    keep = [
        "pixel_id", "source_tile", "grid_row", "grid_col",
        "x_utm", "y_utm", "centroid_x_utm", "centroid_y_utm",
        "lon", "lat", "centroid_lon", "centroid_lat",
        "cell_width_m", "cell_height_m", "pixel_area_ha",
        "WRB2_CODE", "ndvi_valid_years"
    ]
    attrs = {c: work[c].values for c in keep if c in work.columns}
    attrs["grid_label"] = label
    attrs["is_sample"] = bool(sampled)
    attrs["n_source"] = int(len(df))

    gdf = gpd.GeoDataFrame(attrs, geometry=geoms, crs=crs_utm)
    return gdf


def export_grid_vectors(proj_df, donor_df, crs_utm, out_dir):
    """
    Exports vector grids for audit and visual inspection.

    Defaults:
      - full PA grid: GPKG + SHP
      - donor sampled grid: GPKG + SHP
      - full donor grid: only if EXPORT_DONOR_GRID_FULL=True, GPKG only recommended

    Full donor SHP is deliberately skipped above SHAPEFILE_MAX_FEATURES.
    """
    if not EXPORT_GRID_VECTORS:
        return {}

    out_dir = Path(out_dir) / GRID_VECTOR_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    def _write(gdf, stem, write_shp=True):
        written = {}
        if EXPORT_GRID_GPKG:
            gpkg = out_dir / f"{stem}.gpkg"
            _safe_remove_vector(gpkg)
            gdf.to_file(gpkg, driver="GPKG", layer=stem)
            written["gpkg"] = str(gpkg)
        if EXPORT_GRID_SHP and write_shp:
            shp = out_dir / f"{stem}.shp"
            _safe_remove_vector(shp)
            gdf.to_file(shp, driver="ESRI Shapefile")
            written["shp"] = str(shp)
        return written

    if EXPORT_PROJECT_GRID_FULL:
        print("    Export vector grid: project_grid_cells (full)")
        gdf = _grid_cells_gdf(proj_df, crs_utm, "project", max_features=None)
        outputs["project_grid_cells"] = _write(
            gdf, "project_grid_cells", write_shp=(len(gdf) <= SHAPEFILE_MAX_FEATURES)
        )

    if EXPORT_DONOR_GRID_SAMPLE:
        n = min(int(EXPORT_DONOR_GRID_SAMPLE_N), len(donor_df))
        print(f"    Export vector grid: donor_grid_cells_sample ({n:,}/{len(donor_df):,})")
        gdf = _grid_cells_gdf(donor_df, crs_utm, "donor_sample", max_features=n)
        outputs["donor_grid_cells_sample"] = _write(
            gdf, "donor_grid_cells_sample", write_shp=(len(gdf) <= SHAPEFILE_MAX_FEATURES)
        )

    if EXPORT_DONOR_GRID_FULL:
        print("    Export vector grid: donor_grid_cells_FULL (large file, GPKG only recommended)")
        gdf = _grid_cells_gdf(donor_df, crs_utm, "donor_full", max_features=None)
        outputs["donor_grid_cells_FULL"] = _write(
            gdf, "donor_grid_cells_FULL", write_shp=(len(gdf) <= SHAPEFILE_MAX_FEATURES)
        )

    manifest = {
        "description": "Vector exports of native raster pixel footprints clipped by Project Area or Donor Pool masks.",
        "crs": str(crs_utm),
        "project_features": int(len(proj_df)),
        "donor_features_total": int(len(donor_df)),
        "donor_sample_features": int(min(int(EXPORT_DONOR_GRID_SAMPLE_N), len(donor_df))) if EXPORT_DONOR_GRID_SAMPLE else 0,
        "full_donor_export_enabled": bool(EXPORT_DONOR_GRID_FULL),
        "outputs": outputs,
        "note": "Raster values remain in project_pixels_raw/donor_pixels_raw parquet. Vector files contain audit geometry and key identifiers only."
    }
    with open(out_dir / "grid_vector_export_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    outputs["manifest"] = str(out_dir / "grid_vector_export_manifest.json")
    return outputs


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
                              zoom_size_pix_panel_c=None,
                              zoom_size_pix_fig2=None):
    """
    Diagnostic plot of native raster cells.

    Why previous images could be empty:
      - too many rectangles were pushed to one PatchCollection;
      - matplotlib/Colab sometimes writes a blank PNG under memory pressure;
      - full-PA panels at 30 m are visually saturated anyway.

    This version:
      - never reconstructs grid cells from lon/lat;
      - plots sampled footprints for overview panels;
      - plots true cell footprints for local details;
      - saves with explicit canvas draw and white facecolor.
    """
    t_total = time.time()
    crs_utm = meta.get("crs_src", "EPSG:32734")
    pix = float(meta.get("pixel_size_m", 30.0))
    run_id = meta.get("run_id", "")
    zoom_size_pix_panel_c = int(zoom_size_pix_panel_c or PLOT_DETAIL_ZOOM_PIXELS)
    zoom_size_pix_fig2 = int(zoom_size_pix_fig2 or PLOT_DETAIL_ZOOM_PIXELS)

    required = {"x_utm", "y_utm", "cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax"}
    miss_p = sorted(required - set(proj_df.columns))
    miss_d = sorted(required - set(donor_df.columns))
    if miss_p or miss_d:
        raise ValueError(
            "Step 01 GRID_NATIVE richiede colonne native. "
            f"Project missing={miss_p}; Donor missing={miss_d}. "
            "Rigenerare project_pixels_raw e donor_pixels_raw con questo script."
        )

    def arr(df, col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    proj_xs, proj_ys = arr(proj_df, "x_utm"), arr(proj_df, "y_utm")
    donor_xs, donor_ys = arr(donor_df, "x_utm"), arr(donor_df, "y_utm")
    p_x0, p_y0 = arr(proj_df, "cell_xmin"), arr(proj_df, "cell_ymin")
    p_x1, p_y1 = arr(proj_df, "cell_xmax"), arr(proj_df, "cell_ymax")
    d_x0, d_y0 = arr(donor_df, "cell_xmin"), arr(donor_df, "cell_ymin")
    d_x1, d_y1 = arr(donor_df, "cell_xmax"), arr(donor_df, "cell_ymax")
    p_w, p_h = p_x1 - p_x0, p_y1 - p_y0
    d_w, d_h = d_x1 - d_x0, d_y1 - d_y0

    if len(proj_df) == 0 or len(donor_df) == 0:
        raise RuntimeError("Project o donor vuoti: impossibile creare diagnostica griglia.")

    rng = np.random.default_rng(42)

    def sample_indices(n, max_n):
        if n <= int(max_n):
            return np.arange(n, dtype=np.int64), False
        return rng.choice(n, int(max_n), replace=False), True

    def nearest_real_center(xs, ys):
        mx, my = float(np.nanmedian(xs)), float(np.nanmedian(ys))
        idx = int(np.nanargmin((xs - mx) ** 2 + (ys - my) ** 2))
        return float(xs[idx]), float(ys[idx]), idx

    def add_patch_cells(ax, x0, y0, w, h, idx, face, edge, alpha=0.70, lw=0.10, label=None):
        if len(idx) == 0:
            return 0
        rects = _build_rect_patches(x0[idx], y0[idx], w[idx], h[idx])
        coll = PatchCollection(rects, facecolor=face, edgecolor=edge, linewidth=lw,
                               alpha=alpha, label=label)
        ax.add_collection(coll, autolim=True)
        ax.update_datalim(np.column_stack([np.r_[x0[idx], x0[idx] + w[idx]],
                                           np.r_[y0[idx], y0[idx] + h[idx]]]))
        return len(rects)

    def set_limits_from_arrays(ax, xs0, ys0, xs1, ys1, margin_frac=0.04):
        minx, maxx = float(np.nanmin(xs0)), float(np.nanmax(xs1))
        miny, maxy = float(np.nanmin(ys0)), float(np.nanmax(ys1))
        mx = max((maxx - minx) * margin_frac, pix * 3)
        my = max((maxy - miny) * margin_frac, pix * 3)
        ax.set_xlim(minx - mx, maxx + mx)
        ax.set_ylim(miny - my, maxy + my)

    def draw_detail(ax, xs, ys, x0, y0, w, h, center, zoom_pix, face, edge, label):
        zw = pix * int(zoom_pix)
        cx, cy = center
        xmin, xmax = cx - zw / 2.0, cx + zw / 2.0
        ymin, ymax = cy - zw / 2.0, cy + zw / 2.0
        mask = ((xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax))
        idx = np.where(mask)[0]
        if len(idx) == 0:
            # Fallback: force at least the nearest real cell, preventing blank diagnostic images.
            near = int(np.nanargmin((xs - cx) ** 2 + (ys - cy) ** 2))
            idx = np.array([near], dtype=np.int64)
            cx, cy = float(xs[near]), float(ys[near])
            xmin, xmax = cx - zw / 2.0, cx + zw / 2.0
            ymin, ymax = cy - zw / 2.0, cy + zw / 2.0
        add_patch_cells(ax, x0, y0, w, h, idx, face, edge, alpha=0.78, lw=0.45)
        vx = np.unique(np.r_[x0[idx], x0[idx] + w[idx]])
        vy = np.unique(np.r_[y0[idx], y0[idx] + h[idx]])
        vx = vx[(vx >= xmin) & (vx <= xmax)]
        vy = vy[(vy >= ymin) & (vy <= ymax)]
        for xg in vx:
            ax.axvline(float(xg), color="black", lw=0.25, alpha=0.35, zorder=0)
        for yg in vy:
            ax.axhline(float(yg), color="black", lw=0.25, alpha=0.35, zorder=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{label} — true raster cells\n{len(idx):,} cells in frame | zoom {zw:.0f}×{zw:.0f} m")
        ax.set_xlabel(f"Easting UTM ({crs_utm}) — m")
        ax.set_ylabel("Northing UTM — m")
        ax.grid(False)
        return int(len(idx))

    pa_cx, pa_cy, _ = nearest_real_center(proj_xs, proj_ys)
    do_cx, do_cy, _ = nearest_real_center(donor_xs, donor_ys)

    # FIGURE 1: overview + PA full/sample + PA local detail.
    t0 = time.time()
    fig1, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")

    p_idx_over, p_sampled = sample_indices(len(proj_df), PLOT_MAX_PROJECT_OVERVIEW_CELLS)
    d_idx_over, d_sampled = sample_indices(len(donor_df), PLOT_MAX_DONOR_OVERVIEW_CELLS)

    ax = axes[0]
    add_patch_cells(ax, d_x0, d_y0, d_w, d_h, d_idx_over,
                    "steelblue", "steelblue", alpha=0.18, lw=0.03,
                    label=f"Donor cells {'sample' if d_sampled else 'full'} ({len(d_idx_over):,}/{len(donor_df):,})")
    add_patch_cells(ax, p_x0, p_y0, p_w, p_h, p_idx_over,
                    "tomato", "darkred", alpha=0.70, lw=0.04,
                    label=f"PA cells {'sample' if p_sampled else 'full'} ({len(p_idx_over):,}/{len(proj_df):,})")
    set_limits_from_arrays(
        ax,
        np.r_[d_x0[d_idx_over], p_x0[p_idx_over]],
        np.r_[d_y0[d_idx_over], p_y0[p_idx_over]],
        np.r_[d_x1[d_idx_over], p_x1[p_idx_over]],
        np.r_[d_y1[d_idx_over], p_y1[p_idx_over]],
    )
    ax.set(xlabel=f"Easting UTM ({crs_utm}) — m", ylabel="Northing UTM — m",
           title="Overview native raster cells\n(sampled for rendering stability; geometry remains native)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.20)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    p_idx_full, p_full_sampled = sample_indices(len(proj_df), PLOT_MAX_PROJECT_FULL_CELLS)
    add_patch_cells(ax, p_x0, p_y0, p_w, p_h, p_idx_full,
                    "tomato", "darkred", alpha=0.75, lw=0.08)
    set_limits_from_arrays(ax, p_x0[p_idx_full], p_y0[p_idx_full], p_x1[p_idx_full], p_y1[p_idx_full])
    ax.set(xlabel=f"Easting UTM ({crs_utm})", ylabel="Northing UTM",
           title=f"PA clipped raster cells — {'sample' if p_full_sampled else 'full'}\n"
                 f"{len(p_idx_full):,}/{len(proj_df):,} cells shown | {pix:.0f}×{pix:.0f} m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    pa_zoom_n = draw_detail(axes[2], proj_xs, proj_ys, p_x0, p_y0, p_w, p_h,
                            (pa_cx, pa_cy), zoom_size_pix_panel_c,
                            "tomato", "darkred", "PA local detail")

    fig1.suptitle(
        f"GS STARR — Native raster pixel grid | {crs_utm} | {pix:.0f} m\n"
        f"Displayed cells are the raster cells used for value extraction | RUN_ID: {run_id}",
        fontsize=10)
    fig1.tight_layout()
    fig1.canvas.draw()
    print(f"    [2] Fig1 native-grid diagnostic: {time.time()-t0:.2f}s")
    if out_dir:
        fig1.savefig(Path(out_dir) / "pixel_grids_aligned.png", dpi=160,
                     bbox_inches="tight", facecolor="white")
        print("        → pixel_grids_aligned.png")

    # FIGURE 2: PA local detail + COMPLETE DONOR GRID OVERVIEW.
    # No donor super-zoom: donor is visualised across its full clipped extent.
    # For million-cell donor pools, the display is sampled to avoid blank PNG/OOM,
    # but axis limits and counts describe the complete donor grid.
    t0 = time.time()
    fig2, axes2 = plt.subplots(1, 2, figsize=(17, 8), facecolor="white")
    pa2_n = draw_detail(axes2[0], proj_xs, proj_ys, p_x0, p_y0, p_w, p_h,
                         (pa_cx, pa_cy), zoom_size_pix_fig2,
                         "tomato", "darkred", "PA local pixel detail")

    d_idx_complete, d_complete_sampled = sample_indices(len(donor_df), PLOT_MAX_DONOR_COMPLETE_CELLS)
    add_patch_cells(axes2[1], d_x0, d_y0, d_w, d_h, d_idx_complete,
                    "#5b9bd5", "navy", alpha=0.22, lw=0.02,
                    label=f"Donor cells {'sample' if d_complete_sampled else 'full'} ({len(d_idx_complete):,}/{len(donor_df):,})")
    # Use the full donor extent for the plot, not the sampled extent.
    set_limits_from_arrays(axes2[1], d_x0, d_y0, d_x1, d_y1, margin_frac=0.02)
    axes2[1].set_aspect("equal", adjustable="box")
    axes2[1].set_xlabel(f"Easting UTM ({crs_utm}) — m")
    axes2[1].set_ylabel("Northing UTM — m")
    axes2[1].set_title(
        f"Donor complete clipped raster grid extent\n"
        f"{len(d_idx_complete):,}/{len(donor_df):,} cells drawn | full extent shown | {pix:.0f}×{pix:.0f} m"
    )
    axes2[1].legend(fontsize=8)
    axes2[1].grid(alpha=0.18)

    info = (
        f"PA valid cells     : {len(proj_df):,} | local frame: {pa2_n:,}\n"
        f"Donor valid cells  : {len(donor_df):,} | complete extent shown\n"
        f"Donor drawn cells  : {len(d_idx_complete):,} ({'sampled' if d_complete_sampled else 'full'})\n"
        f"CRS                : {crs_utm}\n"
        f"Pixel              : {pix:.0f} m\n"
        f"Grid source        : raster row/col + transform\n"
        f"Value source       : same raster cells\n"
        f"Vector export      : disabled in Step 01"
    )
    axes2[0].text(0.02, 0.98, info, transform=axes2[0].transAxes,
                  fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))
    fig2.suptitle(
        "Native raster grid diagnostic — no donor super-zoom; donor shown over complete clipped extent.",
        fontsize=10)
    fig2.tight_layout()
    fig2.canvas.draw()
    print(f"    [3] Fig2 PA detail + donor complete overview: {time.time()-t0:.2f}s")
    if out_dir:
        fig2.savefig(Path(out_dir) / "pixel_grid_detail_UTM.png", dpi=160,
                     bbox_inches="tight", facecolor="white")
        fig2.savefig(Path(out_dir) / "donor_grid_complete_UTM.png", dpi=160,
                     bbox_inches="tight", facecolor="white")
        print("        → pixel_grid_detail_UTM.png")
        print("        → donor_grid_complete_UTM.png")

    print(f"    TOTAL native grid plotting: {time.time()-t_total:.2f}s")
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

    if EXPORT_GRID_VECTORS:
        print("\n[4b] Export GPKG/SHP griglie raster clipped...")
        vector_grid_outputs = export_grid_vectors(proj_df, donor_df, crs_src, out_dir)
    else:
        print("\n[4b] Export GPKG/SHP griglie raster clipped: SKIPPED (default low-RAM).")
        vector_grid_outputs = {
            "enabled": False,
            "reason": "disabled_by_default_to_prevent_RAM_saturation",
            "standalone_exporter": "01b_STARR_export_grid_vectors.py"
        }

    # Native grid audit: these are raster cells, not reconstructed vector grids.
    grid_audit = {
        "grid_logic": "native_raster_row_col_transform",
        "project_grid_cells_n": int(len(proj_df)),
        "donor_grid_cells_n": int(len(donor_df)),
        "grid_columns": [
            "pixel_id", "grid_row", "grid_col",
            "cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax",
            "cell_width_m", "cell_height_m",
            "centroid_x_utm", "centroid_y_utm", "centroid_lon", "centroid_lat"
        ],
        "cell_geometry_definition": "30x30m raster footprint clipped by PA/DP mask; centroid attached to each cell",
        "pixel_values_extracted_from_same_cells": True,
    }

    report = {**{k: v for k, v in meta.items() if k != "raster_transform"},
              "project_n":       int(len(proj_df)),
              "donor_n":         int(len(donor_df)),
              "raster_transform": tr_dict,
              "grid_audit":      grid_audit,
              "vector_grid_outputs": vector_grid_outputs,
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