# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB | 04_STARR_reference_area_lock.py
STEP 04 — Pixel/Block-Support Reference Area + Lock
============================================================

Main changes vs previous version:
  - The final RA is not a smoothed hull by default.
  - Matched donor pixels are converted into 30 m square support cells.
  - Output can remain MultiPolygon. This is intentional and more defensible than
    a large simplified envelope that includes many unvalidated donor pixels.
  - Optional block mode can aggregate matched pixels into larger regular blocks.
  - Monitoring points default to matched donor pixel centres. A 500 m grid over
    disconnected 30 m cells is usually not useful.

Default output:
  reference_area_FINAL.gpkg/shp/geojson
  matched_donor_pixels.gpkg
  reference_area_support_cells.gpkg
  monitoring_points_FIXED.gpkg/csv
  covariate_summary.csv
  diagnostics.png
  reference_area_FINAL_manifest.json
  GEE_load_reference_area.js
"""

import warnings
warnings.filterwarnings("ignore")

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point, box, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid


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
# USER PARAMETERS
# ================================================================

# "pixel" = each matched donor pixel becomes a 30 m support cell.
# "block" = matched donor pixels are snapped to regular BLOCK_SIZE_M cells.
RA_SUPPORT_MODE = "pixel"
PIXEL_SIZE_M = 30.0
BLOCK_SIZE_M = 150.0

# Keep these at 0 for final compliance. Increase only if you explicitly want
# a cartographic generalisation and then revalidate all pixels inside the RA.
EDGE_BUFFER_M = 0.0
CLOSE_GAPS_M = 0.0
SIMPLIFY_TOL_M = 0.0
MIN_POLYGON_HA = 0.0

# "matched_pixels" is robust for disconnected pixel/block support.
# "grid" creates a systematic grid inside the RA and may return zero points if
# the RA is a set of disconnected 30 m cells.
MONITORING_MODE = "matched_pixels"
MONITORING_SPACING_M = 500.0
MONITORING_SEED = 42

# Extra vector exports. The canonical audit format is GPKG; SHP is added for GIS interoperability.
EXPORT_SUPPORT_CELLS_SHP = True
EXPORT_MATCHED_POINTS_SHP = True
EXPORT_MONITORING_POINTS_SHP = True
EXPORT_NATIVE_MATCHED_CELLS_GPKG = True
EXPORT_NATIVE_MATCHED_CELLS_SHP = True

CRS_GEO = "EPSG:4326"


# ================================================================
# IO HELPERS
# ================================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_df(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File non trovato: {p}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Formato non supportato: {p.suffix}")


def find_file(base_dirs, stem):
    for base in base_dirs:
        base = Path(base)
        for ext in ["parquet", "csv"]:
            p = base / f"{stem}.{ext}"
            if p.exists():
                print(f"    Trovato: {p.name}")
                return p
    raise FileNotFoundError(f"'{stem}' non trovato in {base_dirs}")


def safe_remove(path):
    path = Path(path)
    if path.suffix.lower() == ".shp":
        stem = path.with_suffix("")
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
            p = stem.with_suffix(ext)
            if p.exists():
                p.unlink()
    elif path.exists() and path.is_file():
        path.unlink()


def safe_valid(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    try:
        return make_valid(geom)
    except Exception:
        return geom.buffer(0)


def polygonal_only(geom):
    if geom is None or geom.is_empty:
        return None
    geom = safe_valid(geom)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = []
        for g in geom.geoms:
            if isinstance(g, Polygon):
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend(list(g.geoms))
        if not polys:
            return None
        return safe_valid(unary_union(polys))
    return None


def geom_union(geoms):
    try:
        return geoms.union_all()
    except Exception:
        return unary_union(list(geoms))


def gdf_is_nonempty(gdf):
    return gdf is not None and len(gdf) > 0


def gdf_nrows(gdf):
    return 0 if gdf is None else int(len(gdf))


def auto_utm(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def detect_cont_covs(meta, df):
    covs = meta.get("continuous_covariates", []) if meta else []
    if covs:
        return [c for c in covs if c in df.columns]
    skip = {"lon", "lat", "ref_lon", "ref_lat", "proj_lon", "proj_lat", "pixel_area_ha", "is_project", "is_donor"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


# ================================================================
# GEODATAFRAME / SUPPORT CELLS
# ================================================================

def make_points_gdf(df):
    if {"ref_lon", "ref_lat"}.issubset(df.columns):
        lon_col, lat_col = "ref_lon", "ref_lat"
    elif {"lon", "lat"}.issubset(df.columns):
        lon_col, lat_col = "lon", "lat"
    else:
        raise ValueError("Coordinate reference non trovate. Servono ref_lon/ref_lat o lon/lat.")

    df_u = df.drop_duplicates(subset=[lon_col, lat_col]).reset_index(drop=True)

    # P2 FIX: i pixel PA senza un match donor valido portano NaN nelle coordinate
    # ref_lon/ref_lat (e nei ref_cell_*). points_from_xy + to_crs falliscono su NaN.
    # Scartiamo qui le righe non-finite, registrando quante.
    n_before = len(df_u)
    coord_ok = np.isfinite(df_u[lon_col].to_numpy()) & np.isfinite(df_u[lat_col].to_numpy())
    df_u = df_u.loc[coord_ok].reset_index(drop=True)
    n_dropped = int(n_before - len(df_u))
    if n_dropped > 0:
        print(f"    [P2] {n_dropped:,} righe senza coordinate reference valide "
              f"(pixel non matchati) scartate dalla costruzione RA.")
    if len(df_u) == 0:
        raise RuntimeError(
            "Nessun pixel con coordinate reference valide dopo il filtro NaN. "
            "Verificare che lo Step 02 abbia prodotto match validi."
        )

    crs_m = auto_utm(float(df_u[lon_col].mean()), float(df_u[lat_col].mean()))
    gdf = gpd.GeoDataFrame(
        df_u,
        geometry=gpd.points_from_xy(df_u[lon_col], df_u[lat_col]),
        crs=CRS_GEO,
    ).to_crs(crs_m)
    return gdf, df_u, crs_m, lon_col, lat_col


def _has_native_ref_bounds(gdf):
    cols = {"ref_cell_xmin", "ref_cell_ymin", "ref_cell_xmax", "ref_cell_ymax"}
    return cols.issubset(set(gdf.columns))


def build_support_cells(points_gdf, mode=RA_SUPPORT_MODE):
    if mode not in {"pixel", "block"}:
        raise ValueError("RA_SUPPORT_MODE deve essere 'pixel' o 'block'.")

    # Preferred audit mode: use native raster cell bounds propagated from Step 01 → Step 02 → Step 03.
    # This avoids rebuilding a 30 m square from a reprojected lon/lat point.
    if mode == "pixel" and _has_native_ref_bounds(points_gdf):
        geoms = []
        attrs = []
        for i, row in points_gdf.reset_index(drop=True).iterrows():
            try:
                x0 = float(row["ref_cell_xmin"])
                y0 = float(row["ref_cell_ymin"])
                x1 = float(row["ref_cell_xmax"])
                y1 = float(row["ref_cell_ymax"])
                geoms.append(box(x0, y0, x1, y1))
                attrs.append({
                    "support_id": i + 1,
                    "support_mode": "pixel_native_bounds",
                    "support_m": PIXEL_SIZE_M,
                    "ref_pixel_id": str(row.get("ref_pixel_id", "")),
                    "ref_grid_row": int(row.get("ref_grid_row", -1)) if pd.notna(row.get("ref_grid_row", np.nan)) else -1,
                    "ref_grid_col": int(row.get("ref_grid_col", -1)) if pd.notna(row.get("ref_grid_col", np.nan)) else -1,
                })
            except Exception:
                p = row.geometry
                half = PIXEL_SIZE_M / 2.0
                geoms.append(box(p.x - half, p.y - half, p.x + half, p.y + half))
                attrs.append({"support_id": i + 1, "support_mode": "pixel_centroid_fallback", "support_m": PIXEL_SIZE_M})
        support = gpd.GeoDataFrame(attrs, geometry=geoms, crs=points_gdf.crs)
        support["area_ha"] = support.geometry.area / 10000.0
        return support

    geoms = []
    attrs = []
    if mode == "pixel":
        half = PIXEL_SIZE_M / 2.0
        for i, p in enumerate(points_gdf.geometry):
            geoms.append(box(p.x - half, p.y - half, p.x + half, p.y + half))
            attrs.append({"support_id": i + 1, "support_mode": "pixel_centroid_fallback", "support_m": PIXEL_SIZE_M})
    else:
        keys = {}
        size = BLOCK_SIZE_M
        for p in points_gdf.geometry:
            x0 = np.floor(p.x / size) * size
            y0 = np.floor(p.y / size) * size
            key = (float(x0), float(y0))
            keys[key] = keys.get(key, 0) + 1
        for i, ((x0, y0), n) in enumerate(keys.items()):
            geoms.append(box(x0, y0, x0 + size, y0 + size))
            attrs.append({"support_id": i + 1, "support_mode": "block", "support_m": size, "n_matched_pts": int(n)})

    support = gpd.GeoDataFrame(attrs, geometry=geoms, crs=points_gdf.crs)
    support["area_ha"] = support.geometry.area / 10000.0
    return support

def build_reference_area(points_gdf, support_gdf, meta=None):
    geom = polygonal_only(geom_union(support_gdf.geometry))
    if geom is None or geom.is_empty:
        raise RuntimeError("Reference Area vuota dopo union dei support cells.")

    method_parts = [f"support_{RA_SUPPORT_MODE}"]
    if EDGE_BUFFER_M > 0:
        geom = polygonal_only(safe_valid(geom.buffer(EDGE_BUFFER_M)))
        method_parts.append(f"edge_buffer_{EDGE_BUFFER_M:g}m")
    if CLOSE_GAPS_M > 0:
        geom = polygonal_only(safe_valid(geom.buffer(CLOSE_GAPS_M).buffer(-CLOSE_GAPS_M)))
        method_parts.append(f"close_gaps_{CLOSE_GAPS_M:g}m")
    if SIMPLIFY_TOL_M > 0:
        geom = polygonal_only(safe_valid(geom.simplify(SIMPLIFY_TOL_M, preserve_topology=True)))
        method_parts.append(f"simplify_{SIMPLIFY_TOL_M:g}m")

    if geom is None or geom.is_empty:
        raise RuntimeError("Reference Area vuota dopo post-processing.")

    # Optional polygon size filter. Default is 0, meaning keep all valid matched supports.
    if MIN_POLYGON_HA > 0 and geom.geom_type == "MultiPolygon":
        polys = [p for p in geom.geoms if p.area / 10000.0 >= MIN_POLYGON_HA]
        if not polys:
            raise RuntimeError("Tutti i poligoni sono sotto MIN_POLYGON_HA.")
        geom = polygonal_only(unary_union(polys))
        method_parts.append(f"min_poly_{MIN_POLYGON_HA:g}ha")

    area_ha = float(geom.area / 10000.0)
    # PERF: conteggio punti-dentro-RA vettorizzato (STRtree/sjoin) invece di
    # un loop Python geom.contains(p) per ogni punto (O(N) ops shapely su un
    # MultiPolygon potenzialmente enorme → collo di bottiglia con ~150k punti).
    try:
        ra_gdf = gpd.GeoDataFrame(geometry=[geom], crs=points_gdf.crs)
        joined = gpd.sjoin(points_gdf[["geometry"]], ra_gdf,
                           how="inner", predicate="intersects")
        n_in = int(joined.index.nunique())
    except Exception:
        # Fallback robusto: usa STRtree direttamente
        try:
            from shapely import STRtree
            tree = STRtree(list(points_gdf.geometry))
            n_in = int(len(tree.query(geom, predicate="intersects")))
        except Exception:
            n_in = int(sum(1 for p in points_gdf.geometry
                           if geom.contains(p) or geom.intersects(p)))
    pct_in = float(100.0 * n_in / len(points_gdf)) if len(points_gdf) else 0.0

    meta = meta or {}
    bounds_method = "+".join(method_parts)
    out = gpd.GeoDataFrame({
        "project": [meta.get("project_name", "unknown")],
        "run_id": [meta.get("run_id", "unknown")],
        "bounds_method": [bounds_method],
        "support_mode": [RA_SUPPORT_MODE],
        "support_m": [PIXEL_SIZE_M if RA_SUPPORT_MODE == "pixel" else BLOCK_SIZE_M],
        "total_ha": [round(area_ha, 4)],
        "n_matched_pts": [n_in],
        "pct_pts_inside": [round(pct_in, 2)],
        "lock_status": ["LOCKED"],
        "validity_years": [10],
        "description": [
            "GS STARR locked Reference Area built from matched donor pixel/block support. "
            "The boundary remains fixed during the 10-year baseline validity period."
        ],
    }, geometry=[geom], crs=points_gdf.crs)
    return out, bounds_method, area_ha, n_in, pct_in


# ================================================================
# MONITORING POINTS
# ================================================================

def monitoring_points(bounds_gdf, points_gdf, meta=None):
    meta = meta or {}
    run_id = meta.get("run_id", "")
    proj_name = meta.get("project_name", "")

    if MONITORING_MODE == "matched_pixels":
        gdf = points_gdf.copy()
        gdf = gdf[["geometry"]].copy()
        gdf["point_id"] = np.arange(1, len(gdf) + 1)
        gdf["project"] = proj_name
        gdf["run_id"] = run_id
        gdf["method"] = "matched_pixel_centres"
        gdf["fixed"] = True
        gdf["description"] = "Fixed matched donor pixel centre used as monitoring support point."
        gdf = gdf.to_crs(CRS_GEO)
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        return gdf

    geom = bounds_gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx, maxx, MONITORING_SPACING_M)
    ys = np.arange(miny, maxy, MONITORING_SPACING_M)
    pts = []
    for x in xs:
        for y in ys:
            p = Point(x, y)
            if geom.contains(p):
                pts.append(p)

    if not pts:
        print("    WARNING: grid monitoring returned zero points; fallback to matched pixel centres.")
        gdf = points_gdf.copy()
        gdf = gdf[["geometry"]].copy()
        gdf["point_id"] = np.arange(1, len(gdf) + 1)
        gdf["project"] = proj_name
        gdf["run_id"] = run_id
        gdf["method"] = "matched_pixel_centres_fallback"
        gdf["fixed"] = True
        gdf["description"] = "Fallback fixed matched donor pixel centre used as monitoring support point."
        gdf = gdf.to_crs(CRS_GEO)
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        return gdf

    gdf = gpd.GeoDataFrame({
        "point_id": range(1, len(pts) + 1),
        "project": proj_name,
        "run_id": run_id,
        "method": f"grid_{MONITORING_SPACING_M:g}m",
        "spacing_m": MONITORING_SPACING_M,
        "fixed": True,
        "description": "Fixed systematic monitoring point. Coordinates locked at Year 0.",
    }, geometry=pts, crs=bounds_gdf.crs).to_crs(CRS_GEO)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf


# ================================================================
# SUMMARIES / DIAGNOSTICS
# ================================================================

def cov_summary(df, cont_covs):
    rows = []
    for col in cont_covs or []:
        if col not in df.columns:
            continue
        v = df[col].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        if len(v) == 0:
            continue
        rows.append({
            "covariate": col,
            "n": int(len(v)),
            "mean": round(float(v.mean()), 6),
            "median": round(float(v.median()), 6),
            "std": round(float(v.std()), 6),
            "q02": round(float(v.quantile(0.02)), 6),
            "q98": round(float(v.quantile(0.98)), 6),
        })
    return pd.DataFrame(rows)


def diagnostics_plot(points_gdf, support_gdf, bounds_gdf, mon_gdf, out_dir=None):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    ax = axes[0]
    points_gdf.plot(ax=ax, markersize=1.2, alpha=0.65)
    ax.set_title(f"1. Twin-tested matched donor pixels\n{len(points_gdf):,} unique centres")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)

    ax = axes[1]
    support_gdf.plot(ax=ax, alpha=0.25, edgecolor="black", linewidth=0.1)
    points_gdf.plot(ax=ax, markersize=0.5, alpha=0.5)
    ax.set_title(f"2. RA support cells\n{len(support_gdf):,} cells/blocks | {support_gdf.geometry.area.sum()/10000:,.2f} ha")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)

    ax = axes[2]
    bounds_gdf.plot(ax=ax, alpha=0.25, edgecolor="darkgreen", linewidth=1.0)
    if gdf_is_nonempty(mon_gdf):
        mon_gdf.to_crs(bounds_gdf.crs).plot(ax=ax, markersize=1.2, alpha=0.6)
    ax.set_title(f"3. Final locked RA\n{bounds_gdf.total_ha.iloc[0]:,.2f} ha | monitoring={gdf_nrows(mon_gdf):,}")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / "diagnostics.png", dpi=160, bbox_inches="tight")
    return fig




def build_native_matched_cells(points_gdf):
    """Return matched reference cells as native raster footprints when available."""
    if not _has_native_ref_bounds(points_gdf):
        return None
    geoms = []
    rows = []
    _bcols = ["ref_cell_xmin", "ref_cell_ymin", "ref_cell_xmax", "ref_cell_ymax"]
    for i, row in points_gdf.reset_index(drop=True).iterrows():
        # P2: salta righe con bounds nativi NaN (pixel non matchati)
        if not all(pd.notna(row.get(c, np.nan)) for c in _bcols):
            continue
        try:
            geoms.append(box(float(row["ref_cell_xmin"]), float(row["ref_cell_ymin"]),
                             float(row["ref_cell_xmax"]), float(row["ref_cell_ymax"])))
            rows.append({
                "cell_id": i + 1,
                "ref_pixel_id": str(row.get("ref_pixel_id", "")),
                "ref_row": int(row.get("ref_grid_row", -1)) if pd.notna(row.get("ref_grid_row", np.nan)) else -1,
                "ref_col": int(row.get("ref_grid_col", -1)) if pd.notna(row.get("ref_grid_col", np.nan)) else -1,
                "ref_lon": float(row.get("ref_lon", row.get("lon", np.nan))) if pd.notna(row.get("ref_lon", row.get("lon", np.nan))) else np.nan,
                "ref_lat": float(row.get("ref_lat", row.get("lat", np.nan))) if pd.notna(row.get("ref_lat", row.get("lat", np.nan))) else np.nan,
                "area_ha": float(box(float(row["ref_cell_xmin"]), float(row["ref_cell_ymin"]),
                                      float(row["ref_cell_xmax"]), float(row["ref_cell_ymax"])).area / 10000.0),
            })
        except Exception:
            continue
    if not rows:
        return None
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=points_gdf.crs)

# ================================================================
# EXPORT + LOCK
# ================================================================

def export_and_lock(bounds_gdf, support_gdf, mon_gdf, points_gdf, df_u, twin_report, out_dir, crs_m, bounds_method, area_ha, n_in, pct_in, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds_geo = bounds_gdf.to_crs(CRS_GEO)
    support_geo = support_gdf.to_crs(CRS_GEO)
    points_geo = points_gdf.to_crs(CRS_GEO)

    run_id = meta.get("run_id", "unknown")
    proj_name = meta.get("project_name", "unknown")
    cont_covs = detect_cont_covs(meta, df_u)
    cov_df = cov_summary(df_u, cont_covs)

    paths = {
        "gpkg": out_dir / "reference_area_FINAL.gpkg",
        "shp": out_dir / "reference_area_FINAL.shp",
        "geojson": out_dir / "reference_area_FINAL.geojson",
        "support": out_dir / "reference_area_support_cells.gpkg",
        "support_shp": out_dir / "reference_area_support_cells.shp",
        "native_cells": out_dir / "matched_reference_native_cells.gpkg",
        "native_cells_shp": out_dir / "matched_reference_native_cells.shp",
        "matched": out_dir / "matched_donor_pixels.gpkg",
        "matched_shp": out_dir / "matched_donor_pixels.shp",
        "mon_gpkg": out_dir / "monitoring_points_FIXED.gpkg",
        "mon_shp": out_dir / "monitoring_points_FIXED.shp",
        "mon_csv": out_dir / "monitoring_points_FIXED.csv",
        "cov": out_dir / "covariate_summary.csv",
        "diag": out_dir / "diagnostics.png",
        "manifest": out_dir / "reference_area_FINAL_manifest.json",
        "gee_js": out_dir / "GEE_load_reference_area.js",
    }
    for p in paths.values():
        if isinstance(p, Path):
            safe_remove(p)

    bounds_geo.to_file(paths["gpkg"], driver="GPKG")
    bounds_geo.to_file(paths["geojson"], driver="GeoJSON")
    shp_cols = ["project", "run_id", "bounds_method", "support_mode", "support_m", "total_ha", "n_matched_pts", "pct_pts_inside", "lock_status", "validity_years", "geometry"]
    bounds_geo[shp_cols].to_file(paths["shp"], driver="ESRI Shapefile")
    support_geo.to_file(paths["support"], driver="GPKG")
    if EXPORT_SUPPORT_CELLS_SHP:
        support_geo.to_file(paths["support_shp"], driver="ESRI Shapefile")

    native_cells = build_native_matched_cells(points_gdf)
    if native_cells is not None and EXPORT_NATIVE_MATCHED_CELLS_GPKG:
        native_cells_geo = native_cells.to_crs(CRS_GEO)
        native_cells_geo.to_file(paths["native_cells"], driver="GPKG")
        if EXPORT_NATIVE_MATCHED_CELLS_SHP:
            native_cells_geo.to_file(paths["native_cells_shp"], driver="ESRI Shapefile")

    points_geo.to_file(paths["matched"], driver="GPKG")
    if EXPORT_MATCHED_POINTS_SHP:
        points_geo.to_file(paths["matched_shp"], driver="ESRI Shapefile")
    if mon_gdf is not None:
        mon_gdf.to_file(paths["mon_gpkg"], driver="GPKG")
        if EXPORT_MONITORING_POINTS_SHP:
            mon_gdf.to_file(paths["mon_shp"], driver="ESRI Shapefile")
        mon_gdf.drop(columns=["geometry"], errors="ignore").to_csv(paths["mon_csv"], index=False)
    cov_df.to_csv(paths["cov"], index=False)

    fig_diag = diagnostics_plot(points_gdf, support_gdf, bounds_gdf, mon_gdf, out_dir)

    safe_proj_name = str(proj_name).replace(" ", "_").replace("-", "_")
    gee_js = (
        f"// GS STARR Locked Reference Area - {proj_name}\n"
        f"// RUN_ID: {run_id}\n"
        f"// Area: {area_ha:,.4f} ha\n"
        f"// Bounds method: {bounds_method}\n"
        f"// Upload reference_area_FINAL.gpkg/geojson as a GEE asset, then replace the asset path below.\n\n"
        f"var refArea = ee.FeatureCollection(\"users/YOUR_USER/{safe_proj_name}_reference_area_FINAL\");\n"
        f"print(\"Reference Area ha\", refArea.geometry().area().divide(10000));\n"
        f"Map.addLayer(refArea, {{color: \"orange\"}}, \"GS STARR Locked Reference Area\", true);\n"
        f"Map.centerObject(refArea, 10);\n"
    )
    with open(paths["gee_js"], "w", encoding="utf-8") as f:
        f.write(gee_js)

    sha = {}
    for p in paths.values():
        if isinstance(p, Path) and p.exists() and p.is_file():
            sha[p.name] = sha256_file(p)

    manifest = {
        "run_id": run_id,
        "project": proj_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": "GS STARR Track 1 SEMDB",
        "pipeline_version": "v07_native_pixel_support_with_vector_exports",
        "donor_pool_rule": "Non-Forest at Year 0 and >5 km from Activity Boundary",
        "reference_area_definition": {
            "description": "Locked RA built from matched donor native raster pixel/block support, not from a smoothed global hull.",
            "support_mode": RA_SUPPORT_MODE,
            "pixel_size_m": PIXEL_SIZE_M,
            "block_size_m": BLOCK_SIZE_M,
            "edge_buffer_m": EDGE_BUFFER_M,
            "close_gaps_m": CLOSE_GAPS_M,
            "simplify_tol_m": SIMPLIFY_TOL_M,
            "bounds_method": bounds_method,
            "total_ha": round(float(area_ha), 6),
            "n_matched_pts": int(n_in),
            "pct_pts_inside": round(float(pct_in), 2),
            "validity_years": 10,
            "lock_status": "LOCKED",
            "crs_metric": crs_m,
            "native_bounds_used": bool(_has_native_ref_bounds(points_gdf)),
            "vector_exports": {
                "support_cells_gpkg": str(paths["support"]),
                "support_cells_shp": str(paths["support_shp"]) if EXPORT_SUPPORT_CELLS_SHP else None,
                "native_matched_cells_gpkg": str(paths["native_cells"]) if paths["native_cells"].exists() else None,
                "native_matched_cells_shp": str(paths["native_cells_shp"]) if paths["native_cells_shp"].exists() else None,
                "matched_points_gpkg": str(paths["matched"]),
                "matched_points_shp": str(paths["matched_shp"]) if EXPORT_MATCHED_POINTS_SHP else None,
            },
        },
        "monitoring": {
            "mode": MONITORING_MODE,
            "spacing_m": MONITORING_SPACING_M,
            "n_points": gdf_nrows(mon_gdf),
            "description": "Fixed monitoring support points. Default is matched donor pixel centres.",
        },
        "twin_test_summary": twin_report or {},
        "twin_test_compliant": twin_report.get("twin_test_compliant", None) if twin_report else None,
        "twin_noncompliant_override_used": bool(allow_noncompliant_twin and
                                                twin_report.get("twin_test_compliant") is False),
        "covariate_summary": cov_df.set_index("covariate").to_dict() if not cov_df.empty else {},
        "files_sha256": sha,
        "lock_status": "LOCKED",
    }
    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return paths, manifest, fig_diag


# ================================================================
# MAIN STEP
# ================================================================

def run_reference_area_lock(base_dirs=None, output_dir=None, passed_df=None, meta=None,
                            twin_report=None, allow_noncompliant_twin=False, verbose=True):
    """Returns: bounds_gdf, mon_gdf, fig_diag, out_dir, manifest

    allow_noncompliant_twin : se False (default) e il twin test NON è conforme
        (twin_test_compliant=False nel report Step 03), Step 04 si ferma.
        Impostare True solo con decisione esplicita del PM documentata nel PDD.
    """
    if base_dirs is None:
        raise RuntimeError(
            "base_dirs non fornito. Passare base_dirs=[out03] dal runner "
            "oppure specificare la directory di output dello Step 03."
        )
    base_dirs = [Path(b) for b in base_dirs]
    out_dir = Path(output_dir) if output_dir else base_dirs[0].parent / "04_reference_area_lock"
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta is None:
        for candidate in [
            base_dirs[0].parent / "01_extract" / "extraction_report.json",
            base_dirs[0].parent / "02_matching" / "matching_summary.json",
        ]:
            if candidate.exists():
                meta = json.load(open(candidate, encoding="utf-8"))
                break
    meta = meta or {}

    if twin_report is None:
        rp = base_dirs[0] / "twin_test_report.json"
        twin_report = json.load(open(rp, encoding="utf-8")) if rp.exists() else {}

    # B4 fix: blocca se il twin test non è conforme, salvo override esplicito.
    twin_compliant = twin_report.get("twin_test_compliant", None)
    if twin_compliant is False and not allow_noncompliant_twin:
        raise RuntimeError(
            "STEP 04 bloccato: il twin/parallel-trend test NON è conforme "
            f"(selection_mode='{twin_report.get('selection_mode')}', "
            f"aggregate_twin_passed={twin_report.get('aggregate_twin_passed')}).\n"
            "I pixel selezionati sono 'best available' e non hanno superato il test "
            "aggregato; costruire la Reference Area su di essi non è difendibile in "
            "validazione.\nAzioni: rivedere Step 02/03 (rilassare PAIR_SLOPE_DIFF_MAX, "
            "ampliare la finestra pre-intervento, ricontrollare il donor pool), "
            "oppure forzare con allow_noncompliant_twin=True documentando nel PDD."
        )
    if twin_compliant is False and allow_noncompliant_twin:
        print("    WARNING: twin test NON conforme — proceduto con override esplicito "
              "(allow_noncompliant_twin=True).")

    if passed_df is None:
        passed_df = load_df(find_file(base_dirs, "twin_tested_pixels"))

    if verbose:
        print(f"\n{'=' * 60}")
        print("STEP 04 - Pixel/Block Support Reference Area Lock")
        print(f"Support mode : {RA_SUPPORT_MODE}")
        print(f"Monitoring   : {MONITORING_MODE}")
        print(f"Output       : {out_dir}")
        print(f"{'=' * 60}")

    if len(passed_df) == 0:
        raise RuntimeError("passed_df è vuoto. Step 04 non può costruire la RA.")

    print(f"    Pixel accettati dal twin test: {len(passed_df):,}")

    print("\n[1] Reference points")
    points_gdf, df_u, crs_m, lon_col, lat_col = make_points_gdf(passed_df)
    print(f"    Coordinate usate: {lon_col}, {lat_col}")
    print(f"    CRS metrico     : {crs_m}")
    print(f"    Pixel unici     : {len(points_gdf):,}")

    print("\n[2] Build support cells")
    support_gdf = build_support_cells(points_gdf, RA_SUPPORT_MODE)
    print(f"    Support features: {len(support_gdf):,}")
    print(f"    Support area    : {support_gdf.geometry.area.sum()/10000:,.4f} ha")

    print("\n[3] Build locked RA")
    bounds_gdf, bounds_method, area_ha, n_in, pct_in = build_reference_area(points_gdf, support_gdf, meta)
    print(f"    Reference Area: {area_ha:,.4f} ha | {bounds_method}")
    print(f"    Matched points inside: {n_in:,}/{len(points_gdf):,} ({pct_in:.1f}%)")

    print("\n[4] Monitoring points")
    mon_gdf = monitoring_points(bounds_gdf, points_gdf, meta)
    print(f"    Monitoring points: {gdf_nrows(mon_gdf):,}")

    print("\n[5] Export + Lock")
    paths, manifest, fig_diag = export_and_lock(
        bounds_gdf=bounds_gdf,
        support_gdf=support_gdf,
        mon_gdf=mon_gdf,
        points_gdf=points_gdf,
        df_u=df_u,
        twin_report=twin_report,
        out_dir=out_dir,
        crs_m=crs_m,
        bounds_method=bounds_method,
        area_ha=area_ha,
        n_in=n_in,
        pct_in=pct_in,
        meta=meta,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print("REFERENCE AREA LOCKED")
        print(f"Area       : {area_ha:,.4f} ha")
        print(f"Method     : {bounds_method}")
        print(f"Monitoring : {gdf_nrows(mon_gdf):,} points")
        print(f"Output     : {out_dir}")
        print(f"{'=' * 60}")

    return bounds_gdf, mon_gdf, fig_diag, out_dir, manifest


def main():
    run_reference_area_lock()


if __name__ == "__main__":
    main()
