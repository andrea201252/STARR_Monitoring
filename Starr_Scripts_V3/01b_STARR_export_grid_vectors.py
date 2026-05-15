# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB | 01b_STARR_export_grid_vectors.py
OPTIONAL — Sampled vector export of native raster pixel grids
============================================================

This script is intentionally separate from Step 01.
Reason: converting millions of 30 m raster cells to GPKG/SHP creates one
Shapely polygon per pixel and can saturate RAM. Step 01 must only extract and
save parquet tables. Use this script only for visual/audit samples.

Default outputs:
  01_extract/grid_vectors/project_grid_cells_sample.gpkg
  01_extract/grid_vectors/donor_grid_cells_sample.gpkg
  01_extract/grid_vectors/grid_vector_export_manifest.json

Full exports are disabled by default. Enable them only for very small grids.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


RUN_ID = "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v06_raster"
BASE_DIR_CANDIDATES = [
    "/content/drive/MyDrive/STARR_Idiofa_New_V2",
    "/content/content/MyDrive/STARR_Idiofa_New_V2",
]

OUTPUT_DIR = None
OUTPUT_FORMAT = "gpkg"  # gpkg only by default. Shapefile is intentionally disabled.

PROJECT_SAMPLE_N = 50_000
DONOR_SAMPLE_N = 50_000
EXPORT_PROJECT_FULL = False
EXPORT_DONOR_FULL = False
WRITE_SHP = False
SHAPEFILE_MAX_FEATURES = 50_000
RANDOM_STATE = 42

GEOM_COLS = [
    "pixel_id", "source_tile", "grid_row", "grid_col",
    "x_utm", "y_utm", "centroid_x_utm", "centroid_y_utm",
    "lon", "lat", "centroid_lon", "centroid_lat",
    "cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax",
    "cell_width_m", "cell_height_m", "pixel_area_ha",
    "WRB2_CODE", "ndvi_valid_years",
]


def detect_root():
    for b in BASE_DIR_CANDIDATES:
        base = Path(b)
        if base.exists():
            return base / "STARR_outputs" / RUN_ID
    raise FileNotFoundError(f"No base directory found: {BASE_DIR_CANDIDATES}")


def read_table(path, columns=None):
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(p).schema.names)
        cols = [c for c in (columns or available) if c in available]
        return pd.read_parquet(p, columns=cols)
    available = list(pd.read_csv(p, nrows=0).columns)
    cols = [c for c in (columns or available) if c in available]
    return pd.read_csv(p, usecols=cols)


def find_pixels_file(step01_dir, stem):
    for ext in ["parquet", "csv"]:
        p = step01_dir / f"{stem}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {stem}.parquet/csv in {step01_dir}")


def safe_remove(path):
    path = Path(path)
    if path.suffix.lower() == ".shp":
        stem = path.with_suffix("")
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
            q = stem.with_suffix(ext)
            if q.exists():
                q.unlink()
    elif path.exists():
        path.unlink()


def to_grid_gdf(df, crs, label, sample_n=None):
    required = {"cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label}: missing native cell bounds: {missing}")

    sampled = False
    if sample_n is not None and len(df) > int(sample_n):
        df = df.sample(int(sample_n), random_state=RANDOM_STATE).reset_index(drop=True)
        sampled = True
    else:
        df = df.reset_index(drop=True)

    import geopandas as gpd
    from shapely.geometry import box

    geoms = [box(float(x0), float(y0), float(x1), float(y1))
             for x0, y0, x1, y1 in zip(df["cell_xmin"], df["cell_ymin"], df["cell_xmax"], df["cell_ymax"])]

    keep = [c for c in GEOM_COLS if c in df.columns and c not in {"cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax"}]
    attrs = df[keep].copy()
    attrs["grid_label"] = label[:20] if WRITE_SHP else label
    attrs["is_sample"] = bool(sampled)
    attrs["n_source"] = int(len(df))
    return gpd.GeoDataFrame(attrs, geometry=geoms, crs=crs), sampled


def write_vector(gdf, out_dir, stem):
    out = {}
    gpkg = out_dir / f"{stem}.gpkg"
    safe_remove(gpkg)
    gdf.to_file(gpkg, driver="GPKG", layer=stem)
    out["gpkg"] = str(gpkg)

    if WRITE_SHP:
        if len(gdf) > SHAPEFILE_MAX_FEATURES:
            out["shp_skipped"] = f"feature_count_{len(gdf)}_above_limit_{SHAPEFILE_MAX_FEATURES}"
        else:
            shp = out_dir / f"{stem}.shp"
            safe_remove(shp)
            gdf.to_file(shp, driver="ESRI Shapefile")
            out["shp"] = str(shp)
    return out


def main(step01_dir=None, output_dir=None):
    root = detect_root()
    step01_dir = Path(step01_dir) if step01_dir else root / "01_extract"
    output_dir = Path(output_dir) if output_dir else step01_dir / "grid_vectors"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = step01_dir / "extraction_report.json"
    report = json.load(open(report_path, encoding="utf-8")) if report_path.exists() else {}
    crs = report.get("crs_src", "EPSG:32734")

    project_path = find_pixels_file(step01_dir, "project_pixels_raw")
    donor_path = find_pixels_file(step01_dir, "donor_pixels_raw")

    outputs = {}
    project_df = read_table(project_path, GEOM_COLS)
    donor_df = read_table(donor_path, GEOM_COLS)

    if EXPORT_PROJECT_FULL:
        n = None
        stem = "project_grid_cells_FULL"
    else:
        n = PROJECT_SAMPLE_N
        stem = "project_grid_cells_sample"
    gdf, sampled = to_grid_gdf(project_df, crs, "project", n)
    outputs[stem] = write_vector(gdf, output_dir, stem)
    del gdf, project_df

    if EXPORT_DONOR_FULL:
        n = None
        stem = "donor_grid_cells_FULL"
    else:
        n = DONOR_SAMPLE_N
        stem = "donor_grid_cells_sample"
    gdf, sampled = to_grid_gdf(donor_df, crs, "donor", n)
    outputs[stem] = write_vector(gdf, output_dir, stem)
    del gdf, donor_df

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "crs": crs,
        "source_step01_dir": str(step01_dir),
        "note": "Optional sampled vector export. Step 01 itself does not generate GPKG/SHP by default to avoid RAM saturation.",
        "project_sample_n": PROJECT_SAMPLE_N,
        "donor_sample_n": DONOR_SAMPLE_N,
        "export_project_full": EXPORT_PROJECT_FULL,
        "export_donor_full": EXPORT_DONOR_FULL,
        "write_shp": WRITE_SHP,
        "outputs": outputs,
    }
    with open(output_dir / "grid_vector_export_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    return outputs, manifest


if __name__ == "__main__":
    main()
