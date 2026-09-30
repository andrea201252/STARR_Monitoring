"""
starr_geo_io.py — GDAL/pyproj raster & vector I/O for the ArcGIS Pro port of the
GS STARR Track 1 SEMDB pipeline.

Design
------
The heavy statistical core (Step 02 matching, Step 03 twin test, Step 05 CI /
UNCBSL / baseline math) is REUSED UNCHANGED from the Colab modules — same
numpy / pandas / scikit-learn / scipy code, therefore numerically identical
results. Only the *I/O layer* (raster reading + point sampling + vector output),
which in Colab used `rasterio` / `geopandas`, is re-implemented here on top of
`osgeo.gdal` / `osgeo.ogr` / `pyproj`, ALL of which ship inside the ArcGIS Pro
Python environment (arcgispro-py3) — no extra installation required.

`rasterio` is a thin wrapper over GDAL, so this port is close to mechanical and
the pixel reads are the same values.

Everything that does NOT need GDAL lives in the PURE functions below
(`build_pixels_dataframe`, `world_to_pixel`, …) so it can be unit-tested without
ArcGIS. The GDAL-touching wrappers are kept intentionally thin.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# GDAL / OGR / pyproj are resolved lazily so this file can be imported (and its
# pure functions unit-tested) on a machine without GDAL.
try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    _HAS_GDAL = True
except Exception:
    gdal = ogr = osr = None
    _HAS_GDAL = False

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except Exception:
    Transformer = None
    _HAS_PYPROJ = False


# ── Reprojection via GDAL/osr (preferred — always present with GDAL; some
#    ArcGIS installs do NOT expose an importable pyproj, so osr is the robust
#    path and pyproj is only a fallback). ──────────────────────────────────────
def _osr_srs(spec):
    """osr.SpatialReference from an EPSG int or a WKT string, forced to
    traditional GIS axis order (x=lon, y=lat) to match pyproj always_xy=True."""
    s = osr.SpatialReference()
    if isinstance(spec, int):
        s.ImportFromEPSG(int(spec))
    else:
        s.ImportFromWkt(str(spec))
    try:
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    return s


def _osr_ct(src_spec, dst_spec):
    """Build a GDAL CoordinateTransformation (src/dst = EPSG int or WKT str)."""
    if not _HAS_GDAL:
        raise RuntimeError("GDAL/osr not available for reprojection.")
    return osr.CoordinateTransformation(_osr_srs(src_spec), _osr_srs(dst_spec))


def _ct_apply(ct, x, y, chunk=500_000):
    """Apply a prebuilt osr transform to x/y arrays → (x2, y2) float arrays.
    Chunked so a multi-million-pixel donor raster does not spike memory."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = x.size
    if n == 0:
        return x, y
    ox = np.empty(n, dtype=float)
    oy = np.empty(n, dtype=float)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        res = np.asarray(ct.TransformPoints(list(zip(x[i:j].tolist(), y[i:j].tolist()))),
                         dtype=float)
        ox[i:j] = res[:, 0]
        oy[i:j] = res[:, 1]
    return ox, oy


def _reproject_xy(src_spec, dst_spec, x, y):
    """One-shot reprojection with osr, pyproj fallback."""
    if _HAS_GDAL:
        try:
            return _ct_apply(_osr_ct(src_spec, dst_spec), x, y)
        except Exception:
            pass
    if _HAS_PYPROJ:
        def _s(v):
            return f"EPSG:{v}" if isinstance(v, int) else v
        tr = Transformer.from_crs(_s(src_spec), _s(dst_spec), always_xy=True)
        xx, yy = tr.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(xx, float), np.asarray(yy, float)
    raise RuntimeError("Neither GDAL/osr nor pyproj can reproject coordinates. "
                       "The ArcGIS Pro Python normally provides both.")


# ─────────────────────────────────────────────────────────────────────────────
# PURE CORE  (no GDAL — unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def cell_center_coords(geotransform, rows, cols):
    """
    Cell-CENTER coordinates in the raster CRS for the given (row, col) arrays.

    geotransform: GDAL 6-tuple (ox, pw, rx, oy, ry, ph). North-up rasters have
    rx = ry = 0 and ph < 0 — identical convention to the rasterio Affine used by
    the Colab Step 01 (`x = c + (col+0.5)*a`, `y = f + (row+0.5)*e`).
    """
    ox, pw, rx, oy, ry, ph = [float(v) for v in geotransform]
    x = ox + (cols + 0.5) * pw + (rows + 0.5) * rx
    y = oy + (cols + 0.5) * ry + (rows + 0.5) * ph
    return x.astype(np.float64), y.astype(np.float64)


def build_pixels_dataframe(band_stack, band_names, geotransform,
                           project_xy_to_lonlat, row_offset=0, tile_name=""):
    """
    Flatten a (n_bands, H, W) array into the one-row-per-valid-pixel DataFrame
    that Steps 02/03/04/05 expect. Mirrors Step 01 `_process_block` exactly.

    A pixel is VALID when every band is finite (nodata must already be np.nan).

    project_xy_to_lonlat : callable(x_arr, y_arr) -> (lon_arr, lat_arr)
        Projects raster-CRS coordinates to WGS84 lon/lat (identity if the raster
        is already geographic). See `make_lonlat_projector`.

    Returns an empty DataFrame if no valid pixel.
    """
    band_stack = np.asarray(band_stack, dtype=np.float64)
    if band_stack.ndim != 3:
        raise ValueError("band_stack must be (n_bands, H, W)")
    n_b, H, W = band_stack.shape
    if len(band_names) != n_b:
        raise ValueError(f"band_names ({len(band_names)}) != n_bands ({n_b})")

    valid_2d = np.isfinite(band_stack).all(axis=0)
    if not valid_2d.any():
        return pd.DataFrame()

    rows, cols = np.where(valid_2d)
    global_rows = rows + int(row_offset)

    x, y = cell_center_coords(geotransform, global_rows, cols)
    lon, lat = project_xy_to_lonlat(x, y)

    _, pw, _, _, _, ph = [float(v) for v in geotransform]
    df = pd.DataFrame(band_stack[:, rows, cols].T, columns=list(band_names))
    df["x_utm"]     = x
    df["y_utm"]     = y
    df["lon"]       = np.asarray(lon, dtype=np.float64)
    df["lat"]       = np.asarray(lat, dtype=np.float64)
    df["grid_row"]  = global_rows.astype(np.int32)
    df["grid_col"]  = cols.astype(np.int32)
    df["cell_xmin"] = x - abs(pw) / 2.0
    df["cell_ymin"] = y - abs(ph) / 2.0
    df["cell_xmax"] = x + abs(pw) / 2.0
    df["cell_ymax"] = y + abs(ph) / 2.0
    df["source_tile"] = tile_name
    df["pixel_id"]  = [f"{tile_name}::r{int(r)}::c{int(c)}"
                       for r, c in zip(global_rows, cols)]
    return df


def world_to_pixel(geotransform, x, y):
    """Inverse geotransform: world (raster CRS) → fractional (col, row).
    For north-up rasters (rx=ry=0). Returns integer (col, row) of the containing
    pixel via floor — same as gdal/rasterio index()."""
    ox, pw, rx, oy, ry, ph = [float(v) for v in geotransform]
    if rx == 0.0 and ry == 0.0:
        col = np.floor((np.asarray(x, float) - ox) / pw).astype(np.int64)
        row = np.floor((np.asarray(y, float) - oy) / ph).astype(np.int64)
        return col, row
    # general affine inverse
    det = pw * ph - rx * ry
    xx = np.asarray(x, float) - ox
    yy = np.asarray(y, float) - oy
    col = np.floor((ph * xx - rx * yy) / det).astype(np.int64)
    row = np.floor((-ry * xx + pw * yy) / det).astype(np.int64)
    return col, row


def detect_continuous_covariates(band_names):
    """EXACT copy of Colab Step 01 detect_continuous_covariates: continuous
    covariates = all bands except the flag/categorical/bookkeeping ones
    (is_project, is_donor, lon, lat, precip bins, …), the NDVI_<year> / band_<n>
    series, and WRB2_CODE (which becomes the texture class)."""
    import re
    skip = {"is_project", "is_donor", "ndvi_valid_years", "pixel_area_ha",
            "lon", "lat", "precip_bin", "precip_bin_100mm", "precip_bin_250mm"}
    skip_pat = re.compile(r"^(NDVI_\d{4}|band_\d+)$")
    return [b for b in band_names
            if b not in skip and not skip_pat.match(b) and b != "WRB2_CODE"]


def detect_ndvi_year_cols(band_names):
    import re
    pat = re.compile(r"^NDVI_(\d{4})$")
    hits = [(int(pat.match(b).group(1)), b) for b in band_names if pat.match(b)]
    hits.sort()
    return [b for _, b in hits]


# ─────────────────────────────────────────────────────────────────────────────
# GDAL WRAPPERS  (thin — validated inside ArcGIS Pro)
# ─────────────────────────────────────────────────────────────────────────────

def _require_gdal():
    if not _HAS_GDAL:
        raise RuntimeError(
            "GDAL (osgeo) not importable. Run this tool with the ArcGIS Pro "
            "Python (arcgispro-py3), which ships GDAL.")


def read_raster_stack(path):
    """
    Read a multi-band GeoTIFF into (band_stack, band_names, geotransform,
    srs_wkt, is_geographic, pixel_size). Nodata is converted to np.nan.

    Band names come from each band's GDAL description (what GEE writes, e.g.
    'SOC_g_kg'); falls back to 'band{i}' when a description is empty.
    """
    _require_gdal()
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    gt = ds.GetGeoTransform()
    srs_wkt = ds.GetProjection()
    n = ds.RasterCount
    stack, names = [], []
    for i in range(1, n + 1):
        b = ds.GetRasterBand(i)
        arr = b.ReadAsArray().astype(np.float64)
        nod = b.GetNoDataValue()
        if nod is not None:
            arr = np.where(arr == nod, np.nan, arr)
        stack.append(arr)
        desc = (b.GetDescription() or "").strip()
        names.append(desc if desc else f"band{i}")
    ds = None
    is_geo = False
    if srs_wkt:
        sr = osr.SpatialReference(); sr.ImportFromWkt(srs_wkt)
        is_geo = bool(sr.IsGeographic())
    pixel_size = abs(float(gt[1]))
    return np.stack(stack, axis=0), names, gt, srs_wkt, is_geo, pixel_size


def make_lonlat_projector(srs_wkt, is_geographic):
    """Return callable(x, y) -> (lon, lat). Identity when the raster is already
    geographic; otherwise reproject the raster CRS to EPSG:4326 with GDAL/osr
    (preferred, built once) and pyproj as a fallback."""
    if is_geographic or not srs_wkt:
        return lambda x, y: (np.asarray(x, float), np.asarray(y, float))
    if _HAS_GDAL:
        try:
            ct = _osr_ct(srs_wkt, 4326)               # raster CRS -> lon/lat
            return lambda x, y: _ct_apply(ct, x, y)
        except Exception:
            pass
    if _HAS_PYPROJ:
        tr = Transformer.from_crs(srs_wkt, "EPSG:4326", always_xy=True)
        def _proj(x, y):
            lon, lat = tr.transform(np.asarray(x, float), np.asarray(y, float))
            return np.asarray(lon, float), np.asarray(lat, float)
        return _proj
    raise RuntimeError("Neither GDAL/osr nor pyproj can reproject the raster to "
                       "lon/lat. The ArcGIS Pro Python normally provides both; "
                       "check the raster CRS.")


def vector_srs_wkt(path):
    """CRS (WKT) of a vector dataset (e.g. the FNF shapefile), via OGR. None if
    unavailable. Used as a fallback CRS when a covariate raster has no embedded
    CRS."""
    if not _HAS_GDAL or not path:
        return None
    try:
        ds = ogr.Open(str(path))
        lyr = ds.GetLayer(0)
        sr = lyr.GetSpatialRef()
        wkt = sr.ExportToWkt() if sr is not None else None
        ds = None
        return wkt or None
    except Exception:
        return None


def read_covariate_raster_to_df(path, tile_name=None, fallback_srs_wkt=None):
    """
    High-level: GeoTIFF covariate stack → (df, band_names, meta) with the exact
    schema Step 02 expects. `meta` carries band_names, continuous_covariates,
    ndvi_year_cols, crs, pixel_size.

    fallback_srs_wkt : if the raster has NO embedded CRS, assume this one
    (e.g. the FNF shapefile's CRS) so the pixels can still be georeferenced.
    """
    stack, names, gt, srs_wkt, is_geo, px = read_raster_stack(path)
    if (not srs_wkt) and fallback_srs_wkt:
        srs_wkt = fallback_srs_wkt
        try:
            _sr = osr.SpatialReference(); _sr.ImportFromWkt(srs_wkt)
            is_geo = bool(_sr.IsGeographic())
        except Exception:
            is_geo = False
    projector = make_lonlat_projector(srs_wkt, is_geo)
    import os
    tile = tile_name if tile_name is not None else os.path.basename(path)
    df = build_pixels_dataframe(stack, names, gt, projector, tile_name=tile)
    meta = {
        "band_names": names,
        "continuous_covariates": detect_continuous_covariates(names),
        "ndvi_year_cols": detect_ndvi_year_cols(names),
        "crs": srs_wkt,
        "pixel_size": px,
        "raster_is_geographic": is_geo,
    }
    return df, names, meta


def sample_raster_at_lonlat(path, lon, lat, band=1, project_from_lonlat=True):
    """
    Sample one band of a raster at lon/lat points (returns np.nan off-raster or
    on nodata). Mirrors rasterio's `src.sample`. If the raster is projected and
    `project_from_lonlat`, the lon/lat are reprojected into the raster CRS first.
    """
    _require_gdal()
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    gt = ds.GetGeoTransform()
    srs_wkt = ds.GetProjection()
    b = ds.GetRasterBand(int(band))
    nod = b.GetNoDataValue()
    W, H = ds.RasterXSize, ds.RasterYSize

    x = np.asarray(lon, float); y = np.asarray(lat, float)
    if project_from_lonlat and srs_wkt:
        sr = osr.SpatialReference(); sr.ImportFromWkt(srs_wkt)
        if not sr.IsGeographic():
            x, y = _reproject_xy(4326, srs_wkt, x, y)   # lon/lat -> raster CRS (osr, pyproj fallback)

    col, row = world_to_pixel(gt, x, y)
    out = np.full(len(np.atleast_1d(col)), np.nan, dtype=np.float64)
    inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    arr = b.ReadAsArray().astype(np.float64)
    ci = col[inb]; ri = row[inb]
    vals = arr[ri, ci]
    if nod is not None:
        vals = np.where(vals == nod, np.nan, vals)
    out[inb] = vals
    ds = None
    return out


def raster_pixel_area_ha_gdal(path):
    """Pixel area in hectares. If projected (metres) → |pw*ph|/1e4. If geographic
    → approximate at the raster-centre latitude (deg → m via 111320)."""
    _require_gdal()
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform(); srs_wkt = ds.GetProjection()
    H = ds.RasterYSize
    pw, ph = abs(gt[1]), abs(gt[5])
    is_geo = False
    if srs_wkt:
        sr = osr.SpatialReference(); sr.ImportFromWkt(srs_wkt); is_geo = bool(sr.IsGeographic())
    if not is_geo:
        area_m2 = pw * ph
    else:
        lat_c = gt[3] + (H / 2.0) * gt[5]
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat_c))
        area_m2 = (pw * m_per_deg_lon) * (ph * m_per_deg_lat)
    ds = None
    return float(area_m2 / 1e4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP-05 COMPATIBLE SAMPLER  (drop-in replacement for s05.sample_raster_values)
# ─────────────────────────────────────────────────────────────────────────────

def sample_raster_values(points_df, lon_col, lat_col, raster_path,
                         band=None, band_name=None, out_col="value",
                         chunk_size=None):
    """
    GDAL drop-in for s05.sample_raster_values — SAME signature and SAME return
    contract ``(values, meta)`` — so it can be monkeypatched onto the Step 05
    module (``s05.sample_raster_values = starr_geo_io.sample_raster_values``) to
    make the raster-based baseline run under GDAL while every downstream number
    (AGB→C conversion, ΔC, CI, UNCBSL) is computed by the ORIGINAL, unchanged
    Step 05 code.

    Points are lon/lat (WGS84); they are reprojected into the raster CRS when the
    raster is projected. Off-raster / nodata / non-finite → np.nan.
    """
    _require_gdal()
    lons = pd.to_numeric(points_df[lon_col], errors="coerce").to_numpy(np.float64)
    lats = pd.to_numeric(points_df[lat_col], errors="coerce").to_numpy(np.float64)
    if np.isnan(lons).any() or np.isnan(lats).any():
        raise ValueError(f"Found NaN coordinates while sampling {out_col}.")

    ds = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {raster_path}")
    gt = ds.GetGeoTransform()
    srs_wkt = ds.GetProjection()
    W, H, count = ds.RasterXSize, ds.RasterYSize, ds.RasterCount

    # resolve band index (explicit index wins; else by GDAL band description)
    if band is not None:
        bidx = int(band)
        if bidx < 1 or bidx > count:
            raise ValueError(f"Invalid band index {bidx} for {raster_path} ({count} bands).")
    elif band_name:
        bidx = None
        for i in range(1, count + 1):
            if (ds.GetRasterBand(i).GetDescription() or "").strip() == str(band_name):
                bidx = i
                break
        if bidx is None:
            raise ValueError(f"band_name '{band_name}' not found in {raster_path}.")
    else:
        bidx = 1

    b = ds.GetRasterBand(bidx)
    nod = b.GetNoDataValue()

    is_geo = False
    xs, ys = lons, lats
    if srs_wkt:
        sr = osr.SpatialReference(); sr.ImportFromWkt(srs_wkt)
        is_geo = bool(sr.IsGeographic())
        if not is_geo:
            xs, ys = _reproject_xy(4326, srs_wkt, lons, lats)   # lon/lat -> raster CRS (osr, pyproj fallback)
            xs = np.asarray(xs, np.float64); ys = np.asarray(ys, np.float64)
    else:
        raise ValueError(f"The raster has no CRS: {raster_path}")

    col, row = world_to_pixel(gt, xs, ys)
    values = np.full(len(points_df), np.nan, dtype=np.float64)
    inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    arr = b.ReadAsArray().astype(np.float64)
    v = arr[row[inb], col[inb]]
    if nod is not None:
        v = np.where(np.isclose(v, float(nod)), np.nan, v)
    v = np.where(np.isfinite(v), v, np.nan)
    values[inb] = v

    pixel_area_ha = (float(abs(gt[1] * gt[5]) / 10_000.0) if not is_geo else None)
    meta = {
        "path": str(raster_path),
        "band_index": int(bidx),
        "band_name": band_name,
        "raster_crs": srs_wkt,
        "raster_width": int(W),
        "raster_height": int(H),
        "raster_count": int(count),
        "raster_nodata": None if nod is None else float(nod),
        "pixel_area_ha": pixel_area_ha,
    }
    ds = None
    return values, meta


def raster_pixel_area_ha(path):
    """GDAL drop-in for s05.raster_pixel_area_ha (projected → ha; geographic → None)."""
    _require_gdal()
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        return None
    gt = ds.GetGeoTransform(); srs_wkt = ds.GetProjection()
    is_geo = True
    if srs_wkt:
        sr = osr.SpatialReference(); sr.ImportFromWkt(srs_wkt); is_geo = bool(sr.IsGeographic())
    ds = None
    if srs_wkt and not is_geo:
        return float(abs(gt[1] * gt[5]) / 10_000.0)
    return None
