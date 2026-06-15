# -*- coding: utf-8 -*-
"""
============================================================
GS STARR – Track 1 SEMDB  |  01_STARR_raster_extract.py
STEP 01 — Raster Extraction + Shapefile Spatial Masking
============================================================

Strategia di filtraggio:
  GEE esporta TUTTI i pixel con covariate valide (senza filtro forest).
  Questo script applica il filtro spaziale preciso in Python usando
  due shapefile locali su Drive, via rasterizzazione della geometria
  vettoriale sulla stessa griglia del TIF (O(n), nessun point-in-polygon).

SHAPEFILE usati:
  FNF18_fullBuffer.shp
    Poligoni delle aree FORESTALI a T0 (2018).
    Pixel donor DENTRO questi poligoni vengono rimossi (forest ≠ eleggibile).
    Pixel PA DENTRO questi poligoni vengono rimossi (PA deve essere non-forest a T0).

  Eligible_FNF_fullBuffer.shp
    Poligoni delle aree ELEGGIBILI come donor (non-forest per 10 anni ante T0).
    Pixel donor FUORI da questi poligoni vengono rimossi.
    Non applicato alla PA (la PA ha il suo shapefile di riferimento in GEE).

Fix rispetto alla versione precedente:
  BUG-01  _snap: fix offset ½ pixel (formula rasterio corretta)
  BUG-02  add_patch_cells: autoscale_view() esplicita (Matplotlib >= 3.6)
  BUG-03  set_limits: quantili robusti anti-outlier
  BUG-04  BLOCK_HEIGHT sempre attivo (evita OOM donor)
  BUG-05  Donor grid: panel C mostra donor a extent completo
  BUG-06  x_utm/y_utm calcolati in _process_block (no re-proiezione nel plot)
"""

import warnings
warnings.filterwarnings("ignore")

import gc
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


# ── RAM MONITOR + PURGE ──────────────────────────────────────────────

try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def _ram_mb() -> float:
    if not _PSUTIL:
        return 0.0
    return _psutil.Process().memory_info().rss / 1_048_576


def _purge(*objs, label: str = "", close_figs: bool = False) -> None:
    import matplotlib.pyplot as _plt
    for obj in objs:
        if close_figs:
            try: _plt.close(obj)
            except Exception: pass
        try: del obj
        except Exception: pass
    gc.collect()
    if label:
        ram = _ram_mb()
        ram_str = f" | RAM: {ram:.0f} MB" if ram > 0 else ""
        print(f"    [GC] {label}{ram_str}")


# ── CONFIG ────────────────────────────────────────────────────────────
# Impostare RUN_ID_BASE con il nome base del run (senza suffisso _extNkm).
# Esempio: "Idiofa_Lobi_2018_buf50km_excl5km_WRB2_v07_raster"
# Lasciare "" per usare wildcard pura (trova tutti i TIF nella directory).
RUN_ID_BASE  = ""
PROJECT_NAME = "Idiofa_Lobi"

# Lista di directory dove cercare i TIF di input.
# Può essere sovrascritta passando base_dirs a run_extraction().
BASE_DIR_CANDIDATES: list = []

OUTPUT_DIR = None
# I pattern vengono costruiti a runtime in run_extraction() per evitare
# che RUN_ID_BASE vuoto produca glob non validi come "covariates_project_[]*.tif".
PROJECT_TIF_PATTERN = None  # calcolato in run_extraction()
DONOR_TIF_PATTERN   = None  # calcolato in run_extraction()
OUTPUT_FORMAT       = "parquet"
BLOCK_HEIGHT        = 2048

PLOT_MAX_PA_BOXES    = 80_000
PLOT_MAX_DONOR_BOXES = 50_000

# ── SHAPEFILE FILTER ─────────────────────────────────────────────────
# I due shapefile devono trovarsi su Drive nel percorso indicato.
# Impostare a None per saltare il filtro corrispondente.

# Aree FORESTALI a T0 2018 (da escludere da donor E da PA).
FNF_SHAPEFILE = (

)

# Aree ELEGGIBILI per il donor pool (non-forest per 10 anni ante T0).
# Applicato solo al donor, non alla PA.
ELIGIBLE_SHAPEFILE = (
)

# Se True, usa all_touched=False (solo pixel con centroide dentro il poligono).
# Raccomandato per poligoni precisi; mettere True per poligoni grossolani.
SHP_ALL_TOUCHED = False

# ── PA SELECTIVE EXTRACTION ───────────────────────────────────────────
PA_EDGE_EXCLUSION_ENABLED  = True
PA_EDGE_EXCLUSION_N_PIXELS = 1
# P1 FIX: di default NON sottocampionare la PA — prendi TUTTI i pixel.
# Impostare PA_SPATIAL_SAMPLE_ENABLED=True solo se la RAM è insufficiente;
# in quel caso la media ΔC è scalata all'area PA piena (vedi nota A3).
PA_SPATIAL_SAMPLE_ENABLED  = False
PA_MAX_PIXELS              = 150_000
PA_SPATIAL_GRID_STEP_M     = 150.0


# ── PA SELECTIVE FILTERS ─────────────────────────────────────────────

def exclude_edge_pixels(df, pixel_size_m=30.0):
    if not PA_EDGE_EXCLUSION_ENABLED:
        return df, 0
    buf  = PA_EDGE_EXCLUSION_N_PIXELS * pixel_size_m
    mask = ((df["x_utm"] >= df["x_utm"].min() + buf) &
            (df["x_utm"] <= df["x_utm"].max() - buf) &
            (df["y_utm"] >= df["y_utm"].min() + buf) &
            (df["y_utm"] <= df["y_utm"].max() - buf))
    n_excl = int((~mask).sum())
    print(f"    Edge exclusion ({buf:.0f}m): {n_excl:,} px rimossi")
    return df.loc[mask].reset_index(drop=True), n_excl


def spatially_stratified_sample(df):
    if not PA_SPATIAL_SAMPLE_ENABLED or len(df) <= PA_MAX_PIXELS:
        return df, False
    work = df.copy()
    work["_gx"] = (work["x_utm"] // PA_SPATIAL_GRID_STEP_M).astype(np.int64)
    work["_gy"] = (work["y_utm"] // PA_SPATIAL_GRID_STEP_M).astype(np.int64)
    # PERF: 1 pixel per cella di griglia via shuffle + drop_duplicates,
    # invece di groupby.apply(sample(1)) che è O(n_celle) con overhead pandas
    # enorme su milioni di gruppi.
    work = work.sample(frac=1.0, random_state=42)
    idx = work.drop_duplicates(subset=["_gx", "_gy"], keep="first").index
    sampled = df.loc[idx].reset_index(drop=True)
    _purge(work, label=f"spatial sample {len(df):,}->{len(sampled):,} px")
    return sampled, True


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
        return pd.DataFrame()

    rows, cols  = np.where(valid_2d)
    global_rows = rows + row_offset
    pix_x = float(transform.a)
    pix_y = float(transform.e)

    x_utm = (transform.c + (cols       + 0.5) * pix_x).astype(np.float64)
    y_utm = (transform.f + (global_rows + 0.5) * pix_y).astype(np.float64)

    lons, lats = transformer.transform(x_utm, y_utm)

    df = pd.DataFrame(data.data[:, rows, cols].T, columns=band_names)
    df["x_utm"]     = x_utm
    df["y_utm"]     = y_utm
    df["lon"]       = np.asarray(lons, dtype=np.float64)
    df["lat"]       = np.asarray(lats, dtype=np.float64)
    df["grid_row"]  = global_rows.astype(np.int32)
    df["grid_col"]  = cols.astype(np.int32)
    df["cell_xmin"] = x_utm - abs(pix_x) / 2.0
    df["cell_ymin"] = y_utm - abs(pix_y) / 2.0
    df["cell_xmax"] = x_utm + abs(pix_x) / 2.0
    df["cell_ymax"] = y_utm + abs(pix_y) / 2.0
    df["source_tile"] = tile_name
    df["pixel_id"]    = [f"{tile_name}::r{int(r)}::c{int(c)}"
                         for r, c in zip(global_rows, cols)]

    del rows, cols, global_rows, x_utm, y_utm, lons, lats, invalid_2d, valid_2d
    return df


def compute_donor_clip_window(proj_df, donor_tif_path, extent_km):
    """
    Calcola la rasterio Window sul TIF donor = bbox(proj_df) + buffer(extent_km).

    Returns (window, clip_transform) oppure (None, None) se extent_km='full'.

    Il clip è a livello I/O: nessun pixel fuori dal buffer viene mai caricato
    in RAM. Il risparmio è proporzionale a (clip_area / full_donor_area).
    """
    if extent_km == "full":
        return None, None

    import rasterio.windows as _rw
    from rasterio.windows import from_bounds as _from_bounds

    buf_m = float(extent_km) * 1000.0
    minx  = float(proj_df["x_utm"].min()) - buf_m
    maxx  = float(proj_df["x_utm"].max()) + buf_m
    miny  = float(proj_df["y_utm"].min()) - buf_m
    maxy  = float(proj_df["y_utm"].max()) + buf_m

    with rasterio.open(donor_tif_path) as src:
        full_win = _rw.Window(0, 0, src.width, src.height)
        try:
            raw_win  = _from_bounds(minx, miny, maxx, maxy, src.transform)
            clip_win = raw_win.intersection(full_win)
        except Exception as exc:
            print(f"    WARN compute_donor_clip_window: {exc} — uso extent completo")
            return None, None

        clip_w = max(1, int(np.round(clip_win.width)))
        clip_h = max(1, int(np.round(clip_win.height)))
        if clip_w <= 0 or clip_h <= 0:
            raise ValueError(
                f"Clip window vuota per extent_km={extent_km}. "
                "Verificare che CRS di proj_df e TIF donor coincidano."
            )
        clip_transform = _rw.transform(clip_win, src.transform)
        saved = (1.0 - clip_w * clip_h / (src.width * src.height)) * 100.0
        print(f"    Donor clip [{extent_km}km]: {clip_w}×{clip_h} px "
              f"(full: {src.width}×{src.height}) | I/O risparmio: {saved:.0f}%")
        print(f"    BBox donor UTM: ({minx:.0f},{miny:.0f}) → ({maxx:.0f},{maxy:.0f})")

    return clip_win, clip_transform


def extract_tile(tif_path, transformer, clip_window=None, clip_transform=None):
    import rasterio.windows as _rw

    with rasterio.open(tif_path) as src:
        band_names = detect_band_names(src)
        base_transform = src.transform
        pixel_size = abs(base_transform.a)

        if clip_window is not None:
            eff_transform = clip_transform or _rw.transform(clip_window, base_transform)
            read_h  = max(1, int(np.round(clip_window.height)))
            read_w  = max(1, int(np.round(clip_window.width)))
            col_off = int(np.round(clip_window.col_off))
            row_off = int(np.round(clip_window.row_off))
        else:
            eff_transform = base_transform
            read_h  = src.height
            read_w  = src.width
            col_off = 0
            row_off = 0

        print(f"    {tif_path.name}")
        print(f"      {src.width}x{src.height}px | "
              f"EPSG:{src.crs.to_epsg() if src.crs else '?'} | {src.count} bande")
        if clip_window is not None:
            print(f"      → lettura clippata: {read_w}×{read_h} px "
                  f"(col_off={col_off}, row_off={row_off})")
        print(f"      Origin: ({eff_transform.c:.2f}, {eff_transform.f:.2f}) "
              f"| Pixel: {pixel_size:.1f}m")

        dfs = []
        for r0 in range(0, read_h, BLOCK_HEIGHT):
            h   = min(BLOCK_HEIGHT, read_h - r0)
            win = Window(col_off, row_off + r0, read_w, h)
            data = src.read(window=win, masked=True)
            blk  = _process_block(data, eff_transform, transformer,
                                  band_names, r0, tif_path.name)
            del data
            if len(blk) > 0:
                dfs.append(blk)
            del blk

    if not dfs:
        return pd.DataFrame(), band_names, eff_transform
    result = pd.concat(dfs, ignore_index=True)
    _purge(*dfs, label=f"blocchi {tif_path.name} ({len(result):,} px)")
    return result, band_names, eff_transform


def load_raster_to_dataframe(tif_files, label, clip_window=None, clip_transform=None):
    if not tif_files:
        raise FileNotFoundError(f"Nessun TIF trovato per {label}.")

    with rasterio.open(tif_files[0]) as src:
        crs_str    = src.crs.to_string() if src.crs else "EPSG:32734"
        pixel_size = abs(src.transform.a)
    transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
    print(f"    CRS: {crs_str} | pixel: {pixel_size:.1f} m")

    frames, band_names, first_transform = [], None, None
    for f in tif_files:
        t0 = time.time()
        df, bn, tr = extract_tile(f, transformer, clip_window, clip_transform)
        if band_names is None:
            band_names, first_transform = bn, tr
        print(f"      -> {len(df):,} px ({time.time()-t0:.1f}s)")
        if len(df) > 0:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            f"Nessun pixel valido da {label}. "
            "Controllare diagnostics GEE sezione [5] e [6]."
        )

    merged = pd.concat(frames, ignore_index=True)
    _purge(*frames, label=f"tile concat {label} ({len(merged):,} px grezzo)")

    before = len(merged)
    merged = merged.drop_duplicates(subset=["x_utm","y_utm"]).reset_index(drop=True)
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


# ══════════════════════════════════════════════════════════════════════
# SHAPEFILE SPATIAL MASKING
# (rasterizzazione su griglia TIF, O(n) sulla dimensione dei pixel)
# ══════════════════════════════════════════════════════════════════════

def _load_and_reproject_shp(shp_path, target_crs_str):
    """
    Carica uno shapefile e lo riproietta nel CRS del raster.
    Restituisce una lista di geometrie shapely valide.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas richiesto per il filtro shapefile. "
                          "Installare con: pip install geopandas")

    p = Path(shp_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Shapefile non trovato: {p}\n"
            "Verificare il percorso in FNF_SHAPEFILE / ELIGIBLE_SHAPEFILE."
        )

    t0 = time.time()
    gdf = gpd.read_file(str(p))
    print(f"    Caricato {p.name}: {len(gdf)} poligoni | CRS: {gdf.crs}")

    if gdf.crs is None:
        raise ValueError(f"Shapefile {p.name} non ha CRS definito.")

    if str(gdf.crs).upper().replace(" ","") != target_crs_str.upper().replace(" ",""):
        gdf = gdf.to_crs(target_crs_str)
        print(f"    Riproiettato a {target_crs_str} ({time.time()-t0:.2f}s)")
    else:
        print(f"    CRS già corretto ({time.time()-t0:.2f}s)")

    geoms = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
    print(f"    Geometrie valide: {len(geoms)}")
    return geoms


def rasterize_shp_to_tif_grid(shp_path, ref_tif_path, target_crs_str,
                                all_touched=False,
                                clip_window=None, clip_transform=None):
    """
    Rasterizza uno shapefile sulla griglia esatta del TIF di riferimento.

    Se clip_window è fornita, rasterizza sulla finestra clippata (stesse
    dimensioni del donor_df già letto con clip I/O): la mask risultante è
    direttamente indicizzabile tramite clip_transform.

    Approccio:
      - legge transform, height, width dal TIF (senza caricare i dati)
      - rasterizza le geometrie del shapefile → array uint8
      - 1 = pixel dentro un poligono, 0 = fuori

    Vantaggi vs point-in-polygon:
      - O(n_pixel_raster) invece di O(n_punti × n_poligoni)
      - Nessuna copia degli array di coordinate in RAM
      - Indicizzazione diretta mask[x_utm, y_utm] tramite transform
    """
    from rasterio.features import rasterize as rio_rasterize

    geoms = _load_and_reproject_shp(shp_path, target_crs_str)

    with rasterio.open(ref_tif_path) as src:
        base_transform = src.transform
        if clip_window is not None:
            import rasterio.windows as _rw
            transform = clip_transform or _rw.transform(clip_window, base_transform)
            height    = max(1, int(np.round(clip_window.height)))
            width     = max(1, int(np.round(clip_window.width)))
        else:
            transform = base_transform
            height    = src.height
            width     = src.width

    if not geoms:
        print(f"    WARN: nessuna geometria valida in {Path(shp_path).name}. "
              "Mask impostata a 0 (nessun pixel incluso).")
        return np.zeros((height, width), dtype=np.uint8), transform

    t0 = time.time()
    mask = rio_rasterize(
        geoms,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=all_touched,
    )
    n_inside = int(mask.sum())
    print(f"    Rasterizzato {Path(shp_path).name}: "
          f"{n_inside:,} px dentro i poligoni "
          f"({n_inside / (height*width) * 100:.1f}% della griglia) "
          f"— {time.time()-t0:.2f}s")
    return mask, transform


def filter_df_by_mask(df, mask, ref_transform, inside=True, label=""):
    """
    Filtra le righe del DataFrame in base alla mask rasterizzata.

    Usa x_utm/y_utm per trovare (row, col) nella mask tramite il transform,
    poi indicizza mask[row, col]. Nessun punto-in-poligono.

    inside=True  → mantiene pixel con mask==1 (dentro i poligoni)
    inside=False → mantiene pixel con mask==0 (fuori dai poligoni)
    """
    xs = df["x_utm"].values.astype(np.float64)
    ys = df["y_utm"].values.astype(np.float64)

    # (x_utm, y_utm) → (col, row) nella griglia del TIF
    # transform.c = origin_x (angolo top-left)
    # transform.f = origin_y (angolo top-left, valore positivo per nord-up)
    # transform.a = pixel_width (positivo)
    # transform.e = pixel_height (negativo per nord-up)
    mask_cols = np.floor((xs - ref_transform.c) / ref_transform.a).astype(np.int64)
    mask_rows = np.floor((ref_transform.f - ys) / abs(ref_transform.e)).astype(np.int64)

    # Clamp entro i limiti della griglia
    h, w      = mask.shape
    in_grid   = ((mask_rows >= 0) & (mask_rows < h) &
                 (mask_cols >= 0) & (mask_cols < w))

    pixel_vals = np.zeros(len(df), dtype=np.uint8)
    pixel_vals[in_grid] = mask[mask_rows[in_grid], mask_cols[in_grid]]

    # Pixel fuori dalla griglia → considerati fuori dal poligono (val=0)
    keep       = (pixel_vals == 1) if inside else (pixel_vals == 0)
    n_kept     = int(keep.sum())
    n_removed  = int(len(df) - n_kept)
    msg        = "dentro" if inside else "fuori"

    print(f"    {label}: {n_removed:,} px rimossi ({msg} shapefile) "
          f"| {n_kept:,} mantenuti "
          f"| {int(in_grid.sum() - n_kept) if inside else 0:,} out-of-grid trattati come esclusi")

    del xs, ys, mask_cols, mask_rows, in_grid, pixel_vals
    return df.loc[keep].reset_index(drop=True), n_kept, n_removed


def apply_shapefile_filters(df, label, proj_or_donor,
                             ref_tif_path, crs_utm,
                             audit_dict=None,
                             clip_window=None, clip_transform=None,
                             fnf_shapefile=None,
                             eligible_shapefile=None):
    """
    Applica i due filtri shapefile a un DataFrame di pixel raster.

    Per DONOR (proj_or_donor='donor'):
      1. Eligible_FNF_fullBuffer.shp: mantieni pixel DENTRO (aree eleggibili)
      2. FNF18_fullBuffer.shp:        rimuovi pixel DENTRO (foresta a T0)

    Per PA (proj_or_donor='project'):
      1. FNF18_fullBuffer.shp:        rimuovi pixel DENTRO (foresta a T0)
      (Eligible_FNF non applicato alla PA: la PA ha la propria geometria GEE)

    clip_window / clip_transform: se forniti, la rasterizzazione avviene sulla
    finestra clippata (stesso extent del donor già letto con clip I/O).
    Entrambi i filtri possono essere disattivati impostando i path a None.

    fnf_shapefile / eligible_shapefile: path espliciti (override dei globali).
    """
    # Fallback ai globali modulo solo se non forniti come parametro
    _fnf  = fnf_shapefile      if fnf_shapefile      is not None else FNF_SHAPEFILE
    _elig = eligible_shapefile if eligible_shapefile is not None else ELIGIBLE_SHAPEFILE

    n_start = len(df)
    ad      = audit_dict or {}

    print(f"\n    Filtri shapefile per {label} ({n_start:,} px input):")

    # ── ELIGIBLE filter (solo donor) ──────────────────────────────────
    if proj_or_donor == "donor" and _elig is not None:
        print(f"    [A] Eligible_FNF: mantieni solo pixel DENTRO (non-forest eleggibili)")
        try:
            elig_mask, elig_tr = rasterize_shp_to_tif_grid(
                _elig, ref_tif_path, crs_utm, SHP_ALL_TOUCHED,
                clip_window=clip_window, clip_transform=clip_transform)
            df, n_kept, n_rem = filter_df_by_mask(
                df, elig_mask, elig_tr, inside=True,
                label="  Eligible_FNF (dentro=mantieni)")
            del elig_mask
            _purge(label=f"dopo Eligible filter donor ({n_rem:,} rimossi)")
            ad["eligible_shp_removed_n"] = int(n_rem)
            ad["eligible_shp_kept_n"]    = int(n_kept)
        except Exception as e:
            print(f"    WARN Eligible filter: {e}")
            ad["eligible_shp_error"] = str(e)
    else:
        ad["eligible_shp_applied"] = False

    # ── FNF18 filter (donor + PA) ─────────────────────────────────────
    if _fnf is not None:
        print(f"    [B] FNF18: rimuovi pixel DENTRO (foresta a T0)")
        try:
            fnf_mask, fnf_tr = rasterize_shp_to_tif_grid(
                _fnf, ref_tif_path, crs_utm, SHP_ALL_TOUCHED,
                clip_window=clip_window, clip_transform=clip_transform)
            df, n_kept, n_rem = filter_df_by_mask(
                df, fnf_mask, fnf_tr, inside=True,
                label="  FNF18 (dentro=rimuovi)")
            del fnf_mask
            _purge(label=f"dopo FNF filter {label} ({n_rem:,} rimossi)")
            ad["fnf18_shp_removed_n"] = int(n_rem)
            ad["fnf18_shp_kept_n"]    = int(n_kept)
        except Exception as e:
            print(f"    WARN FNF18 filter: {e}")
            ad["fnf18_shp_error"] = str(e)
    else:
        ad["fnf18_shp_applied"] = False

    n_end = len(df)
    print(f"    {label}: {n_start:,} → {n_end:,} px "
          f"({n_start-n_end:,} rimossi dai filtri shapefile)")
    if n_end == 0:
        raise RuntimeError(
            f"Shapefile filters hanno rimosso TUTTI i pixel {label}.\n"
            "Verificare:\n"
            "  1. I path FNF_SHAPEFILE e ELIGIBLE_SHAPEFILE siano corretti.\n"
            "  2. I shapefile coprano l'area di studio.\n"
            "  3. Il CRS dei shapefile sia riproiettabile a quello del raster."
        )
    return df


# ── PLOT ─────────────────────────────────────────────────────────────

def _boxes(ax, xmin_arr, ymin_arr, pix, face, edge, alpha, lw, label=None):
    if len(xmin_arr) == 0:
        return
    rects = [Rectangle((float(x0), float(y0)), pix, pix)
             for x0, y0 in zip(xmin_arr, ymin_arr)]
    coll = PatchCollection(rects, facecolor=face, edgecolor=edge,
                            linewidth=lw, alpha=alpha, label=label)
    ax.add_collection(coll)
    xs = np.concatenate([xmin_arr, xmin_arr + pix])
    ys = np.concatenate([ymin_arr, ymin_arr + pix])
    ax.dataLim.update_from_data_xy(np.column_stack([xs, ys]), ignore=False)
    ax.autoscale_view()


def _qlim(ax, xs_all, ys_all, pix, q=0.001, mfrac=0.04):
    xs = np.asarray(xs_all, dtype=np.float64)
    ys = np.asarray(ys_all, dtype=np.float64)
    xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
    if len(xs) == 0:
        return
    x0, x1 = float(np.quantile(xs, q)), float(np.quantile(xs, 1-q))
    y0, y1 = float(np.quantile(ys, q)), float(np.quantile(ys, 1-q))
    mx = max((x1-x0)*mfrac, pix*3)
    my = max((y1-y0)*mfrac, pix*3)
    ax.set_xlim(x0-mx, x1+mx)
    ax.set_ylim(y0-my, y1+my)


def plot_aligned_pixel_grids(proj_df, donor_df, meta, out_dir=None):
    """
    Figura 1 — 4 panel:
      A) Scatter WGS84 (overview PA + donor)
      B) PA grid UTM (celle reali 30 m)
      C) Donor grid UTM (extent completo donor campionato)
      D) Zoom bordo PA est (regione densa rilevata via 2D histogram)

    Figura 2 — dettaglio zoom UTM al centro PA.
    """
    t_total = time.time()
    crs_utm = meta.get("crs_src", "EPSG:32734")
    pix     = float(meta.get("pixel_size_m", 30.0))
    rng     = np.random.default_rng(42)

    for col in ("x_utm", "y_utm", "cell_xmin", "cell_ymin"):
        if col not in proj_df.columns or col not in donor_df.columns:
            raise ValueError(f"Colonna '{col}' mancante. Rigenerare i parquet.")

    px  = proj_df["x_utm"].to_numpy(np.float64)
    py  = proj_df["y_utm"].to_numpy(np.float64)
    dx  = donor_df["x_utm"].to_numpy(np.float64)
    dy  = donor_df["y_utm"].to_numpy(np.float64)
    px0 = proj_df["cell_xmin"].to_numpy(np.float64)
    py0 = proj_df["cell_ymin"].to_numpy(np.float64)
    dx0 = donor_df["cell_xmin"].to_numpy(np.float64)
    dy0 = donor_df["cell_ymin"].to_numpy(np.float64)

    def _samp(n, cap):
        return np.arange(n, dtype=np.int64) if n <= cap else rng.choice(n, cap, replace=False)

    p_idx = _samp(len(proj_df),  PLOT_MAX_PA_BOXES)
    d_idx = _samp(len(donor_df), PLOT_MAX_DONOR_BOXES)

    # Panel D zoom: regione più densa via 2D histogram
    bin_m  = pix * 25
    x_edges = np.arange(px.min()-bin_m, px.max()+2*bin_m, bin_m)
    y_edges = np.arange(py.min()-bin_m, py.max()+2*bin_m, bin_m)
    if len(x_edges) >= 2 and len(y_edges) >= 2:
        H, xe, ye = np.histogram2d(px, py, bins=[x_edges, y_edges])
        ix, iy = np.unravel_index(H.argmax(), H.shape)
        dense_cx = float((xe[ix]+xe[ix+1])/2)
        dense_cy = float((ye[iy]+ye[iy+1])/2)
    else:
        dense_cx, dense_cy = float(px.mean()), float(py.mean())

    hw_d = pix * 30 / 2   # zoom panel D: 900 m

    t0  = time.time()
    fig1, axes = plt.subplots(1, 4, figsize=(28, 7), facecolor="white")

    # Panel A — scatter WGS84
    ax = axes[0]
    n_d_sc = min(15_000, len(donor_df))
    sc_d   = rng.choice(len(donor_df), n_d_sc, replace=False)
    ax.scatter(donor_df["lon"].values[sc_d], donor_df["lat"].values[sc_d],
               s=0.4, alpha=0.18, color="steelblue",
               label=f"Donor ({len(donor_df):,}, sc={n_d_sc:,})")
    ax.scatter(proj_df["lon"].values, proj_df["lat"].values,
               s=2, alpha=0.70, color="tomato",
               label=f"PA ({len(proj_df):,})")
    ax.set(xlabel="Longitudine", ylabel="Latitudine",
           title="A — Overview WGS84\n(dopo filtri shapefile)")
    ax.legend(fontsize=7, markerscale=4); ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # Panel B — PA grid UTM
    ax = axes[1]
    _boxes(ax, px0[p_idx], py0[p_idx], pix, "tomato", "darkred", 0.75, 0.15,
           f"PA ({len(p_idx):,}/{len(proj_df):,})")
    _qlim(ax, np.concatenate([px0[p_idx], px0[p_idx]+pix]),
               np.concatenate([py0[p_idx], py0[p_idx]+pix]), pix)
    ax.set(xlabel=f"Easting ({crs_utm})", ylabel="Northing",
           title=f"B — PA grid UTM\n{pix:.0f}×{pix:.0f} m celle reali")
    ax.set_aspect("equal")

    # Panel C — Donor grid UTM extent completo
    ax = axes[2]
    _boxes(ax, dx0[d_idx], dy0[d_idx], pix, "#5b9bd5", "navy", 0.55, 0.10,
           f"Donor ({len(d_idx):,}/{len(donor_df):,})")
    _qlim(ax, np.concatenate([dx0[d_idx], dx0[d_idx]+pix]),
               np.concatenate([dy0[d_idx], dy0[d_idx]+pix]), pix)
    ax.set(xlabel=f"Easting ({crs_utm})", ylabel="Northing",
           title=f"C — Donor grid UTM (dopo filtri)\nextent completo donor")
    ax.set_aspect("equal")

    # Panel D — Zoom zona densa PA + donor
    ax = axes[3]
    mp = ((px  >= dense_cx-hw_d) & (px  <= dense_cx+hw_d) &
          (py  >= dense_cy-hw_d) & (py  <= dense_cy+hw_d))
    md = ((dx  >= dense_cx-hw_d) & (dx  <= dense_cx+hw_d) &
          (dy  >= dense_cy-hw_d) & (dy  <= dense_cy+hw_d))
    if md.any():
        _boxes(ax, dx0[md], dy0[md], pix, "#5b9bd5", "navy", 0.55, 0.3)
    if mp.any():
        _boxes(ax, px0[mp], py0[mp], pix, "tomato", "darkred", 0.80, 0.3)
    for xg in np.arange(dense_cx-hw_d, dense_cx+hw_d+pix, pix):
        ax.axvline(xg, color="gray", lw=0.2, alpha=0.3)
    for yg in np.arange(dense_cy-hw_d, dense_cy+hw_d+pix, pix):
        ax.axhline(yg, color="gray", lw=0.2, alpha=0.3)
    ax.set_xlim(dense_cx-hw_d, dense_cx+hw_d)
    ax.set_ylim(dense_cy-hw_d, dense_cy+hw_d)
    ax.set(xlabel=f"Easting ({crs_utm})",
           title=f"D — Zoom zona densa ({hw_d*2:.0f}m)\nPA ({mp.sum()}) + Donor ({md.sum()})")
    ax.set_aspect("equal")
    ax.legend(handles=[
        mpatches.Patch(facecolor="tomato",  edgecolor="darkred", label="PA"),
        mpatches.Patch(facecolor="#5b9bd5", edgecolor="navy",    label="Donor"),
    ], fontsize=8, loc="upper right")

    plt.suptitle(
        f"GS STARR — Griglie pixel {pix:.0f}m | {crs_utm} | {meta.get('run_id','')}\n"
        "Filtri shapefile applicati: FNF18 (esclude foresta) + Eligible_FNF (donor eleggibili)",
        fontsize=9)
    plt.tight_layout()
    fig1.canvas.draw()
    print(f"    Fig1: {time.time()-t0:.2f}s")
    if out_dir:
        fig1.savefig(Path(out_dir)/"pixel_grids_aligned.png",
                     dpi=150, bbox_inches="tight", facecolor="white")
    _purge(fig1, dx, dy, dx0, dy0, d_idx, close_figs=True,
           label="fig1 chiusa + array donor liberati")

    # Figura 2 — dettaglio UTM zona densa (solo PA)
    t0 = time.time()
    hw2  = pix * 15 / 2   # zoom fig2: 450 m
    fig2, ax2 = plt.subplots(figsize=(8, 8), facecolor="white")
    mp2 = ((px  >= dense_cx-hw2) & (px  <= dense_cx+hw2) &
           (py  >= dense_cy-hw2) & (py  <= dense_cy+hw2))
    pz2 = px0[mp2]; pz2y = py0[mp2]
    _boxes(ax2, pz2, pz2y, pix, "tomato", "darkred", 0.85, 0.5)
    for xg in np.arange(dense_cx-hw2, dense_cx+hw2+pix, pix):
        ax2.axvline(xg, color="gray", lw=0.4, alpha=0.4)
    for yg in np.arange(dense_cy-hw2, dense_cy+hw2+pix, pix):
        ax2.axhline(yg, color="gray", lw=0.4, alpha=0.4)
    ax2.set_xlim(dense_cx-hw2, dense_cx+hw2)
    ax2.set_ylim(dense_cy-hw2, dense_cy+hw2)
    ax2.set(xlabel=f"Easting UTM ({crs_utm}) — m",
            ylabel="Northing UTM — m",
            title=f"Dettaglio griglia PA {pix:.0f}×{pix:.0f} m\n"
                  f"Finestra {hw2*2:.0f}m — {mp2.sum()} celle")
    ax2.set_aspect("equal")
    ax2.text(0.02, 0.98,
             f"PA totali  : {len(proj_df):,} px\n"
             f"Donor tot  : {len(donor_df):,} px\n"
             f"CRS        : {crs_utm}\n"
             f"Pixel size : {pix:.0f} m\n"
             f"FNF filter : {'ON' if FNF_SHAPEFILE else 'OFF'}\n"
             f"Eligible   : {'ON' if ELIGIBLE_SHAPEFILE else 'OFF'}",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))
    plt.tight_layout()
    fig2.canvas.draw()
    print(f"    Fig2: {time.time()-t0:.2f}s")
    if out_dir:
        fig2.savefig(Path(out_dir)/"pixel_grid_detail_UTM.png",
                     dpi=150, bbox_inches="tight", facecolor="white")
    _purge(fig2, px, py, px0, py0, pz2, pz2y, mp2, mp, close_figs=True,
           label="fig2 chiusa + array PA liberati")

    print(f"    TOTAL plotting: {time.time()-t_total:.2f}s")
    return None, None


# ── PLOT COVARIATE ────────────────────────────────────────────────────

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
        ax.hist(pv, bins=bins, alpha=0.7, color="tomato",    density=True,
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
        fig.savefig(Path(out_dir)/"covariate_distributions.png", dpi=150, bbox_inches="tight")
    return fig


def plot_ndvi_valid_years(proj_df, donor_df, out_dir=None):
    if "ndvi_valid_years" not in proj_df.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, df, label, color in [
        (axes[0], proj_df, "Project", "tomato"),
        (axes[1], donor_df, "Donor",  "steelblue"),
    ]:
        cnt  = df["ndvi_valid_years"].value_counts().sort_index()
        bars = ax.bar(cnt.index.astype(str), cnt.values, color=color,
                      edgecolor="white", width=0.6)
        ax.set(xlabel="Anni NDVI validi", ylabel="Pixel",
               title=f"{label}\nDistribuzione anni NDVI validi")
        ax.grid(alpha=0.3, axis="y")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,
                    b.get_height()+max(cnt.values)*0.01,
                    f"{int(b.get_height()):,}", ha="center", fontsize=8)
    plt.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir)/"ndvi_valid_years.png", dpi=150, bbox_inches="tight")
    return fig


# ── MAIN ─────────────────────────────────────────────────────────────

def run_extraction(base_dirs=None, output_dir=None, verbose=True,
                   donor_extent_km="full",
                   run_id_base=None,
                   fnf_shapefile=None,
                   eligible_shapefile=None):
    """
    Restituisce (proj_df, donor_df, meta, out_dir).

    Parameters
    ----------
    donor_extent_km : float | "full"
        Estensione del pool donor attorno al bordo della PA.
        "full" → nessun clip (comportamento originale).
        5/10/20/30 → clip I/O del TIF donor al bbox PA + buffer(N km).
        Il clip avviene PRIMA di qualsiasi lettura in RAM: nessun pixel
        oltre il buffer viene mai caricato.

    Pipeline:
      1. Carica TIF progetto + donor da Drive (block-by-block, evita OOM)
      2. Applica edge exclusion e sample spaziale alla PA
      3. [NEW] Calcola clip window donor = bbox(PA) + buffer(donor_extent_km)
      4. Carica TIF donor SOLO nella finestra clippata
      5. Applica FNF18 + Eligible_FNF shapefile filters (su griglia clippata)
      6. Salva parquet + report JSON (in directory isolata per extent)
      7. Plot griglie pixel UTM
    """
    # ── Risoluzione parametri (parametro > globale modulo) ────────────
    _run_id_base     = run_id_base     if run_id_base     is not None else RUN_ID_BASE
    _fnf_shapefile   = fnf_shapefile   if fnf_shapefile   is not None else FNF_SHAPEFILE
    _eligible_shp    = eligible_shapefile if eligible_shapefile is not None else ELIGIBLE_SHAPEFILE

    # ── Pattern TIF: se RUN_ID_BASE è vuoto usa wildcard pura ─────────
    # _id_frag senza "_" finale: "covariates_project_{id}*.tif" batte
    # sia "...{id}.tif" sia "...{id}_extra.tif".
    # Con "_" finale il pattern non matcherebbe file senza suffisso aggiuntivo.
    _id_frag = _run_id_base if _run_id_base else ""
    _proj_pattern = f"covariates_project_{_id_frag}*.tif"
    _donor_pattern = f"covariates_donor_{_id_frag}*.tif"

    # ── Effective RUN_ID (include suffisso extent) ────────────────────
    if donor_extent_km == "full":
        effective_run_id = _run_id_base or "run"
        extent_label     = "FULL"
    else:
        effective_run_id = f"{_run_id_base or 'run'}_ext{donor_extent_km}km"
        extent_label     = f"{donor_extent_km}km"

    if base_dirs is None:
        base_dirs = [Path(p) for p in BASE_DIR_CANDIDATES if Path(p).exists()]
    if not base_dirs:
        raise FileNotFoundError(
            f"Nessuna directory trovata. Passare base_dirs a run_extraction() "
            f"oppure impostare BASE_DIR_CANDIDATES nel modulo.\n"
            f"Candidati attuali: {BASE_DIR_CANDIDATES}"
        )
    base_dirs = [Path(b) for b in base_dirs]

    out_dir = (Path(output_dir) if output_dir
               else base_dirs[0] / "STARR_outputs" / effective_run_id / "01_extract")
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*65}")
        print(f"STEP 01 | {effective_run_id}")
        print(f"Donor extent       : {extent_label}")
        print(f"FNF shapefile      : {Path(_fnf_shapefile).name if _fnf_shapefile else 'OFF'}")
        print(f"Eligible shapefile : {Path(_eligible_shp).name if _eligible_shp else 'OFF'}")
        print(f"Output: {out_dir}")
        print(f"{'='*65}")

    shp_audit = {}

    print("\n[1] Ricerca tiles...")
    proj_tiles  = find_tif_tiles(base_dirs, _proj_pattern, "Project")
    donor_tiles = find_tif_tiles(base_dirs, _donor_pattern, "Donor")

    # ── ESTRAZIONE PROJECT ────────────────────────────────────────────
    print("\n[2] Estrazione PROJECT...")
    _purge(label="[START] RAM prima project")
    (proj_df, band_names, ndvi_year_cols, year_list,
     t0_year, cont_covs, crs_src, pixel_size, proj_transform) = \
        load_raster_to_dataframe(proj_tiles, "Project")

    print("\n[2b] Filtri PA (edge + sample)...")
    n_pa_raw = len(proj_df)
    proj_df, n_edge = exclude_edge_pixels(proj_df, pixel_size)
    proj_df, sampled = spatially_stratified_sample(proj_df)
    _purge(label=f"filtri PA edge/sample ({n_pa_raw:,}->{len(proj_df):,})")

    # ── FILTRO SHAPEFILE PA (FNF18 only) ─────────────────────────────
    if proj_tiles and (_fnf_shapefile is not None):
        print("\n[2c] Filtro shapefile PA (FNF18 — rimuove pixel foresta a T0)...")
        n_before = len(proj_df)
        proj_df  = apply_shapefile_filters(
            proj_df, "PA", "project",
            ref_tif_path=str(proj_tiles[0]),
            crs_utm=crs_src,
            audit_dict=shp_audit.setdefault("project", {}),
            clip_window=None,   # PA: sempre full (nessun clip)
            clip_transform=None,
            fnf_shapefile=_fnf_shapefile,
            eligible_shapefile=None,  # FNF only per PA
        )
        _purge(label=f"FNF PA filter ({n_before:,}->{len(proj_df):,})")

    # ── CLIP WINDOW DONOR ─────────────────────────────────────────────
    # Calcolata DOPO proj_df (bbox reale PA) e PRIMA di caricare il donor.
    # Il clip è a livello I/O: nessun byte fuori dal buffer viene letto.
    print(f"\n[2d] Clip window donor (extent={extent_label})...")
    donor_clip_win, donor_clip_tr = None, None
    if donor_tiles:
        donor_clip_win, donor_clip_tr = compute_donor_clip_window(
            proj_df, donor_tiles[0], donor_extent_km)

    # ── ESTRAZIONE DONOR ──────────────────────────────────────────────
    print("\n[3] Estrazione DONOR...")
    _purge(label="RAM prima donor")
    (donor_df, *_, donor_transform) = load_raster_to_dataframe(
        donor_tiles, "Donor",
        clip_window=donor_clip_win,
        clip_transform=donor_clip_tr,
    )

    # ── FILTRO SHAPEFILE DONOR (Eligible_FNF + FNF18) ─────────────────
    if donor_tiles and (_fnf_shapefile is not None or _eligible_shp is not None):
        print("\n[3b] Filtro shapefile DONOR (Eligible_FNF + FNF18)...")
        n_before = len(donor_df)
        donor_df = apply_shapefile_filters(
            donor_df, "DONOR", "donor",
            ref_tif_path=str(donor_tiles[0]),
            crs_utm=crs_src,
            audit_dict=shp_audit.setdefault("donor", {}),
            clip_window=donor_clip_win,
            clip_transform=donor_clip_tr,
            fnf_shapefile=_fnf_shapefile,
            eligible_shapefile=_eligible_shp,
        )
        _purge(label=f"SHP donor filter ({n_before:,}->{len(donor_df):,})")

    # ── METADATA ─────────────────────────────────────────────────────
    tr_dict = None
    if proj_transform is not None:
        tr_dict = {k: getattr(proj_transform, k)
                   for k in ("a", "b", "c", "d", "e", "f")}

    meta = {
        "run_id":                effective_run_id,
        "run_id_base":           RUN_ID_BASE,
        "project_name":          PROJECT_NAME,
        "donor_extent_km":       donor_extent_km,
        "donor_extent_label":    extent_label,
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
    _purge(label="dopo salvataggio parquet")

    # ── 3× guideline (A2 fix) ─────────────────────────────────────────
    # La linea guida 3× confronta l'AREA donor ELEGGIBILE con l'AREA PA piena,
    # NON il conteggio di pixel donor grezzi vs PA sottocampionata.
    # project_n qui è già post edge-exclusion e post spatial-sample (150k cap),
    # quindi il rapporto sui conteggi è solo un PROXY DIAGNOSTICO, non il test
    # di compliance. Riportiamo entrambi i conteggi + area in ha quando il
    # pixel è a risoluzione metrica nota, e marchiamo esplicitamente il proxy.
    pixel_area_ha = (abs(pixel_size * pixel_size) / 10_000.0) if pixel_size else None
    donor_area_ha = (len(donor_df) * pixel_area_ha) if pixel_area_ha else None
    # PA piena = righe PA grezze (pre-sample) × area pixel: stima dell'area PA
    # effettivamente eleggibile prima del sottocampionamento spaziale.
    pa_full_area_ha = (n_pa_raw * pixel_area_ha) if pixel_area_ha else None
    ratio_count_proxy = round(len(donor_df) / max(len(proj_df), 1), 2)
    ratio_area_eligible = (
        round(donor_area_ha / pa_full_area_ha, 2)
        if (donor_area_ha and pa_full_area_ha) else None
    )
    meets_3x_area = (
        bool(donor_area_ha >= 3.0 * pa_full_area_ha)
        if (donor_area_ha and pa_full_area_ha) else None
    )

    report = {
        **{k: v for k, v in meta.items() if k != "raster_transform"},
        "project_n":        int(len(proj_df)),
        "project_n_raw_pre_sample": int(n_pa_raw),
        "donor_n":          int(len(donor_df)),
        # Proxy sui conteggi (NON è il test di compliance): PA è sottocampionata.
        "ratio_donor_project_count_proxy": ratio_count_proxy,
        "ratio_donor_project": ratio_count_proxy,  # alias retro-compat
        # Test 3× corretto: area donor eleggibile vs area PA piena.
        "donor_area_ha_eligible": donor_area_ha,
        "pa_full_area_ha_estimate": pa_full_area_ha,
        "ratio_donor_project_area": ratio_area_eligible,
        "meets_3x_guideline_area": meets_3x_area,
        # alias retro-compat: ora punta al test AREA (corretto), non ai conteggi.
        "meets_3x_guideline": meets_3x_area if meets_3x_area is not None else (len(donor_df) >= 3 * n_pa_raw),
        "meets_3x_note": (
            "meets_3x_guideline ora confronta area donor eleggibile vs area PA "
            "piena (pre-sottocampionamento). ratio_donor_project_count_proxy è solo "
            "diagnostico perché project_n è post edge-exclusion + spatial-sample."
        ),
        "raster_transform": tr_dict,
        "pa_filter_audit": {
            "pa_raw_n":       int(n_pa_raw),
            "pa_filtered_n":  int(len(proj_df)),
            "pa_edge_excl_n": int(n_edge),
            "pa_sampled":     bool(sampled),
            "pa_sample_grid_step_m": PA_SPATIAL_GRID_STEP_M if sampled else None,
            "pa_sample_bias_note": (
                "A3: la media ΔC finale (Step 05) viene scalata all'area PA piena "
                "assumendo che il campione spaziale a "
                f"{PA_SPATIAL_GRID_STEP_M:.0f}m sia rappresentativo dell'intera PA. "
                "Con forte eterogeneità intra-PA del degrado/recupero la media "
                "campionaria può differire dalla media reale: documentare nel PDD."
            ) if sampled else "PA non sottocampionata: nessun bias di campionamento.",
        },
        "shapefile_filter_audit": shp_audit,
        "shapefile_paths": {
            "fnf18":          _fnf_shapefile,
            "eligible_fnf":   _eligible_shp,
            "all_touched":    SHP_ALL_TOUCHED,
        },
        "donor_clip": {
            "extent_km":      donor_extent_km,
            "extent_label":   extent_label,
            "clip_applied":   donor_clip_win is not None,
        },
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "output_dir":       str(out_dir),
    }
    with open(out_dir / "extraction_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n[5] Plot griglie pixel...")
    grid_figs = plot_aligned_pixel_grids(proj_df, donor_df, meta, out_dir=out_dir)
    _purge(label="dopo plot (figure chiuse)")

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Run ID           : {effective_run_id}")
        print(f"  Donor extent     : {extent_label}")
        print(f"  Project          : {len(proj_df):,} pixel  -> {p1.name}")
        print(f"  Donor            : {len(donor_df):,} pixel  -> {p2.name}")
        _r_area = report.get("ratio_donor_project_area")
        _m_area = report.get("meets_3x_guideline_area")
        if _r_area is not None:
            print(f"  Ratio AREA d/p   : {_r_area:.1f}×  "
                  f"({'✓ ≥3×' if _m_area else '✗ <3×'})  [test compliance]")
        print(f"  Ratio count proxy: {report['ratio_donor_project_count_proxy']:.1f}×  "
              f"(diagnostico — PA sottocampionata)")
        print(f"  T0 (auto)        : {t0_year}")
        print(f"  NDVI anni (auto) : {year_list}")
        print(f"  Covariate (auto) : {cont_covs}")
        print(f"  CRS              : {crs_src} | pixel {pixel_size:.0f} m")
        print(f"  FNF filter       : {'ON: ' + Path(_fnf_shapefile).name if _fnf_shapefile else 'OFF'}")
        print(f"  Eligible filter  : {'ON: ' + Path(_eligible_shp).name if _eligible_shp else 'OFF'}")
        print(f"  Output           : {out_dir}")
        print(f"{'='*65}")

    meta["_grid_figs"] = grid_figs
    return proj_df, donor_df, meta, out_dir


def main():
    run_extraction()


if __name__ == "__main__":
    main()