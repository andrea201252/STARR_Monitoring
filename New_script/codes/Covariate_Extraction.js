// ============================================================
// GOLD STANDARD STARR – Track 1 SEMDB
// RASTER-FIRST PIPELINE — v07
//
// GEE export strategy: export ALL pixels with valid covariates
// (NDVI, elevation, slope, SOC, WRB, roads) and non-permanent-water.
// The forest/non-forest filter and eligible areas are applied
// in Python (Step 01) using two local shapefiles on Drive.
//
// NEW IN v07:
//   - NDVI compositing by selectable SEASON (A growing / C dry / annual)
//     + statistic (median / p90 / max)  → NDVI_SEASON, NDVI_COMPOSITE_STAT
//   - Landsat: only L8 on request (USE_L9 = false)
//   - WRB mask realigned to the HWSD2 legend (excludes the true non-soils
//     12 Glaciers, 16 Islands, 34 Open Water, 35 No Data; 31=Technosols is soil)
//   - SOC (t/ha): mask nodata/0 so it does not enter the pool as zero
//   - DONOR diagnostics + PA vs DONOR covariate statistics at the source
// ============================================================


// ────────────────────────────────────────────────────────────
// 0. USER INPUTS
// ────────────────────────────────────────────────────────────

var PROJECT_ASSET = 'projects/giscentral-gee/assets/Mim_Murraca_Caia/ProjectArea_Muraca';

var ECOREGION_NAME = 'Zambezian-Limpopo mixed woodlands';
var COUNTRY_NAME   = 'Mozambique';

var T0_YEAR     = 2025;
var TREND_YEARS = 5;
var SCALE_M     = 30;

var RUN_ID        = 'Muraca_Caia_2025_buf50km_excl5km_WRB2_v09_raster';
var EXPORT_FOLDER = 'Muraca_Caia_New_V2';

// Donor search buffer around the PA.
var MAX_DISTANCE_FROM_PROJECT_KM = 50;
// Exclusion zone around the PA (leakage buffer).
var CONTROL_EXCLUSION_KM         = 5;

var CLIP_TO_COUNTRY = true;

var WATER_OCCURRENCE_THRESHOLD = 60;

var ROAD_ASSET             = 'projects/sat-io/open-datasets/GRIP4/Africa';
var ROAD_DISTANCE_SEARCH_M = 50000;

var CLOUD_COVER_MAX = 10;
var MIN_VALID_YEARS = 4;

var EXPORT_CRS = 'EPSG:32736';

var DIAGNOSTIC_SCALE_M = 1000;
var MAP_PREVIEW = true;

// ── NDVI compositing: SEASON + STATISTIC ──────────────────
// Southern hemisphere (Mozambique). Choose the annual NDVI composite window:
//   'growing' (A) = growing season: dec(year-1) → apr(year)  [green peak]
//   'dry'     (C) = dry season:      may → sep (same year)  [stable, few clouds]
//   'annual'      = full year apr→apr (original v06 behavior)
// Change ONLY this line to test A vs C without touching anything else.
var NDVI_SEASON         = 'growing';   // 'growing' | 'dry' | 'annual'
// Statistic of the seasonal composite:
//   'median' (robust), 'p90' (near the peak), 'max' (greenest pixel)
var NDVI_COMPOSITE_STAT = 'median';    // 'median' | 'p90' | 'max'

// Landsat: only L8 on request (L9 disabled).
var USE_L9 = false;                    // false = only L8 | true = L8 + L9


// ────────────────────────────────────────────────────────────
// 1. HELPERS
// ────────────────────────────────────────────────────────────

function fcFromGeom(geom) {
  return ee.FeatureCollection([ee.Feature(geom)]);
}

function safeIntersect(g1, g2) {
  return ee.Geometry(g1).intersection(ee.Geometry(g2), ee.ErrorMargin(100));
}

function cleanGeom(geom) {
  return ee.Geometry(geom)
    .buffer(0,    ee.ErrorMargin(1))
    .dissolve(    ee.ErrorMargin(1))
    .buffer(0,    ee.ErrorMargin(1));
}

function addAreaHa(f) {
  return f.set('area_ha', f.geometry().area(ee.ErrorMargin(1)).divide(10000));
}

function areaHa(maskImg, geom, scale) {
  return ee.Image.pixelArea().divide(10000)
    .updateMask(maskImg)
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: geom,
      scale: scale,
      maxPixels: 1e13,
      bestEffort: true,
      tileScale: 4
    }).get('area');
}

function polygonOnlyFc(fc, label) {
  fc = ee.FeatureCollection(fc).map(function(f) {
    // buffer(0) on a GeometryCollection returns a MultiPolygon
    // keeping only the polygonal parts — it is the most robust way in GEE
    var g0 = ee.Geometry(f.geometry()).buffer(0, ee.ErrorMargin(10));
    return ee.Feature(g0, f.toDictionary())
      .set('_gtype0', g0.type())
      .set('_src', label);
  });

  print(label + ' geometry type histogram:', fc.aggregate_histogram('_gtype0'));

  var out = fc
    .filter(ee.Filter.inList('_gtype0', ['Polygon', 'MultiPolygon']))
    .map(function(f) {
      return ee.Feature(cleanGeom(f.geometry()), {
        runid: RUN_ID, t0: T0_YEAR, src: label, typ: 'PA'
      });
    }).map(addAreaHa);

  print(label + ' polygon features kept:', out.size());
  return out;
}

function maskToPolygonFc(maskImg, geom, label, scale) {
  var maskBinary = ee.Image(1)
    .updateMask(maskImg)
    .rename('mask')
    .toByte()
    .clip(geom);
  var vectors = maskBinary.reduceToVectors({
    geometry: geom,
    crs: EXPORT_CRS,
    scale: scale,
    geometryType: 'polygon',
    eightConnected: true,
    labelProperty: 'maskval',
    reducer: ee.Reducer.countEvery(),
    maxPixels: 1e13,
    tileScale: 8,
    bestEffort: false
  });
  return vectors.map(function(f) {
    return ee.Feature(
      ee.Geometry(f.geometry()).buffer(0, ee.ErrorMargin(1)),
      { name: label, typ: label, runid: RUN_ID, t0: T0_YEAR,
        scale_m: scale, pixcnt: f.get('count') }
    );
  }).map(addAreaHa);
}


// ────────────────────────────────────────────────────────────
// 2. GEOMETRIES
// ────────────────────────────────────────────────────────────

var projectFc   = ee.FeatureCollection(PROJECT_ASSET);
var projectGeom = projectFc.geometry();
print('Project geometry type:', projectGeom.type());

var ecoregionFc = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
  .filter(ee.Filter.eq('ECO_NAME', ECOREGION_NAME));
var ecoGeom = cleanGeom(ecoregionFc.geometry());
var analysisEcoGeom = ecoGeom;
if (CLIP_TO_COUNTRY) {
  var countryFc = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level0')
    .filter(ee.Filter.eq('ADM0_NAME', COUNTRY_NAME));
  analysisEcoGeom = cleanGeom(safeIntersect(ecoGeom, countryFc.geometry()));
}

var projectBufferGeom    = cleanGeom(
  projectGeom.buffer(MAX_DISTANCE_FROM_PROJECT_KM * 1000, ee.ErrorMargin(100)));
var projectExclusionGeom = cleanGeom(
  projectGeom.buffer(CONTROL_EXCLUSION_KM * 1000, ee.ErrorMargin(100)));

// Landsat analysis zone (wide — needed to collect images)
var rawDonorSearchGeom = cleanGeom(
  safeIntersect(projectBufferGeom, analysisEcoGeom));

// Donor export zone: buffer ∩ ecoregion \ 5 km exclusion
// Does not apply forest/eligible filters — Python handles that with the shapefiles.
var donorExportGeom = cleanGeom(
  rawDonorSearchGeom.difference(projectExclusionGeom, ee.ErrorMargin(100)));

print('RUN_ID:', RUN_ID, '| T0:', T0_YEAR, '| CRS:', EXPORT_CRS, '| Scale:', SCALE_M, 'm');
print('NDVI season:', NDVI_SEASON, '| stat:', NDVI_COMPOSITE_STAT, '| USE_L9:', USE_L9);
print('Donor export geometry type:', donorExportGeom.type());
print('Forest/eligible filters: applied in Python (Step 01) with Drive shapefiles.');


// ────────────────────────────────────────────────────────────
// 3. LANDSAT — QA MASKING AND NDVI (seasonal, selectable)
// ────────────────────────────────────────────────────────────

function maskL89_QA(img) {
  var qa  = img.select('QA_PIXEL');
  var msk = qa.bitwiseAnd(1 << 0).eq(0)
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0));
  return img.updateMask(msk).updateMask(img.select('QA_RADSAT').eq(0));
}

function prepL89(img) {
  img = maskL89_QA(img);
  var red   = img.select('SR_B4').multiply(0.0000275).add(-0.2).rename('red');
  var nir   = img.select('SR_B5').multiply(0.0000275).add(-0.2).rename('nir');
  var denom = nir.add(red);
  return nir.subtract(red)
    .divide(denom)
    .rename('NDVI')
    .updateMask(denom.abs().gt(1e-6))
    .toFloat()
    .copyProperties(img, ['system:time_start']);
}

// Landsat collection. USE_L9=false → only L8 (request). true → L8 + L9.
function l89Collection(startDate, endDate, geom) {
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate(startDate, endDate)
    .filterBounds(geom)
    .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_COVER_MAX))
    .map(prepL89);
  if (!USE_L9) return l8;
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterDate(startDate, endDate)
    .filterBounds(geom)
    .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_COVER_MAX))
    .map(prepL89);
  return l8.merge(l9);
}

// Time window of the annual composite for the chosen season.
// 'year' = reference year of the composite.
function seasonWindow(year) {
  year = ee.Number(year);
  if (NDVI_SEASON === 'growing') {
    // A — southern hemisphere growing season: dec(year-1) → apr(year) inclusive
    return { start: ee.Date.fromYMD(year.subtract(1), 12, 1),
             end:   ee.Date.fromYMD(year, 5, 1) };
  } else if (NDVI_SEASON === 'dry') {
    // C — dry season: may → sep (same year)
    return { start: ee.Date.fromYMD(year, 5, 1),
             end:   ee.Date.fromYMD(year, 10, 1) };
  }
  // 'annual' — full year apr→apr (original v06)
  return { start: ee.Date.fromYMD(year, 4, 1),
           end:   ee.Date.fromYMD(year, 4, 1).advance(1, 'year') };
}

// Statistic of the composite → 'NDVI' band.
function compositeNDVI(col) {
  if (NDVI_COMPOSITE_STAT === 'p90') {
    return col.reduce(ee.Reducer.percentile([90])).rename('NDVI');
  } else if (NDVI_COMPOSITE_STAT === 'max') {
    return col.max().rename('NDVI');
  }
  return col.median().rename('NDVI');
}

function annualNDVI(year) {
  year = ee.Number(year);
  var w   = seasonWindow(year);
  var col = l89Collection(w.start, w.end, rawDonorSearchGeom);
  var empty = ee.Image.constant(0).toFloat().rename('NDVI')
               .updateMask(ee.Image.constant(0));
  return ee.Image(ee.Algorithms.If(
    col.size().gt(0),
    compositeNDVI(col).toFloat().clip(rawDonorSearchGeom),
    empty
  ));
}

print('--- Landsat scenes (' + (USE_L9 ? 'L8+L9' : 'only L8')
      + ') per season [' + NDVI_SEASON + '] ---');
ee.List.sequence(T0_YEAR - TREND_YEARS, T0_YEAR).evaluate(function(years) {
  years.forEach(function(yr) {
    var w = seasonWindow(yr);
    print('Year ' + yr + ':', l89Collection(w.start, w.end, rawDonorSearchGeom).size());
  });
});


// ────────────────────────────────────────────────────────────
// 4. MASKS — PERMANENT WATER ONLY
//
// In this version NO forest/nonforest filter is applied
// in GEE. All pixels with valid covariates are exported.
// The forest/eligible filtering is applied in Python (Step 01)
// using the shapefiles on Drive:
//   FNF18_fullBuffer.shp       -> excludes forest areas at T0
//   Eligible_FNF_fullBuffer.shp -> keeps only eligible areas
// ────────────────────────────────────────────────────────────

var permanentWaterMask = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence')
  .unmask(0)
  .gte(WATER_OCCURRENCE_THRESHOLD)
  .clip(rawDonorSearchGeom)
  .rename('permanent_water');

// Mask only permanent water; the forest filter is delegated to Python.
var baseMaskDonor   = permanentWaterMask.not().clip(donorExportGeom);
var baseMaskProject = permanentWaterMask.not().clip(projectGeom);


// ────────────────────────────────────────────────────────────
// 5. STATIC COVARIATES
// ────────────────────────────────────────────────────────────

var elevation = ee.Image("USGS/SRTMGL1_003")
  .select('elevation').rename('elevation').toFloat().clip(rawDonorSearchGeom);

var slopeDeg = ee.Terrain.slope(elevation)
  .rename('slope_deg').toFloat().clip(rawDonorSearchGeom);

var precip = ee.Image('WORLDCLIM/V1/BIO')
  .select('bio12').rename('precip_mm_yr').toFloat().clip(rawDonorSearchGeom);

// SOC in t/ha. 'SOC' is the imported asset (define it in the GEE Imports).
// Mask the nodata/0 so they do not enter the pool as zeros.
var soc = SOC
  .updateMask(SOC.gt(0))
  .rename('SOC_g_kg').toFloat().clip(rawDonorSearchGeom);

var hwsd2 = ee.Image('projects/sat-io/open-datasets/FAO/HWSD_V2_SMU')
  .select('WRB2_CODE').rename('WRB2_CODE').toFloat().clip(rawDonorSearchGeom);

var roads = ee.FeatureCollection(ROAD_ASSET)
  .filterBounds(rawDonorSearchGeom.buffer(ROAD_DISTANCE_SEARCH_M, ee.ErrorMargin(100)));

var distRoads = ee.Image(0).byte()
  .paint(roads, 1)
  .rename('roads')
  .clip(rawDonorSearchGeom)
  .fastDistanceTransform(Math.ceil(ROAD_DISTANCE_SEARCH_M / SCALE_M))
  .sqrt()
  .multiply(SCALE_M)
  .min(ROAD_DISTANCE_SEARCH_M)
  .divide(1000)
  .rename('dist_roads_km').toFloat().clip(rawDonorSearchGeom);


// ────────────────────────────────────────────────────────────
// 6. ANNUAL NDVI STACK
// ────────────────────────────────────────────────────────────

var trendStart = T0_YEAR - TREND_YEARS;

var ndviT0 = annualNDVI(T0_YEAR).rename('NDVI_t0').toFloat();

var trendCollection = ee.ImageCollection(
  ee.List.sequence(trendStart, T0_YEAR - 1).map(function(yr) {
    yr = ee.Number(yr);
    return ee.Image.constant(1).rename('constant').toFloat()
      .addBands(ee.Image.constant(yr).rename('year').toFloat())
      .addBands(annualNDVI(yr).rename('NDVI').toFloat());
  })
);

var ndviSlope = trendCollection
  .reduce(ee.Reducer.linearRegression({ numX: 2, numY: 1 }))
  .select('coefficients')
  .arrayProject([0])
  .arrayFlatten([['intercept', 'slope']])
  .select('slope')
  .rename('NDVI_slope_5yr').toFloat();

var ndviAnnualStack = null;
var yearBandNames   = [];
for (var annYr = trendStart; annYr < T0_YEAR; annYr++) {
  var bName = 'NDVI_' + String(annYr);
  yearBandNames.push(bName);
  var yBand = annualNDVI(annYr).rename(bName).toFloat();
  ndviAnnualStack = ndviAnnualStack === null ? yBand : ndviAnnualStack.addBands(yBand);
}

var ndviAnnualValidCount = ndviAnnualStack
  .reduce(ee.Reducer.count()).rename('ndvi_valid_years').toFloat();

var ndviAnnualEnoughData = ndviAnnualValidCount
  .gte(MIN_VALID_YEARS).rename('ndvi_enough_data');

print('Annual NDVI bands:', yearBandNames);


// ────────────────────────────────────────────────────────────
// 7. COVARIATE STACK + VALIDITY MASKS
//
// validCovMask requires:
//   - valid NDVI (at least MIN_VALID_YEARS cloud-free years)
//   - valid elevation, slope, precip, SOC, WRB
//   - no nodata/null data
//
// Does not include forest/nonforest filter — handled in Python.
// ────────────────────────────────────────────────────────────

var pixelAreaHa = ee.Image.pixelArea().divide(10000)
  .rename('pixel_area_ha').toFloat();

var covStackCore = ndviT0
  .addBands(ndviSlope)
  .addBands(ndviAnnualStack)
  .addBands(elevation)
  .addBands(slopeDeg)
  .addBands(precip)
  .addBands(soc)
  .addBands(hwsd2)
  .addBands(distRoads)
  .addBands(ndviAnnualValidCount)
  .addBands(pixelAreaHa)
  .clip(rawDonorSearchGeom)
  .toFloat();

print('Covariate bands:', covStackCore.bandNames());

// WRB2_CODE valid as DONOR. HWSD2 v2.0 legend (table D_WRB2code).
// The non-soils are excluded (12 Glaciers, 16 Islands, 34 Open Water, 35 No Data,
// besides 0=nodata) and Technosols (31): they are a REAL but anthropic soil
// (not a natural analog) and lacking texture data in HWSD2 → not valid as
// donor (they would be discarded anyway in Step 02).
var hwsdCode     = hwsd2.unmask(0);
var hwsdSoilMask = hwsdCode.gt(0)
  .and(hwsdCode.neq(12))   // Glaciers (non-soil)
  .and(hwsdCode.neq(16))   // Islands (non-soil)
  .and(hwsdCode.neq(31))   // Technosols (anthropic, without HWSD2 texture)
  .and(hwsdCode.neq(34))   // Open Water (non-soil)
  .and(hwsdCode.neq(35));  // No Data

var validCovMask = ndviT0.mask()
  .and(ndviAnnualEnoughData)
  .and(elevation.mask())
  .and(slopeDeg.mask())
  .and(precip.mask())
  .and(soc.mask())
  .and(hwsdSoilMask)
  .rename('valid_covariate_mask');

// Final masks: valid covariates + no permanent water
// No forest filter — Python handles it.
var donorValidMask   = validCovMask.and(baseMaskDonor)
  .clip(donorExportGeom).rename('donor_valid_mask');
var projectValidMask = validCovMask.and(baseMaskProject)
  .clip(projectGeom).rename('project_valid_mask');


// ────────────────────────────────────────────────────────────
// 8. DIAGNOSTICS — PROJECT
// ────────────────────────────────────────────────────────────

print('');
print('=== PROJECT DIAGNOSTICS (@ ' + DIAGNOSTIC_SCALE_M + ' m) ===');
print('[1] Total PA area ha:',
  areaHa(ee.Image.constant(1).clip(projectGeom), projectGeom, DIAGNOSTIC_SCALE_M));
print('[2] ndviAnnualEnoughData inside PA ha:',
  areaHa(ndviAnnualEnoughData.clip(projectGeom), projectGeom, DIAGNOSTIC_SCALE_M));
print('[3] WRB valid inside PA ha:',
  areaHa(hwsdSoilMask.clip(projectGeom), projectGeom, DIAGNOSTIC_SCALE_M));
print('[4] permanentWater.not() inside PA ha:',
  areaHa(permanentWaterMask.not().clip(projectGeom), projectGeom, DIAGNOSTIC_SCALE_M));
print('[5] PROJECT valid mask (covariates + no water) ha:',
  areaHa(projectValidMask, projectGeom, DIAGNOSTIC_SCALE_M));


// ────────────────────────────────────────────────────────────
// 8b. DIAGNOSTICS — DONOR (added in v07)
// ────────────────────────────────────────────────────────────

var _pixHa = (SCALE_M * SCALE_M) / 10000.0;   // ha per pixel (30m -> 0.09 ha)

print('');
print('=== DONOR DIAGNOSTICS (@ ' + DIAGNOSTIC_SCALE_M + ' m) ===');
print('[D1] Donor search area (buffer∩eco \\ excl) ha:',
  areaHa(ee.Image.constant(1).clip(donorExportGeom), donorExportGeom, DIAGNOSTIC_SCALE_M));
print('[D2] WRB valid inside DONOR ha:',
  areaHa(hwsdSoilMask.clip(donorExportGeom), donorExportGeom, DIAGNOSTIC_SCALE_M));
print('[D3] ndviAnnualEnoughData inside DONOR ha:',
  areaHa(ndviAnnualEnoughData.clip(donorExportGeom), donorExportGeom, DIAGNOSTIC_SCALE_M));
print('[D4] permanentWater.not() inside DONOR ha:',
  areaHa(permanentWaterMask.not().clip(donorExportGeom), donorExportGeom, DIAGNOSTIC_SCALE_M));
print('[D5] DONOR valid mask ha (before the Python forest/eligible filter):',
  areaHa(donorValidMask, donorExportGeom, DIAGNOSTIC_SCALE_M));
print('[D6] DONOR valid pixel estimate (~area_ha / ' + _pixHa.toFixed(3) + '):',
  ee.Number(areaHa(donorValidMask, donorExportGeom, DIAGNOSTIC_SCALE_M)).divide(_pixHa).round());
print('NOTE: the forest/eligible filter is applied in Python (Step 01).');
print('      FNF18_fullBuffer.shp -> excludes forest areas at T0');
print('      Eligible_FNF_fullBuffer.shp -> keeps only eligible donor areas');


// ────────────────────────────────────────────────────────────
// 8c. COVARIATE STATISTICS — PA vs DONOR (mean/stdDev/min/max)
//     To verify the QUALITY of the data and where the imbalance originates
//     directly at the source (before download).
// ────────────────────────────────────────────────────────────

function covStats(band, mask, geom) {
  return covStackCore.select(band).updateMask(mask).reduceRegion({
    reducer: ee.Reducer.mean()
      .combine({ reducer2: ee.Reducer.stdDev(), sharedInputs: true })
      .combine({ reducer2: ee.Reducer.minMax(),  sharedInputs: true }),
    geometry: geom,
    scale: DIAGNOSTIC_SCALE_M,
    maxPixels: 1e13,
    bestEffort: true,
    tileScale: 4
  });
}

print('');
print('=== PA vs DONOR COVARIATE STATISTICS (@ ' + DIAGNOSTIC_SCALE_M + ' m) ===');
['NDVI_t0', 'NDVI_slope_5yr', 'elevation', 'slope_deg',
 'precip_mm_yr', 'SOC_g_kg', 'dist_roads_km'].forEach(function (b) {
  print('  PA  » ' + b, covStats(b, projectValidMask, projectGeom));
  print('  DON » ' + b, covStats(b, donorValidMask,   donorExportGeom));
});


// ────────────────────────────────────────────────────────────
// 9. FINAL RASTERS
// ────────────────────────────────────────────────────────────

var projectRaster = covStackCore
  .addBands(ee.Image.constant(1).rename('is_project').toFloat())
  .addBands(ee.Image.constant(0).rename('is_donor').toFloat())
  .updateMask(projectValidMask)
  .toFloat();

var donorRaster = covStackCore
  .addBands(ee.Image.constant(0).rename('is_project').toFloat())
  .addBands(ee.Image.constant(1).rename('is_donor').toFloat())
  .updateMask(donorValidMask)
  .toFloat();

print('Project raster bands:', projectRaster.bandNames());
print('Donor raster bands:', donorRaster.bandNames());


// ────────────────────────────────────────────────────────────
// 10. EXPORT BOUNDARY SHAPEFILES
// ────────────────────────────────────────────────────────────

var projectBoundaryFc = projectFc.map(function(f) {
  return ee.Feature(f.geometry(), {
    name: 'project', typ: 'PA', runid: RUN_ID, t0: T0_YEAR
  });
}).map(addAreaHa);

var donorPoolBoundaryFc = ee.FeatureCollection([
  ee.Feature(donorExportGeom, {
    name: 'donorpool', typ: 'DON_DOM', runid: RUN_ID, t0: T0_YEAR,
    buf_km: MAX_DISTANCE_FROM_PROJECT_KM, excl_km: CONTROL_EXCLUSION_KM,
    fnf_filter: 'applied_in_python_FNF18_fullBuffer',
    eligible_filter: 'applied_in_python_Eligible_FNF_fullBuffer'
  })
]).map(addAreaHa);

Export.table.toDrive({
  collection: projectBoundaryFc,
  description: 'STARR_project_area_boundary_' + RUN_ID,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'project_area_boundary_' + RUN_ID,
  fileFormat: 'SHP'
});

Export.table.toDrive({
  collection: donorPoolBoundaryFc,
  description: 'STARR_donor_pool_boundary_' + RUN_ID,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'donor_pool_boundary_' + RUN_ID,
  fileFormat: 'SHP'
});

Export.table.toDrive({
  collection: maskToPolygonFc(projectValidMask, projectGeom, 'PA_VALID', SCALE_M),
  description: 'STARR_project_valid_area_' + RUN_ID,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'project_valid_area_' + RUN_ID,
  fileFormat: 'SHP'
});

Export.table.toDrive({
  collection: maskToPolygonFc(donorValidMask, donorExportGeom, 'DON_VALID', SCALE_M),
  description: 'STARR_donor_valid_area_pre_python_filter_' + RUN_ID,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'donor_valid_area_pre_python_' + RUN_ID,
  fileFormat: 'SHP'
});


// ────────────────────────────────────────────────────────────
// 11. EXPORT COVARIATE RASTERS
// ────────────────────────────────────────────────────────────

Export.image.toDrive({
  image: projectRaster,
  description: 'STARR_project_covariates_raster_' + RUN_ID,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'covariates_project_' + RUN_ID,
  region: projectGeom,
  crs: EXPORT_CRS, scale: SCALE_M, maxPixels: 1e13,
  fileFormat: 'GeoTIFF', skipEmptyTiles: true,
  formatOptions: { cloudOptimized: true }
});

Export.image.toDrive({
  image: donorRaster,
  description: 'STARR_donor_covariates_raster_' + RUN_ID,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'covariates_donor_' + RUN_ID,
  region: donorExportGeom,
  crs: EXPORT_CRS, scale: SCALE_M, maxPixels: 1e13,
  fileFormat: 'GeoTIFF', skipEmptyTiles: true,
  formatOptions: { cloudOptimized: true }
});

print('');
print('=== Export tasks created ===');
print('Raster project  :', 'covariates_project_' + RUN_ID);
print('Raster donor    :', 'covariates_donor_' + RUN_ID,
      '(all pixels with valid covariates, without forest filter)');
print('');
print('After the download, Step 01 Python applies:');
print('  FNF18_fullBuffer.shp       -> excludes forest areas at T0 from donor and PA');
print('  Eligible_FNF_fullBuffer.shp -> keeps only eligible donor areas');


// ────────────────────────────────────────────────────────────
// 12. MAP PREVIEW
// ────────────────────────────────────────────────────────────

if (MAP_PREVIEW) {
  Map.centerObject(projectFc, 10);
  Map.addLayer(projectFc,               { color: 'red'    }, 'PA AOI', true);
  Map.addLayer(fcFromGeom(donorExportGeom), { color: '00ffff' }, 'Donor export geom', false);
  Map.addLayer(fcFromGeom(projectExclusionGeom), { color: 'ff9900' }, 'Excl. buffer 5km', false);
  Map.addLayer(permanentWaterMask.selfMask(), { palette: ['0000ff'] }, 'Permanent water', false);
  Map.addLayer(ndviAnnualValidCount.clip(rawDonorSearchGeom), {
    min: 0, max: TREND_YEARS, palette: ['red','orange','yellow','lime','green']
  }, 'NDVI valid years', true);
  Map.addLayer(validCovMask.selfMask(),    { palette: ['00aaff'] }, 'Valid covariates', false);
  Map.addLayer(projectValidMask.selfMask(), { palette: ['ff2200'] }, 'Project → export', true);
  Map.addLayer(donorValidMask.selfMask(),   { palette: ['00aa00'] }, 'Donor → export (pre Python filter)', false);
  Map.addLayer(ndviT0, { min: 0, max: 0.8, palette: ['brown','yellow','darkgreen'] }, 'NDVI T0', false);
  Map.addLayer(distRoads, { min: 0, max: 20, palette: ['red','yellow','blue'] }, 'Dist roads km', false);
}

// ============================================================
// END v07 — seasonal NDVI (A/C/annual) + only L8 + donor diagnostics
//           forest/eligible filters in Python (Step 01)
// ============================================================
