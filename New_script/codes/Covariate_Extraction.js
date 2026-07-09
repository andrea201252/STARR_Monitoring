// ============================================================
// GOLD STANDARD STARR – Track 1 SEMDB
// RASTER-FIRST PIPELINE — v07
//
// GEE export strategy: esporta TUTTI i pixel con covariate valide
// (NDVI, elevation, slope, SOC, WRB, roads) e non-acqua permanente.
// Il filtro forest/non-forest e le aree eleggibili vengono applicati
// in Python (Step 01) usando due shapefile locali su Drive.
//
// NOVITÀ v07:
//   - NDVI compositing per STAGIONE selezionabile (A growing / C dry / annual)
//     + statistica (median / p90 / max)  → NDVI_SEASON, NDVI_COMPOSITE_STAT
//   - Landsat: solo L8 su richiesta (USE_L9 = false)
//   - Maschera WRB riallineata alla legenda HWSD2 (esclude i veri non-suoli
//     12 Glaciers, 16 Islands, 34 Open Water, 35 No Data; 31=Technosols è suolo)
//   - SOC (t/ha): maschera nodata/0 così non entra nel pool come zero
//   - Diagnostica DONOR + statistiche covariate PA vs DONOR alla sorgente
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

// Buffer di ricerca donor attorno alla PA.
var MAX_DISTANCE_FROM_PROJECT_KM = 50;
// Zona di esclusione intorno alla PA (leakage buffer).
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

// ── NDVI compositing: STAGIONE + STATISTICA ──────────────────
// Emisfero sud (Mozambico). Scegli la finestra del composito NDVI annuale:
//   'growing' (A) = stagione di crescita: dic(anno-1) → apr(anno)  [picco verde]
//   'dry'     (C) = stagione secca:       mag → set (stesso anno)  [stabile, poche nuvole]
//   'annual'      = anno intero apr→apr (comportamento v06 originale)
// Cambia SOLO questa riga per testare A vs C senza toccare altro.
var NDVI_SEASON         = 'growing';   // 'growing' | 'dry' | 'annual'
// Statistica del composito stagionale:
//   'median' (robusta), 'p90' (vicino al picco), 'max' (greenest pixel)
var NDVI_COMPOSITE_STAT = 'median';    // 'median' | 'p90' | 'max'

// Landsat: solo L8 su richiesta (L9 disattivato).
var USE_L9 = false;                    // false = solo L8 | true = L8 + L9


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
    // buffer(0) su una GeometryCollection restituisce un MultiPolygon
    // tenendo solo le parti poligonali — è il modo più robusto in GEE
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
// 2. GEOMETRIE
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

// Zona di analisi Landsat (ampia — serve per raccogliere immagini)
var rawDonorSearchGeom = cleanGeom(
  safeIntersect(projectBufferGeom, analysisEcoGeom));

// Zona di export donor: buffer ∩ ecoregion \ esclusione 5 km
// Non applica filtri forest/eligible — ci pensa Python con gli shapefile.
var donorExportGeom = cleanGeom(
  rawDonorSearchGeom.difference(projectExclusionGeom, ee.ErrorMargin(100)));

print('RUN_ID:', RUN_ID, '| T0:', T0_YEAR, '| CRS:', EXPORT_CRS, '| Scale:', SCALE_M, 'm');
print('NDVI season:', NDVI_SEASON, '| stat:', NDVI_COMPOSITE_STAT, '| USE_L9:', USE_L9);
print('Donor export geometry type:', donorExportGeom.type());
print('Filtri forest/eligible: applicati in Python (Step 01) con shapefile Drive.');


// ────────────────────────────────────────────────────────────
// 3. LANDSAT — QA MASKING E NDVI (stagionale, selezionabile)
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

// Collezione Landsat. USE_L9=false → solo L8 (richiesta). true → L8 + L9.
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

// Finestra temporale del composito annuale per la stagione scelta.
// 'year' = anno di riferimento del composito.
function seasonWindow(year) {
  year = ee.Number(year);
  if (NDVI_SEASON === 'growing') {
    // A — stagione di crescita emisfero sud: dic(anno-1) → apr(anno) incluso
    return { start: ee.Date.fromYMD(year.subtract(1), 12, 1),
             end:   ee.Date.fromYMD(year, 5, 1) };
  } else if (NDVI_SEASON === 'dry') {
    // C — stagione secca: mag → set (stesso anno)
    return { start: ee.Date.fromYMD(year, 5, 1),
             end:   ee.Date.fromYMD(year, 10, 1) };
  }
  // 'annual' — anno intero apr→apr (v06 originale)
  return { start: ee.Date.fromYMD(year, 4, 1),
           end:   ee.Date.fromYMD(year, 4, 1).advance(1, 'year') };
}

// Statistica del composito → banda 'NDVI'.
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

print('--- Scene Landsat (' + (USE_L9 ? 'L8+L9' : 'solo L8')
      + ') per stagione [' + NDVI_SEASON + '] ---');
ee.List.sequence(T0_YEAR - TREND_YEARS, T0_YEAR).evaluate(function(years) {
  years.forEach(function(yr) {
    var w = seasonWindow(yr);
    print('Year ' + yr + ':', l89Collection(w.start, w.end, rawDonorSearchGeom).size());
  });
});


// ────────────────────────────────────────────────────────────
// 4. MASCHERE — SOLO ACQUA PERMANENTE
//
// In questa versione NON si applica alcun filtro forest/nonforest
// in GEE. Tutti i pixel con covariate valide vengono esportati.
// Il filtraggio forest/eligible viene applicato in Python (Step 01)
// usando gli shapefile su Drive:
//   FNF18_fullBuffer.shp       -> esclude aree forest a T0
//   Eligible_FNF_fullBuffer.shp -> mantiene solo aree eleggibili
// ────────────────────────────────────────────────────────────

var permanentWaterMask = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence')
  .unmask(0)
  .gte(WATER_OCCURRENCE_THRESHOLD)
  .clip(rawDonorSearchGeom)
  .rename('permanent_water');

// Masca solo l'acqua permanente; il forest filter è demandato a Python.
var baseMaskDonor   = permanentWaterMask.not().clip(donorExportGeom);
var baseMaskProject = permanentWaterMask.not().clip(projectGeom);


// ────────────────────────────────────────────────────────────
// 5. COVARIATE STATICHE
// ────────────────────────────────────────────────────────────

var elevation = ee.Image("USGS/SRTMGL1_003")
  .select('elevation').rename('elevation').toFloat().clip(rawDonorSearchGeom);

var slopeDeg = ee.Terrain.slope(elevation)
  .rename('slope_deg').toFloat().clip(rawDonorSearchGeom);

var precip = ee.Image('WORLDCLIM/V1/BIO')
  .select('bio12').rename('precip_mm_yr').toFloat().clip(rawDonorSearchGeom);

// SOC in t/ha. 'SOC' è l'asset importato (definiscilo negli Imports GEE).
// Maschera i nodata/0 così non entrano nel pool come zeri.
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
// 6. STACK NDVI ANNUALE
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
// 7. STACK COVARIATE + MASCHERE DI VALIDITÀ
//
// validCovMask richiede:
//   - NDVI valido (almeno MIN_VALID_YEARS anni cloud-free)
//   - elevation, slope, precip, SOC, WRB validi
//   - nessun dato nodata/nullo
//
// Non include filtro forest/nonforest — gestito in Python.
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

// WRB2_CODE valido come DONOR. Legenda HWSD2 v2.0 (tabella D_WRB2code).
// Esclusi i non-suoli (12 Glaciers, 16 Islands, 34 Open Water, 35 No Data,
// oltre a 0=nodata) e i Technosols (31): sono un suolo REALE ma antropico
// (non-analogo naturale) e privi di dato texture in HWSD2 → non validi come
// donor (verrebbero comunque scartati in Step 02).
var hwsdCode     = hwsd2.unmask(0);
var hwsdSoilMask = hwsdCode.gt(0)
  .and(hwsdCode.neq(12))   // Glaciers (non-suolo)
  .and(hwsdCode.neq(16))   // Islands (non-suolo)
  .and(hwsdCode.neq(31))   // Technosols (antropico, senza texture HWSD2)
  .and(hwsdCode.neq(34))   // Open Water (non-suolo)
  .and(hwsdCode.neq(35));  // No Data

var validCovMask = ndviT0.mask()
  .and(ndviAnnualEnoughData)
  .and(elevation.mask())
  .and(slopeDeg.mask())
  .and(precip.mask())
  .and(soc.mask())
  .and(hwsdSoilMask)
  .rename('valid_covariate_mask');

// Maschere finali: covariate valide + no acqua permanente
// Nessun filtro forest — Python ci pensa.
var donorValidMask   = validCovMask.and(baseMaskDonor)
  .clip(donorExportGeom).rename('donor_valid_mask');
var projectValidMask = validCovMask.and(baseMaskProject)
  .clip(projectGeom).rename('project_valid_mask');


// ────────────────────────────────────────────────────────────
// 8. DIAGNOSTICS — PROGETTO
// ────────────────────────────────────────────────────────────

print('');
print('=== DIAGNOSTICS PROGETTO (@ ' + DIAGNOSTIC_SCALE_M + ' m) ===');
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
// 8b. DIAGNOSTICS — DONOR (aggiunta v07)
// ────────────────────────────────────────────────────────────

var _pixHa = (SCALE_M * SCALE_M) / 10000.0;   // ha per pixel (30m -> 0.09 ha)

print('');
print('=== DIAGNOSTICS DONOR (@ ' + DIAGNOSTIC_SCALE_M + ' m) ===');
print('[D1] Donor search area (buffer∩eco \\ excl) ha:',
  areaHa(ee.Image.constant(1).clip(donorExportGeom), donorExportGeom, DIAGNOSTIC_SCALE_M));
print('[D2] WRB valid inside DONOR ha:',
  areaHa(hwsdSoilMask.clip(donorExportGeom), donorExportGeom, DIAGNOSTIC_SCALE_M));
print('[D3] ndviAnnualEnoughData inside DONOR ha:',
  areaHa(ndviAnnualEnoughData.clip(donorExportGeom), donorExportGeom, DIAGNOSTIC_SCALE_M));
print('[D4] permanentWater.not() inside DONOR ha:',
  areaHa(permanentWaterMask.not().clip(donorExportGeom), donorExportGeom, DIAGNOSTIC_SCALE_M));
print('[D5] DONOR valid mask ha (prima del filtro Python forest/eligible):',
  areaHa(donorValidMask, donorExportGeom, DIAGNOSTIC_SCALE_M));
print('[D6] DONOR valid stima pixel (~area_ha / ' + _pixHa.toFixed(3) + '):',
  ee.Number(areaHa(donorValidMask, donorExportGeom, DIAGNOSTIC_SCALE_M)).divide(_pixHa).round());
print('NOTA: il filtro forest/eligible viene applicato in Python (Step 01).');
print('      FNF18_fullBuffer.shp -> esclude aree forest a T0');
print('      Eligible_FNF_fullBuffer.shp -> mantiene solo aree eleggibili donor');


// ────────────────────────────────────────────────────────────
// 8c. STATISTICHE COVARIATE — PA vs DONOR (mean/stdDev/min/max)
//     Per verificare la QUALITÀ del dato e dove nasce lo squilibrio
//     direttamente alla sorgente (prima del download).
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
print('=== STATISTICHE COVARIATE PA vs DONOR (@ ' + DIAGNOSTIC_SCALE_M + ' m) ===');
['NDVI_t0', 'NDVI_slope_5yr', 'elevation', 'slope_deg',
 'precip_mm_yr', 'SOC_g_kg', 'dist_roads_km'].forEach(function (b) {
  print('  PA  » ' + b, covStats(b, projectValidMask, projectGeom));
  print('  DON » ' + b, covStats(b, donorValidMask,   donorExportGeom));
});


// ────────────────────────────────────────────────────────────
// 9. RASTER FINALI
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
// 10. EXPORT SHAPEFILE CONFINI
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
// 11. EXPORT RASTER COVARIATE
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
print('=== Export tasks creati ===');
print('Raster project  :', 'covariates_project_' + RUN_ID);
print('Raster donor    :', 'covariates_donor_' + RUN_ID,
      '(tutti i pixel con covariate valide, senza filtro forest)');
print('');
print('Dopo il download, Step 01 Python applica:');
print('  FNF18_fullBuffer.shp       -> esclude aree forest a T0 da donor e PA');
print('  Eligible_FNF_fullBuffer.shp -> mantiene solo aree eleggibili donor');


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
// END v07 — NDVI stagionale (A/C/annual) + solo L8 + diagnostica donor
//           filtri forest/eligible in Python (Step 01)
// ============================================================
