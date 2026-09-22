"""REST API + лёгкая карта для мониторинга затопления.

Эндпоинты:
  GET  /health
  GET  /pairs                         — список доступных пар
  GET  /areas/{pair_id}               — площади (water_pre, water_peak, flood)
  GET  /contours/{pair_id}            — GeoJSON контуры воды/flood
  POST /report                        — отчёт по bbox или полигону
  GET  /layer/{pair_id}/{layer}       — маски слоя как PNG для карты

Запуск:
  python service.py
  # или
  uvicorn service:app --host 0.0.0.0 --port 8000
"""

import io
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import rasterio.transform
import tifffile
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from rasterio.features import shapes as rio_shapes
from shapely.geometry import Polygon, mapping, shape

import hw

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = hw.DEFAULT_DATA

app = FastAPI(title="HydroWatch Amur — Service", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"])


# ----------------------------- helpers -----------------------------

def _pair_data_dir():
    """Пары → путь к папке с масками и reference."""
    pairs = hw.load_pairs(DEFAULT_DATA)
    return pairs


def _mask_path(pair_id: str, layer: str) -> Optional[Path]:
    """Путь к маске (water_pre, water_peak, flood) для пары."""
    pdir = ROOT / "out" / "predictions"
    if layer == "flood":
        path = pdir / f"{pair_id}_flood.tif"
    else:
        path = pdir / f"{pair_id}_{layer}.tif"
    if not path.exists():
        return None
    return path


def _read_mask(path: Path) -> tuple[np.ndarray, tuple, float, tuple]:
    """Читает маску, возвращает (маска uint8, transform, scale, (w, h))."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = (arr > 0).astype(np.uint8)
    ox, oy, scale = hw.geo(path)
    from rasterio.transform import from_origin
    h, w = arr.shape
    transform = from_origin(ox, oy, scale, scale)
    return mask, transform, scale, (w, h)


def _mask_to_geojson(mask: np.ndarray, transform, scale: float, layer: str,
                    min_pixels: int = 5) -> list[dict]:
    """Превращает бинарную маску в список полигонов (GeoJSON в WGS84) с площадью."""
    import rasterio.warp as rw
    feats = []
    for geom, val in rio_shapes(mask, mask=mask > 0, transform=transform):
        if val == 0:
            continue
        poly = shape(geom)
        if poly.geom_type != "Polygon":
            continue
        area_m2 = poly.area
        if area_m2 / (scale * scale) < min_pixels:
            continue
        area_ha = round(area_m2 / 10_000.0, 2)
        # конвертируем в WGS84 для отображения на Leaflet
        poly_4326 = rw.transform_geom(
            "EPSG:32652", "EPSG:4326", mapping(poly)
        )
        feats.append({
            "type": "Feature",
            "geometry": poly_4326,
            "properties": {
                "layer": layer,
                "area_ha": area_ha,
                "area_km2": round(area_ha / 100.0, 4),
            },
        })
    return feats


def _mask_bbox_wgs84(mask: np.ndarray, transform) -> list[float]:
    """Bbox маски в WGS84: [west, south, east, north]."""
    h, w = mask.shape
    west, south, east, north = rasterio.transform.array_bounds(h, w, transform)
    w_w, s_w, e_w, n_w = rasterio.warp.transform_bounds(
        "EPSG:32652", "EPSG:4326", west, south, east, north
    )
    return [w_w, s_w, e_w, n_w]


def _bbox_intersects_mask(mask: np.ndarray, transform, bbox_4326: list[float]) -> float:
    """Доля пикселей маски внутри bbox (в WGS84)."""
    h, w = mask.shape
    west, south, east, north = rasterio.transform.array_bounds(h, w, transform)
    # Конвертируем bbox раstra в WGS84 (transform_bounds принимает (west, south, east, north))
    px_min, py_min, px_max, py_max = rasterio.warp.transform_bounds(
        "EPSG:32652", "EPSG:4326", west, south, east, north
    )
    # bbox_4326: [minx, miny, maxx, maxy]
    bx0, by0, bx1, by1 = bbox_4326
    ix0 = max(px_min, bx0)
    iy0 = max(py_min, by0)
    ix1 = min(px_max, bx1)
    iy1 = min(py_max, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0) / ((px_max - px_min) * (py_max - py_min))


def _reference_areas_for_pair(row: dict) -> dict:
    """Эталонные площади для пары."""
    template = DEFAULT_DATA / row["reference_mask"]
    arr = hw._bands(hw.read_tif(template), 5)
    _, _, scale = hw.geo(template)
    h, w = arr.shape[1:]
    counts = {name: int((arr[i] > 0).sum()) for i, name in enumerate(hw.REF_BANDS)}
    return {name: hw.hectares(c, scale) for name, c in counts.items()}


def _pair_summary(row: dict) -> dict:
    """Словарь площадей по паре: predicted (маски) + reference (эталон)."""
    pred_dir = ROOT / "out" / "predictions"
    pair_id = row["pair_id"]
    out = {"pair_id": pair_id, "aoi_name": row["aoi_name"], "event": row["event_name"],
           "date_pre": row["date_pre_sar"], "date_peak": row["date_peak_sar"],
           "event_kind": row["event_kind"]}
    # predicted
    pred = {}
    for layer in ("water_pre", "water_peak", "flood"):
        path = pred_dir / f"{pair_id}_{layer}.tif"
        if path.exists():
            mask = tifffile.imread(str(path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            _, _, scale = hw.geo(path)
            pred[layer] = round(int((mask > 0).sum()) * scale * scale / 10_000.0, 2)
        else:
            pred[layer] = None
    # reference
    ref = _reference_areas_for_pair(row)
    out["predicted_ha"] = pred
    out["reference_ha"] = {k: round(v, 2) for k, v in ref.items()}
    return out


# ----------------------------- endpoints -----------------------------

@app.get("/health")
def health():
    return {"status": "ok", "data_dir": str(DEFAULT_DATA), "out_dir": str(ROOT / "out")}


@app.get("/pairs")
def list_pairs():
    rows = hw.load_pairs(DEFAULT_DATA)
    return [
        {
            "pair_id": r["pair_id"],
            "aoi_name": r["aoi_name"],
            "event": r["event_name"],
            "event_kind": r["event_kind"],
            "date_pre": r["date_pre_sar"],
            "date_peak": r["date_peak_sar"],
            "has_optical_pre": bool(r["date_pre_opt"]),
            "has_optical_peak": bool(r["date_peak_opt"]),
        }
        for r in rows
    ]


@app.get("/areas/{pair_id}")
def get_areas(pair_id: str):
    rows = hw.load_pairs(DEFAULT_DATA)
    row = next((r for r in rows if r["pair_id"] == pair_id), None)
    if row is None:
        raise HTTPException(404, f"pair_id {pair_id} не найден")
    return _pair_summary(row)


@app.get("/contours/{pair_id}")
def get_contours(pair_id: str,
                layer: str = Query("flood", pattern="^(flood|water_pre|water_peak)$"),
                min_pixels: int = 5):
    path = _mask_path(pair_id, layer)
    if path is None:
        raise HTTPException(404, f"маска {layer} для {pair_id} не найдена")
    mask, transform, scale, (w, h) = _read_mask(path)
    feats = _mask_to_geojson(mask, transform, scale, layer, min_pixels=min_pixels)
    bbox_wgs84 = _mask_bbox_wgs84(mask, transform)
    return {
        "type": "FeatureCollection",
        "pair_id": pair_id,
        "layer": layer,
        "crs": "EPSG:4326",
        "bbox_wgs84": bbox_wgs84,
        "features": feats,
    }


class ReportRequest(BaseModel):
    pair_id: Optional[str] = None
    bbox: Optional[list[float]] = Field(None, description="[minx, miny, maxx, maxy] в WGS84")
    polygon: Optional[dict] = Field(None, description="GeoJSON Polygon")
    min_pixels: int = 5


@app.post("/report")
def make_report(req: ReportRequest):
    """Отчёт по паре или произвольному bbox/полигону в WGS84."""
    rows = hw.load_pairs(DEFAULT_DATA)
    if req.pair_id is None:
        raise HTTPException(400, "pair_id обязателен (пока без bbox)")

    row = next((r for r in rows if r["pair_id"] == req.pair_id), None)
    if row is None:
        raise HTTPException(404, f"pair_id {req.pair_id} не найден")

    pred_dir = ROOT / "out" / "predictions"
    summary = _pair_summary(row)

    # разбивка по ROI
    contours = {}
    if req.bbox:
        bbox_4326 = req.bbox
        for layer in ("flood", "water_pre", "water_peak"):
            path = pred_dir / f"{req.pair_id}_{layer}.tif"
            if not path.exists():
                continue
            mask, transform, scale, (w, h) = _read_mask(path)
            cover = _bbox_intersects_mask(mask, transform, bbox_4326)
            contours[layer] = {"intersection_frac": round(cover, 4)}
    elif req.polygon:
        poly = shape(req.polygon)
        poly_area_m2 = poly.area
        for layer in ("flood", "water_pre", "water_peak"):
            path = pred_dir / f"{req.pair_id}_{layer}.tif"
            if not path.exists():
                continue
            mask, transform, scale, (w, h) = _read_mask(path)
            contours[layer] = {"polygon_area_m2": round(poly_area_m2, 4)}

    # прирост воды
    pre = summary["predicted_ha"].get("water_pre") or 0
    peak = summary["predicted_ha"].get("water_peak") or 0
    flood = summary["predicted_ha"].get("flood") or 0
    ref_pre = summary["reference_ha"].get("water_pre") or 0
    ref_peak = summary["reference_ha"].get("water_peak") or 0
    ref_flood = summary["reference_ha"].get("flood") or 0

    # описание на человеческом
    descr = []
    if peak and pre:
        gain = peak - pre
        sign_word = "выросло" if gain > 0 else ("сократилось" if gain < 0 else "не изменилось")
        descr.append(f"водное зеркало {sign_word} на {abs(gain):.1f} га ({abs(gain)/100:.3f} км²) с pre до peak")
    if flood:
        descr.append(f"новое затопление {flood:.1f} га ({flood/100:.3f} км²) "
                     f"от {flood/float(row['aoi_km2'])*100:.3f}% AOI")
    if ref_flood and not flood:
        descr.append(f"эталон показывает {ref_flood:.1f} га затопления, "
                     f"но предсказание дало 0 — вероятно, нет валидного S1 для этой пары")

    return {
        "pair_id": req.pair_id,
        "description": "; ".join(descr) if descr else "нет данных",
        "aoi": {"name": row["aoi_name"], "km2": float(row["aoi_km2"])},
        "dates": {"pre": row["date_pre_sar"], "peak": row["date_peak_sar"]},
        "event": {"name": row["event_name"], "kind": row["event_kind"]},
        "predicted_ha": summary["predicted_ha"],
        "reference_ha": {k: round(v, 2) for k, v in summary["reference_ha"].items()},
        "gain_ha": round(peak - pre, 2),
        "gain_km2": round((peak - pre) / 100.0, 4),
        "flood_share_of_aoi_pct": round(flood / float(row["aoi_km2"]) * 100, 4) if row["aoi_km2"] else 0,
        "by_layer_in_roi": contours,
        "masks_available": {
            "flood": (pred_dir / f"{req.pair_id}_flood.tif").exists(),
            "water_pre": (pred_dir / f"{req.pair_id}_water_pre.tif").exists(),
            "water_peak": (pred_dir / f"{req.pair_id}_water_peak.tif").exists(),
        },
    }


@app.get("/layer/{pair_id}/{layer}.png")
def get_layer_png(pair_id: str, layer: str):
    """Возвращает бинарную маску как PNG для оверлея на карте."""
    path = _mask_path(pair_id, layer)
    if path is None:
        raise HTTPException(404, f"маска {layer} для {pair_id} не найдена")
    mask, transform, scale, (w, h) = _read_mask(path)
    bbox_wgs84 = _mask_bbox_wgs84(mask, transform)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = (mask * 200)
    rgba[..., 0] = mask * 255
    rgba[..., 1] = 0
    rgba[..., 2] = 0
    from PIL import Image
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"X-Mask-Bbox": ",".join(f"{x:.6f}" for x in bbox_wgs84)})


# ----------------------------- map page -----------------------------

_MAP_HTML = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>HydroWatch Amur — карта</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body { margin: 0; font-family: sans-serif; }
  #header { padding: 8px 12px; background: #1a1f2c; color: #fff; display: flex; align-items: center; gap: 12px; }
  #header label { font-size: 13px; }
  #header select { padding: 4px 8px; }
  #map { height: calc(100vh - 50px); }
  .legend { background: #fff; padding: 8px 12px; font-size: 13px; line-height: 1.6; }
  .legend .sw { display: inline-block; width: 16px; height: 12px; margin-right: 6px; vertical-align: middle; }
</style>
</head>
<body>
<div id="header">
  <label>Пара: <select id="pair"></select></label>
  <label>Слой: <select id="layer">
    <option value="flood">flood</option>
    <option value="water_pre">water_pre</option>
    <option value="water_peak">water_peak</option>
  </select></label>
  <span id="status" style="margin-left: auto; opacity: 0.7;"></span>
</div>
<div id="map"></div>
<script>
  const map = L.map('map').setView([50.5, 127.5], 6);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18, attribution: '© OSM contributors'
  }).addTo(map);

  const legend = L.control({position: 'topright'});
  legend.onAdd = () => L.DomUtil.create('div', 'legend');
  legend.addTo(map);

  let pngLayer = null;
  let contourLayer = null;

  function setStatus(text) { document.getElementById('status').textContent = text; }
  function setLegend(text) { document.querySelector('.legend').innerHTML = text; }

  async function loadPairs() {
    const r = await fetch('/pairs');
    const data = await r.json();
    const sel = document.getElementById('pair');
    data.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.pair_id;
      opt.textContent = p.pair_id + ' (' + p.event_kind + ')';
      sel.appendChild(opt);
    });
    update();
  }

  async function update() {
    const pair = document.getElementById('pair').value;
    const layer = document.getElementById('layer').value;
    if (pngLayer) { map.removeLayer(pngLayer); pngLayer = null; }
    if (contourLayer) { map.removeLayer(contourLayer); contourLayer = null; }
    setStatus('загрузка...');

    try {
      const cResp = await fetch('/contours/' + pair + '?layer=' + layer);
      if (!cResp.ok) { setStatus('нет маски'); setLegend('нет маски'); return; }
      const geo = await cResp.json();
      const bbox = geo.bbox_wgs84;
      const sw = [bbox[1], bbox[0]];
      const ne = [bbox[3], bbox[2]];

      // PNG-оверлей (полупрозрачный)
      pngLayer = L.imageOverlay('/layer/' + pair + '/' + layer + '.png',
                                [sw, ne], {opacity: 0.6, crossOrigin: true});
      pngLayer.addTo(map);

      // Контурный слой (поверх PNG)
      contourLayer = L.geoJSON(geo, {
        style: { color: '#cc0033', weight: 1, fillOpacity: 0 }
      }).addTo(map);

      map.fitBounds([sw, ne]);
      setStatus(geo.features.length + ' полигонов');
      setLegend(
        '<b>' + pair + ' / ' + layer + '</b><br>' +
        '<span><i class="sw" style="background:#c00;opacity:0.6"></i>вода</span>' +
        '<span><i class="sw" style="background:#c03;border:1px solid #c03"></i>контур</span>'
      );
    } catch(e) {
      setStatus('ошибка: ' + e.message);
      setLegend('ошибка загрузки');
    }
  }

  document.getElementById('pair').addEventListener('change', update);
  document.getElementById('layer').addEventListener('change', update);
  loadPairs();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def map_page():
    return _MAP_HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
