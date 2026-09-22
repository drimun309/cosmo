"""Клипует Sentinel-1 RTC и Sentinel-2 L2A на сетку эталонной маски.

Google Earth Engine с этой машины недоступен. Берём Planetary Computer:
S1 RTC (gamma0, уже террейн-коррекция) переводим в дБ; S2 L2A — в отражение 0–1
и индексы. ponytail: gamma0, не sigma0 GEE; на крутых склонах разница есть,
в пойме Амура для базовой линии достаточно. Облака гасим по SCL.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import tifffile
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

import hw

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")
os.environ.setdefault("VSI_CACHE", "TRUE")

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
S2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
CLOUD_SCL = {0, 1, 3, 8, 9, 10, 11}


def _get(url, payload=None, retries=4):
    import time
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"User-Agent": "hydrowatch"}
    if data:
        headers["Content-Type"] = "application/json"
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait = 10 * (i + 1)
                print(f"  HTTP {e.code}, жду {wait}s", flush=True)
                time.sleep(wait)
                last = e
                continue
            raise
    raise last


def search(collection, bbox, day, query=None):
    body = {
        "collections": [collection],
        "bbox": [round(v, 6) for v in bbox],
        "datetime": f"{day}T00:00:00Z/{day}T23:59:59Z",
        "limit": 30,
    }
    if query:
        body["query"] = query
    feats = _get(STAC, body).get("features", [])
    return sorted(feats, key=lambda f: f["id"])


def sign(href):
    return _get(SIGN + urllib.parse.quote(href, safe=""))["href"]


def grid(template: Path):
    ox, oy, scale = hw.geo(template)
    with tifffile.TiffFile(template) as tif:
        page = tif.pages[0]
        h, w = page.imagelength, page.imagewidth
    return from_origin(ox, oy, scale, scale), h, w


def bbox_wgs84(transform, h, w):
    west, south, east, north = rasterio.transform.array_bounds(h, w, transform)
    xs, ys = Transformer.from_crs(32652, 4326, always_xy=True).transform(
        [west, east, west, east], [north, north, south, south]
    )
    return [min(xs), min(ys), max(xs), max(ys)]


def read_onto(url, transform, h, w, resampling):
    with rasterio.open(url) as src:
        west, south, east, north = rasterio.transform.array_bounds(h, w, transform)
        left, bottom, right, top = transform_bounds(
            "EPSG:32652", src.crs, west, south, east, north, densify_pts=21
        )
        window = from_bounds(left, bottom, right, top, src.transform)
        window = Window(window.col_off - 2, window.row_off - 2, window.width + 4, window.height + 4)
        window = window.round_offsets(op="floor").round_lengths(op="ceil")
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width < 1 or window.height < 1:
            return None
        data = src.read(1, window=window).astype(np.float32)
        scale = src.scales[0] if src.scales else 1.0
        offset = src.offsets[0] if src.offsets else 0.0
        if scale != 1 or offset != 0:
            data = data * np.float32(scale) + np.float32(offset)
        nodata = src.nodata
        if nodata is not None:
            data[data == np.float32(nodata)] = np.nan
        data[~np.isfinite(data)] = -9999
        dest = np.full((h, w), np.nan, np.float32)
        reproject(
            data,
            dest,
            src_transform=src.window_transform(window),
            src_crs=src.crs,
            src_nodata=-9999,
            dst_transform=transform,
            dst_crs="EPSG:32652",
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return dest


def mosaic(items, asset, transform, h, w, resampling):
    acc = np.full((h, w), np.nan, np.float32)
    used = []
    for feat in items:
        if asset not in feat["assets"]:
            continue
        part = read_onto(sign(feat["assets"][asset]["href"]), transform, h, w, resampling)
        if part is None:
            continue
        fill = ~np.isfinite(acc) & np.isfinite(part)
        acc[fill] = part[fill]
        used.append(feat["id"])
    return acc, used


def to_db(lin):
    out = np.full(lin.shape, np.nan, np.float32)
    ok = np.isfinite(lin) & (lin > 0)
    out[ok] = (10.0 * np.log10(lin[ok])).astype(np.float32)
    return out


def norm_diff(a, b):
    out = np.full(a.shape, np.nan, np.float32)
    den = a + b
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(den) > 1e-6)
    out[ok] = ((a[ok] - b[ok]) / den[ok]).astype(np.float32)
    return out


def write_stack(path: Path, stack, template: Path, bands, scene_date, scene_meta=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        np.moveaxis(stack, 0, -1).astype(np.float32),
        photometric="minisblack",
        compression="zlib",
        compressionargs={"level": 1},
        extratags=hw.geo_tags(template),
    )
    meta = {
        "bands": list(bands),
        "shape": list(stack.shape[1:]),
        "scene_date": scene_date,
        "crs": "EPSG:32652",
        "source": "Microsoft Planetary Computer",
    }
    if scene_meta:
        meta["scenes"] = scene_meta
    path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def valid_frac(arr):
    return float(np.isfinite(arr).mean())


def fetch_s1(folder, prefix, day, orbit, bbox, transform, h, w, template):
    dest = folder / f"{prefix}{day}.tif"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  есть {dest.name}")
        return True
    items = search(
        "sentinel-1-rtc",
        bbox,
        day,
        {"sat:orbit_state": {"eq": "descending"}, "sat:relative_orbit": {"eq": int(float(orbit))}},
    )
    if not items:
        print(f"  нет S1 {day} orbit {orbit}", flush=True)
        return False
    print(f"  качаю S1 {day} ({len(items)})", flush=True)
    vv, ids = mosaic(items, "vv", transform, h, w, Resampling.bilinear)
    vh, _ = mosaic(items, "vh", transform, h, w, Resampling.bilinear)
    vv_db, vh_db = to_db(vv), to_db(vh)
    ratio = vv_db - vh_db
    write_stack(dest, np.stack([vv_db, vh_db, ratio]), template,
                bands=("VV", "VH", "VV_VH_ratio"), scene_date=day, scene_meta=ids)
    frac = valid_frac(vv_db)
    print(f"  {dest.name}  сцены {ids}  VV конечны {frac:.2f}", flush=True)
    return frac > 0.3


def fetch_s2(folder, prefix, day, bbox, transform, h, w, template):
    dest = folder / f"{prefix}{day}.tif"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  есть {dest.name}")
        return True
    items = search("sentinel-2-l2a", bbox, day)
    if not items:
        print(f"  нет S2 {day}", flush=True)
        return False
    print(f"  качаю S2 {day} ({len(items)})", flush=True)
    bands = {}
    ids = []
    for name in S2_BANDS:
        arr, ids = mosaic(items, name, transform, h, w, Resampling.bilinear)
        bands[name] = arr
    scl, _ = mosaic(items, "SCL", transform, h, w, Resampling.nearest)
    bad = np.isin(np.nan_to_num(scl, nan=-1), list(CLOUD_SCL))
    for arr in bands.values():
        arr[bad] = np.nan
    b2, b3, b4, b8, b11, b12 = (bands[n] for n in S2_BANDS)
    ndwi = norm_diff(b3, b8)
    mndwi = norm_diff(b3, b11)
    ndvi = norm_diff(b8, b4)
    awe = b2 + 2.5 * b3 - 1.5 * (b8 + b11) - 0.25 * b12
    awe[~np.isfinite(b2) | ~np.isfinite(b12)] = np.nan
    stack = np.stack([b3, b4, b8, b11, ndwi, mndwi, ndvi, awe])
    write_stack(dest, stack, template,
                bands=("B3", "B4", "B8", "B11", "NDWI", "MNDWI", "NDVI", "AWEIsh"),
                scene_date=day, scene_meta=ids)
    print(f"  {dest.name}  сцены {ids}  оптика жива {valid_frac(ndwi):.2f}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=hw.DEFAULT_DATA)
    p.add_argument("--only", help="один pair_id")
    args = p.parse_args()
    fails = []
    for row in hw.load_pairs(args.data):
        if args.only and row["pair_id"] != args.only:
            continue
        folder = args.data / row["rasters_dir"]
        template = args.data / row["reference_mask"]
        transform, h, w = grid(template)
        bbox = bbox_wgs84(transform, h, w)
        orbit = float(row["relative_orbit"])
        print(row["pair_id"], f"{h}x{w}", "orbit", int(orbit), flush=True)
        if not fetch_s1(folder, "S1_pre_", row["date_pre_sar"], orbit, bbox, transform, h, w, template):
            fails.append((row["pair_id"], "S1_pre"))
        if not fetch_s1(folder, "S1_peak_", row["date_peak_sar"], orbit, bbox, transform, h, w, template):
            fails.append((row["pair_id"], "S1_peak"))
        if row["date_pre_opt"]:
            fetch_s2(folder, "SENTINEL2_pre_", row["date_pre_opt"], bbox, transform, h, w, template)
        if row["date_peak_opt"]:
            fetch_s2(folder, "SENTINEL2_peak_", row["date_peak_opt"], bbox, transform, h, w, template)
    if fails:
        print("\nПары с пустыми S1:", flush=True)
        for pair, what in fails:
            print(f"  {pair}: {what} (нет валидных пикселей, AOI вне сцены)")


if __name__ == "__main__":
    main()
