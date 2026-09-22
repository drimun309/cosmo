"""Сборка AUX_terrain_gsw.tif для каждой пары: 6 каналов в сетке эталона.

Источники через Planetary Computer STAC:
  - cop-dem-glo-30: Copernicus DEM GLO-30 -> slope (градусы) и HAND (м)
  - jrc-gsw:        JRC GSW v1.4 -> occurrence, seasonality, max_extent
  - esa-worldcover: ESA WorldCover v200 -> builtup (класс 50 = built-up)

HAND считаем сами из DEM: max(0, h_max - h) для каждого пикселя,
где h_max = максимальная высота по окрестности, а высота стока берётся
через гидравлическую привязку. Это грубый, но воспроизводимый HAND
без отдельных зависимостей.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import tifffile
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

import hw

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
HEAD = {"User-Agent": "hydrowatch"}

DEM_BANDS = ("data",)
GSW_BANDS = ("occurrence", "seasonality", "max_extent")
WC_BANDS = ("map",)

BUILTUP_CLASS = 50  # ESA WorldCover: built-up


def get(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {**HEAD}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def search_first(collection, bbox):
    feats = get(
        STAC,
        {
            "collections": [collection],
            "bbox": [round(v, 6) for v in bbox],
            "limit": 1,
        },
    ).get("features", [])
    return feats[0] if feats else None


def sign(href):
    return get(SIGN + urllib.parse.quote(href, safe=""))["href"]


def grid(template: Path):
    ox, oy, scale = hw.geo(template)
    with tifffile.TiffFile(template) as tif:
        h, w = tif.pages[0].imagelength, tif.pages[0].imagewidth
    return from_origin(ox, oy, scale, scale), h, w, scale


def bbox_wgs84(transform, h, w):
    west, south, east, north = (
        transform.c,
        transform.f - h * transform.e,
        transform.c + w * transform.a,
        transform.f,
    )
    xs, ys = Transformer.from_crs(32652, 4326, always_xy=True).transform(
        [west, east, west, east], [north, north, south, south]
    )
    return [min(xs), min(ys), max(xs), max(ys)]


def read_band(url, transform, h, w):
    """Читает первый канал удалённого TIFF на сетку эталона (nearest для масок, bilinear для DEM)."""
    import rasterio

    with rasterio.open(url) as src:
        dest = np.full((h, w), np.nan, np.float32)
        reproject(
            src.read(1).astype(np.float32) if False else None,
            dest,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata if src.nodata is not None else -9999.0,
            dst_transform=transform,
            dst_crs="EPSG:32652",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return dest


def read_band_window(url, transform, h, w, resampling=Resampling.bilinear):
    """Читает канал через окно в источнике (быстрее на больших DEM)."""
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds

    with rasterio.open(url) as src:
        west, south, east, north = array_bounds(h, w, transform)
        if src.crs.to_string() != "EPSG:32652":
            left, bottom, right, top = transform_bounds(
                "EPSG:32652", src.crs, west, south, east, north, densify_pts=21
            )
        else:
            left, bottom, right, top = west, south, east, north
        from rasterio.windows import from_bounds, Window

        window = from_bounds(left, bottom, right, top, src.transform)
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width < 1 or window.height < 1:
            return None
        data = src.read(1, window=window).astype(np.float32)
        scale = src.scales[0] if src.scales else 1.0
        offset = src.offsets[0] if src.offsets else 0.0
        if scale != 1.0 or offset != 0.0:
            data = data * np.float32(scale) + np.float32(offset)
        if src.nodata is not None:
            data[data == np.float32(src.nodata)] = np.nan
        data[~np.isfinite(data)] = np.nan
        dest = np.full((h, w), np.nan, np.float32)
        reproject(
            data,
            dest,
            src_transform=src.window_transform(window),
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs="EPSG:32652",
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return dest


def slope_degrees(dem):
    """Градиент DEM в градусы."""
    sy, sx = np.gradient(dem, 10.0, 10.0)
    slope_rad = np.arctan(np.sqrt(sx * sx + sy * sy))
    return np.degrees(slope_rad).astype(np.float32)


def hand_proxy(dem):
    """Грубый HAND: превышение над локальным минимумом в большом окне.

    На равнинах Амура это верхняя оценка. На горных участках будет
    завышать, поэтому порог HAND потом подбираем отдельно по району.
    """
    from scipy.ndimage import minimum_filter

    valid = np.isfinite(dem)
    fill = float(np.nanmedian(dem)) if valid.any() else 0.0
    dem_filled = np.where(valid, dem, fill)
    # окно 3 км = 300 пикселей при 10 м
    base = minimum_filter(dem_filled, size=300)
    hand = dem_filled - base
    hand = np.where(valid, hand, np.nan).astype(np.float32)
    hand[hand < 0] = 0
    return hand


def fetch_one(data: Path, row: dict) -> bool:
    folder = data / row["rasters_dir"]
    dest = folder / "AUX_terrain_gsw.tif"
    json_dest = folder / "AUX_terrain_gsw.json"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  есть {dest.name}")
        return True

    template = data / row["reference_mask"]
    transform, h, w, scale = grid(template)
    bbox = bbox_wgs84(transform, h, w)
    print(f"{row['pair_id']}: {h}x{w}, bbox={[round(b,3) for b in bbox]}")

    # 1. DEM GLO-30 -> slope, HAND
    print("  ищу cop-dem-glo-30...", flush=True)
    feat = search_first("cop-dem-glo-30", bbox)
    if not feat:
        print("  DEM не найден", flush=True)
        return False
    dem_url = sign(feat["assets"]["data"]["href"])
    dem = read_band_window(dem_url, transform, h, w, Resampling.bilinear)
    if dem is None:
        print("  DEM не выгрузился", flush=True)
        return False
    slope = slope_degrees(dem)
    hand = hand_proxy(dem)
    print(f"  DEM записан ({feat['id']})", flush=True)

    # 2. JRC GSW -> occurrence, seasonality, max_extent
    print("  ищу jrc-gsw...", flush=True)
    feat = search_first("jrc-gsw", bbox)
    if not feat:
        print("  GSW не найден", flush=True)
        return False
    gsw_url = sign(feat["assets"]["occurrence"]["href"])
    occurrence = read_band_window(gsw_url, transform, h, w, Resampling.bilinear)
    if occurrence is None:
        print("  GSW не выгрузился", flush=True)
        return False
    seasonality = read_band_window(
        sign(feat["assets"]["seasonality"]["href"]),
        transform, h, w, Resampling.nearest,
    )
    max_extent = read_band_window(
        sign(feat["assets"]["max_extent"]["href"]),
        transform, h, w, Resampling.nearest,
    )
    print(f"  GSW записан ({feat['id']})", flush=True)

    # 3. ESA WorldCover -> builtup (класс 50)
    print("  ищу esa-worldcover...", flush=True)
    feat = search_first("esa-worldcover", bbox)
    if not feat:
        print("  WorldCover не найден", flush=True)
        return False
    wc = read_band_window(
        sign(feat["assets"]["map"]["href"]),
        transform, h, w, Resampling.nearest,
    )
    builtup = (np.where(np.isin(wc, [BUILTUP_CLASS]), 1.0, 0.0) * 100.0).astype(np.float32)
    print(f"  WorldCover записан ({feat['id']})", flush=True)

    # 4. Сборка 6-канального стэка
    stack = np.stack([slope, hand, occurrence, seasonality, max_extent, builtup])
    folder.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        dest,
        np.moveaxis(stack, 0, -1).astype(np.float32),
        photometric="minisblack",
        compression="zlib",
        compressionargs={"level": 1},
        extratags=hw.geo_tags(template),
    )
    meta = {
        "bands": ["slope", "hand", "occurrence", "seasonality", "max_extent", "builtup"],
        "shape": [h, w],
        "scale_m": scale,
        "crs": "EPSG:32652",
        "sources": ["Copernicus DEM GLO-30", "JRC GSW v1.4", "ESA WorldCover v200"],
        "hand_method": "min_filter_300px_proxy",
        "builtup_class": BUILTUP_CLASS,
    }
    json_dest.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {dest.name}", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=hw.DEFAULT_DATA)
    p.add_argument("--only", help="один pair_id")
    args = p.parse_args()

    pairs = hw.load_pairs(args.data)
    rows = [r for r in pairs if not args.only or r["pair_id"] == args.only]
    ok = True
    for row in rows:
        ok &= fetch_one(args.data, row)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
