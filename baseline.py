"""Базовая линия: Оцу по VV, индексы оптики, фильтр рельефа.

Снимки класть рядом с паспортами:
  rasters/<event>/<aoi>/S1_pre_*.tif
  rasters/<event>/<aoi>/S1_peak_*.tif
  rasters/<event>/<aoi>/SENTINEL2_pre_*.tif   (если есть)
  rasters/<event>/<aoi>/SENTINEL2_peak_*.tif

    python baseline.py
    python score.py --submission out/submission.csv --predictions out/predictions
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

import hw

OPT_NAMES = ("NDWI", "MNDWI", "NDVI", "AWEISH")


def _logic_check() -> None:
    pre = np.array([[1, 1, 0], [0, 0, 0]], bool)
    peak = np.array([[1, 0, 1], [1, 0, 0]], bool)
    perm = np.array([[1, 0, 0], [0, 0, 0]], bool)
    flood = peak & ~pre & ~perm
    receded = pre & ~peak & ~perm
    assert flood.tolist() == [[False, False, True], [True, False, False]]
    assert receded.tolist() == [[False, True, False], [False, False, False]]
    sar = np.array([True, True, False])
    opt = np.array([True, False, True])
    valid = np.array([True, True, False])
    fused = fuse(sar, opt, valid, True)
    assert fused.tolist() == [True, False, False]


def fuse(sar, opt, opt_valid, optical_ok):
    if not optical_ok:
        return sar
    out = sar.copy()
    out[opt_valid] = sar[opt_valid] & opt[opt_valid]
    return out


def smooth_vv(vv, valid, size):
    if size <= 1 or not valid.any():
        return vv
    fill = float(np.median(vv[valid]))
    out = ndimage.median_filter(np.where(valid, vv, fill), size=size)
    return np.where(valid, out, np.nan)


def vv_threshold(vv, valid, lo, hi):
    sample = vv[valid]
    if sample.size < 64:
        return (lo + hi) / 2
    if sample.size > 1_000_000:
        sample = sample[:: sample.size // 1_000_000]
    try:
        thr = float(threshold_otsu(sample))
    except ValueError:
        thr = (lo + hi) / 2
    return float(np.clip(thr, lo, hi))


def remove_small(mask, min_pixels):
    if min_pixels <= 1 or not mask.any():
        return mask
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return keep[labels]


def scene_tif(folder: Path, prefix: str) -> Path | None:
    hits = sorted(folder.glob(prefix + "*.tif"))
    return hits[0] if hits else None


def expected_tif(folder: Path, prefix: str) -> Path:
    js = sorted(folder.glob(prefix + "*.json"))
    if js:
        return js[0].with_suffix(".tif")
    return folder / f"{prefix}MISSING.tif"


def optical_water(arr, names, cfg):
    idx = {name: i for i, name in enumerate(names)}
    if any(name not in idx for name in OPT_NAMES):
        return None
    ndwi = arr[idx["NDWI"]]
    mndwi = arr[idx["MNDWI"]]
    ndvi = arr[idx["NDVI"]]
    awe = arr[idx["AWEISH"]]
    valid = np.isfinite(ndwi) & np.isfinite(mndwi) & np.isfinite(ndvi) & np.isfinite(awe)
    water = (
        valid
        & (mndwi > cfg["mndwi_min"])
        & (ndwi > cfg["ndwi_min"])
        & (ndvi <= cfg["ndvi_max"])
        & (awe > cfg["aweish_min"])
    )
    return water, valid, float(valid.mean())


def sar_water(path: Path, cfg) -> np.ndarray:
    names = hw.band_names(path)
    arr = hw._bands(hw.read_tif(path), len(names))
    vv = arr[names.index("VV")]
    valid = np.isfinite(vv) & (vv >= cfg["vv_valid_min"]) & (vv <= cfg["vv_valid_max"])
    vv = smooth_vv(vv, valid, int(cfg["median_size"]))
    thr = vv_threshold(vv, valid, cfg["vv_db_min"], cfg["vv_db_max"])
    return valid & (vv < thr), thr, arr.shape[1:]


def one_date(s1_path: Path, s2_path: Path | None, keep: np.ndarray, cfg) -> tuple[np.ndarray, float]:
    water, thr, shape = sar_water(s1_path, cfg)
    if water.shape != keep.shape:
        raise SystemExit(
            f"{s1_path.name}: размер {water.shape}, сетка эталона {keep.shape}. "
            "Снимок нужно привести к сетке reference mask."
        )
    optical_ok = False
    if s2_path is not None:
        names = hw.band_names(s2_path)
        opt_arr = hw._bands(hw.read_tif(s2_path), len(names))
        parsed = optical_water(opt_arr, names, cfg)
        if parsed is not None and opt_arr.shape[1:] == keep.shape:
            opt, valid, frac = parsed
            optical_ok = frac >= cfg["optical_min_valid_frac"]
            water = fuse(water, opt, valid, optical_ok)
    water = remove_small(water & keep, int(cfg["min_pixels"]))
    return water, thr


def hydro_keep(aux, cfg) -> tuple[np.ndarray, np.ndarray]:
    slope, hand, occurrence, _, _, builtup = aux
    keep = np.isfinite(slope) & (slope <= cfg["slope_max_deg"])
    keep &= np.isfinite(hand) & (hand <= cfg["hand_max_m"])
    keep &= ~(np.isfinite(builtup) & (builtup >= cfg["builtup_min"]))
    permanent = np.isfinite(occurrence) & (occurrence >= cfg["occurrence_permanent"])
    return keep, permanent


def main() -> None:
    _logic_check()
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=hw.DEFAULT_DATA)
    p.add_argument("--config", type=Path, default=hw.ROOT / "config.json")
    p.add_argument("--out", type=Path, default=hw.ROOT / "out")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    np.random.seed(int(cfg["seed"]))
    pairs = hw.load_pairs(args.data)
    missing = []
    for row in pairs:
        folder = args.data / row["rasters_dir"]
        for prefix in ("S1_pre_", "S1_peak_"):
            if scene_tif(folder, prefix) is None:
                missing.append(expected_tif(folder, prefix))
    if missing:
        print("В lite-архиве нет растров Sentinel-1, только паспорта json.")
        print("Базовая линия ждёт GeoTIFF рядом с паспортом, в сетке эталонной маски:")
        for path in missing:
            print(" ", path)
        raise SystemExit(2)

    pred_dir = args.out / "predictions"
    rows_out = []
    for row in pairs:
        folder = args.data / row["rasters_dir"]
        template = args.data / row["reference_mask"]
        aux = hw.read_aux(folder, template)
        keep, permanent = hydro_keep(aux, cfg)
        pre, thr_pre = one_date(scene_tif(folder, "S1_pre_"), scene_tif(folder, "SENTINEL2_pre_"), keep, cfg)
        peak, thr_peak = one_date(scene_tif(folder, "S1_peak_"), scene_tif(folder, "SENTINEL2_peak_"), keep, cfg)
        flood = peak & ~pre & ~permanent
        flood = remove_small(flood, int(cfg["min_pixels"]))
        _, _, scale = hw.geo(template)
        areas = {
            "pair_id": row["pair_id"],
            "flood_ha": hw.hectares(int(flood.sum()), scale),
            "water_pre_ha": hw.hectares(int(pre.sum()), scale),
            "water_peak_ha": hw.hectares(int(peak.sum()), scale),
        }
        if areas["flood_ha"] > areas["water_peak_ha"]:
            raise SystemExit(f"{row['pair_id']}: flood > water_peak")
        hw.write_flood(pred_dir / f"{row['pair_id']}_flood.tif", flood, template)
        rows_out.append(areas)
        print(
            f"{row['pair_id']}: flood {areas['flood_ha']}  pre {areas['water_pre_ha']}  "
            f"peak {areas['water_peak_ha']}  thr {thr_pre:.2f}/{thr_peak:.2f} dB"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "submission.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["pair_id", "flood_ha", "water_pre_ha", "water_peak_ha"])
        w.writeheader()
        w.writerows(rows_out)
    print(csv_path)


if __name__ == "__main__":
    sys.exit(main())
