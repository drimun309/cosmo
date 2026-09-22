"""Чтение сетки кейса и подсчёт Score. Площадь: пиксель 10 м = 0.01 га."""

import csv
import json
import math
import zlib
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "hydrowatch_amur"

# Веса и пороги знаменателя — из постановки, не из подбора.
W_FLOOD, W_PEAK, W_PRE, W_SPEC = 0.45, 0.25, 0.15, 0.15
FLOOR_FLOOD_HA, FLOOR_WATER_HA, SPEC_FRAC = 50.0, 200.0, 0.005
REF_BANDS = ("flood", "water_pre", "water_peak", "permanent", "receded")


def load_pairs(data: Path) -> list[dict]:
    with (data / "pairs.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"пустой pairs.csv в {data}")
    return rows


def hectares(count: int, scale_m: float) -> float:
    return round(count * scale_m * scale_m / 10_000.0, 2)


def _bands(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.ndim == 2 and n == 1:
        return arr[None]
    if arr.ndim == 3 and arr.shape[0] == n:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == n:
        return np.moveaxis(arr, -1, 0)
    raise ValueError(f"ожидалось {n} каналов, пришло {arr.shape}")


def _decode_f32_tile(blob: bytes, tw: int, th: int) -> np.ndarray:
    raw = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
    # ponytail: predictor 3 = горизонтальная разность байт, затем группы по старшинству
    acc = np.cumsum(raw.reshape(th, tw * 4), axis=1, dtype=np.uint32) & 0xFF
    groups = acc.reshape(th, 4, tw).astype(np.uint8)
    be = np.empty((th, tw, 4), dtype=np.uint8)
    be[..., 0] = groups[:, 0]
    be[..., 1] = groups[:, 1]
    be[..., 2] = groups[:, 2]
    be[..., 3] = groups[:, 3]
    return be.view(np.dtype(">f4")).reshape(th, tw)


def _read_fp_tiles(tif: tifffile.TiffFile, page) -> np.ndarray:
    if page.predictor != 3 or not page.is_tiled or page.dtype != np.float32:
        raise ValueError(
            f"неподдерживаемый TIFF: predictor={page.predictor} tiled={page.is_tiled} dtype={page.dtype}"
        )
    if int(page.planarconfig) != 2:
        raise ValueError("float-predictor пока только planar=separate, как AUX в этом наборе")
    tw, th = page.tilewidth, page.tilelength
    w, h, spp = page.imagewidth, page.imagelength, page.samplesperpixel
    ntx, nty = math.ceil(w / tw), math.ceil(h / th)
    out = np.empty((spp, h, w), np.float32)
    fh = tif.filehandle
    for band in range(spp):
        for ty in range(nty):
            for tx in range(ntx):
                i = band * ntx * nty + ty * ntx + tx
                fh.seek(page.dataoffsets[i])
                tile = _decode_f32_tile(fh.read(page.databytecounts[i]), tw, th)
                r0, c0 = ty * th, tx * tw
                r1, c1 = min(r0 + th, h), min(c0 + tw, w)
                out[band, r0:r1, c0:c1] = tile[: r1 - r0, : c1 - c0]
    return out


def read_tif(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        try:
            return tif.asarray()
        except ValueError as e:
            if "PREDICTOR" not in str(e) and "imagecodecs" not in str(e):
                raise
            return _read_fp_tiles(tif, page)


def geo(path: Path) -> tuple[float, float, float]:
    """Верхний левый угол (x, y) и размер пикселя, м. Ось y смотрит на север."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        sx, sy, _ = page.tags["ModelPixelScaleTag"].value
        i, j, _, x, y, _ = page.tags["ModelTiepointTag"].value
    return x - i * sx, y + j * sy, float(sx)


def warp_nearest(src, src_xy, src_scale, dst_xy, dst_scale, dst_hw):
    sh, sw = src.shape[-2:]
    dh, dw = dst_hw
    y = dst_xy[1] - (np.arange(dh) + 0.5) * dst_scale
    x = dst_xy[0] + (np.arange(dw) + 0.5) * dst_scale
    sr = np.floor((src_xy[1] - y) / src_scale).astype(np.int32)
    sc = np.floor((x - src_xy[0]) / src_scale).astype(np.int32)
    ok_r = (sr >= 0) & (sr < sh)
    ok_c = (sc >= 0) & (sc < sw)
    picked = src[:, np.clip(sr, 0, sh - 1)[:, None], np.clip(sc, 0, sw - 1)[None, :]]
    picked = picked.copy()
    picked[:, ~(ok_r[:, None] & ok_c[None, :])] = np.nan
    return picked


def read_aux(folder: Path, template: Path) -> np.ndarray:
    path = folder / "AUX_terrain_gsw.tif"
    arr = _bands(read_tif(path), 6)
    sxy, ss = geo(path)[:2], geo(path)[2]
    dxy_s = geo(template)
    with tifffile.TiffFile(template) as tif:
        h, w = tif.pages[0].imagelength, tif.pages[0].imagewidth
    return warp_nearest(arr, sxy, ss, dxy_s[:2], dxy_s[2], (h, w))


def reference_areas(data: Path, pairs: list[dict]) -> dict[str, dict]:
    out = {}
    for row in pairs:
        path = data / row["reference_mask"]
        arr = _bands(read_tif(path), 5)
        _, _, scale = geo(path)
        counts = {name: int((arr[i] > 0).sum()) for i, name in enumerate(REF_BANDS)}
        out[row["pair_id"]] = {
            "ha": {name: hectares(counts[name], scale) for name in REF_BANDS},
            "aoi_ha": round(float(row["aoi_km2"]) * 100.0, 2),
            "kind": row["event_kind"],
            "scale": scale,
            "shape": arr.shape[1:],
        }
    return out


def q_one(pred: float, ref: float, floor: float) -> float:
    return max(0.0, 1.0 - abs(pred - ref) / max(ref, floor))


def score_areas(pred: dict[str, dict], ref: dict[str, dict], pairs: list[dict]) -> dict:
    qf, qp, qpre, spec = [], [], [], []
    rows = []
    for row in pairs:
        pid = row["pair_id"]
        r, p = ref[pid]["ha"], pred[pid]
        item = {
            "pair_id": pid,
            "kind": row["event_kind"],
            "flood_ha": p["flood_ha"],
            "water_pre_ha": p["water_pre_ha"],
            "water_peak_ha": p["water_peak_ha"],
            "ref_flood_ha": r["flood"],
            "ref_water_pre_ha": r["water_pre"],
            "ref_water_peak_ha": r["water_peak"],
        }
        if row["event_kind"] == "baseline":
            frac = max(0.0, p["flood_ha"] - r["flood"]) / ref[pid]["aoi_ha"]
            item["spec"] = 1.0 - min(1.0, frac / SPEC_FRAC)
            spec.append(item["spec"])
        else:
            item["q_flood"] = q_one(p["flood_ha"], r["flood"], FLOOR_FLOOD_HA)
            item["q_peak"] = q_one(p["water_peak_ha"], r["water_peak"], FLOOR_WATER_HA)
            item["q_pre"] = q_one(p["water_pre_ha"], r["water_pre"], FLOOR_WATER_HA)
            qf.append(item["q_flood"])
            qp.append(item["q_peak"])
            qpre.append(item["q_pre"])
        rows.append(item)
    parts = {
        "Q_flood": sum(qf) / len(qf),
        "Q_water_peak": sum(qp) / len(qp),
        "Q_water_pre": sum(qpre) / len(qpre),
        "Spec_base": sum(spec) / len(spec),
    }
    parts["Score"] = (
        W_FLOOD * parts["Q_flood"]
        + W_PEAK * parts["Q_water_peak"]
        + W_PRE * parts["Q_water_pre"]
        + W_SPEC * parts["Spec_base"]
    )
    return {"parts": parts, "rows": rows}


def geo_tags(template: Path) -> list:
    tags = []
    with tifffile.TiffFile(template) as tif:
        page = tif.pages[0]
        for code in (33550, 33922, 34735, 34737):
            tag = page.tags.get(code)
            if tag is not None:
                tags.append((code, tag.dtype, tag.count, tag.value, False))
    return tags


def write_flood(path: Path, mask: np.ndarray, template: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        mask.astype(np.uint8),
        photometric="minisblack",
        compression="zlib",
        extratags=geo_tags(template),
    )


def load_submission(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        out[row["pair_id"]] = {
            "flood_ha": float(row["flood_ha"]),
            "water_pre_ha": float(row["water_pre_ha"]),
            "water_peak_ha": float(row["water_peak_ha"]),
        }
    return out


def band_names(path: Path) -> list[str]:
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["bands"]
