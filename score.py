"""Score постановки. На эталоне против самого себя обязан быть 1.0.

    python score.py
    python score.py --submission out/submission.csv --predictions out/predictions
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

import hw


def _check_write(template: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.tif"
        hw.write_flood(path, np.array([[0, 1], [1, 0]], np.uint8), template)
        import tifffile

        with tifffile.TiffFile(path) as tif:
            scale = tif.pages[0].tags["ModelPixelScaleTag"].value[0]
            arr = tif.asarray()
        if scale != 10 or int(arr.sum()) != 2:
            raise SystemExit(f"геопривязка маски не сохранилась: scale={scale}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=hw.DEFAULT_DATA)
    p.add_argument("--submission", type=Path)
    p.add_argument("--predictions", type=Path)
    args = p.parse_args()
    pairs = hw.load_pairs(args.data)
    ref = hw.reference_areas(args.data, pairs)
    _check_write(args.data / pairs[0]["reference_mask"])
    if args.submission is None:
        pred = {
            pid: {
                "flood_ha": v["ha"]["flood"],
                "water_pre_ha": v["ha"]["water_pre"],
                "water_peak_ha": v["ha"]["water_peak"],
            }
            for pid, v in ref.items()
        }
        title = "эталон против эталона"
    else:
        pred = hw.load_submission(args.submission)
        missing = [row["pair_id"] for row in pairs if row["pair_id"] not in pred]
        if missing:
            raise SystemExit("в сабмите нет пар: " + ", ".join(missing))
        title = str(args.submission)
        if args.predictions:
            _check_masks(args.predictions, pred, ref)
    result = hw.score_areas(pred, ref, pairs)
    print(title)
    print(f"{'pair_id':<42} {'flood':>10} {'pre':>10} {'peak':>10}  роль")
    for row in result["rows"]:
        print(
            f"{row['pair_id']:<42} {row['flood_ha']:10.2f} {row['water_pre_ha']:10.2f} "
            f"{row['water_peak_ha']:10.2f}  {row['kind']}"
        )
    parts = result["parts"]
    print(
        f"Score {parts['Score']:.4f}  Q_flood {parts['Q_flood']:.4f}  "
        f"Q_peak {parts['Q_water_peak']:.4f}  Q_pre {parts['Q_water_pre']:.4f}  "
        f"Spec {parts['Spec_base']:.4f}"
    )
    if args.submission is None and abs(parts["Score"] - 1.0) > 1e-9:
        raise SystemExit("Score эталона не 1.0 — ошибка в площадях")
    if args.submission is None:
        print("self-check ok")


def _check_masks(pred_dir: Path, pred: dict, ref: dict) -> None:
    for pid, areas in pred.items():
        path = pred_dir / f"{pid}_flood.tif"
        if not path.exists():
            raise SystemExit(f"нет маски {path}")
        arr = hw.read_tif(path)
        mask = arr if arr.ndim == 2 else arr[..., 0]
        scale = ref[pid]["scale"]
        got = hw.hectares(int((mask > 0).sum()), scale)
        base = max(areas["flood_ha"], 0.01)
        if abs(got - areas["flood_ha"]) / base > 0.02:
            raise SystemExit(f"{pid}: маска {got} га, csv {areas['flood_ha']} га, расхождение > 2%")


if __name__ == "__main__":
    sys.exit(main())
