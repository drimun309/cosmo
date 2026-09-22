"""Комбинированный инференс: UNetSmall (flood) + UNetMulti (water_pre, water_peak)."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

import hw
import model as model_mod
import train
import train_multi


def predict_full_single(net, ch, device, tile=256, stride=192):
    return train.predict_full(net, ch, device, tile=tile, stride=stride)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=hw.DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    p.add_argument("--ckpt-flood", type=Path, default=Path(__file__).resolve().parent / "out" / "unet_all.pt")
    p.add_argument("--ckpt-water", type=Path, default=Path(__file__).resolve().parent / "out" / "unet_multi.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--thr-flood", type=float, default=0.8)
    p.add_argument("--thr-water", type=float, default=0.5)
    args = p.parse_args()

    pairs = hw.load_pairs(args.data)
    net_flood = model_mod.UNetSmall(in_ch=10, base=32).to(args.device)
    net_flood.load_state_dict(torch.load(args.ckpt_flood, map_location=args.device))
    net_water = model_mod.UNetMulti(in_ch=10, base=32).to(args.device)
    net_water.load_state_dict(torch.load(args.ckpt_water, map_location=args.device))

    rows_out = []
    for row in pairs:
        a = train_multi.load_pair_arrays(args.data, row)
        shape = a["ref_mask"].shape[1:]
        ch = train_multi.stack_channels(a, shape)
        if ch is None:
            continue
        ch = train_multi.normalize(ch)

        # flood от UNetSmall (thr=0.8 дал q=0.435 на rain_flood)
        prob_flood = predict_full_single(net_flood, ch, args.device)
        # water_pre/peak от UNetMulti
        prob3 = train_multi.predict_multi(net_water, ch, args.device)
        prob_pre = prob3[..., 1]
        prob_peak = prob3[..., 2]

        template = args.data / row["reference_mask"]
        aux_folder = args.data / row["rasters_dir"]
        if (aux_folder / "AUX_terrain_gsw.tif").exists():
            aux = hw.read_aux(aux_folder, template)
            occ = aux[2]
            permanent = (np.isfinite(occ) & (occ >= 80))
        else:
            permanent = np.zeros_like(prob_flood, dtype=bool)

        flood_bin = ((prob_flood > args.thr_flood) & ~permanent).astype(np.uint8)
        water_pre_bin = (prob_pre > args.thr_water).astype(np.uint8)
        water_peak_bin = (prob_peak > args.thr_water).astype(np.uint8)

        pred_dir = args.out / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        hw.write_flood(pred_dir / f"{row['pair_id']}_flood.tif", flood_bin, template)
        # сохраняем и pre/peak, чтобы сервис мог отдавать площади по обоим слоям
        tifffile.imwrite(str(pred_dir / f"{row['pair_id']}_water_pre.tif"),
                         water_pre_bin.astype(np.uint8), compression="zlib",
                         extratags=hw.geo_tags(template))
        tifffile.imwrite(str(pred_dir / f"{row['pair_id']}_water_peak.tif"),
                         water_peak_bin.astype(np.uint8), compression="zlib",
                         extratags=hw.geo_tags(template))

        _, _, scale = hw.geo(template)
        flood_ha = hw.hectares(int(flood_bin.sum()), scale)
        pre_ha = hw.hectares(int(water_pre_bin.sum()), scale)
        peak_ha = hw.hectares(int(water_peak_bin.sum()), scale)

        rows_out.append({
            "pair_id": row["pair_id"],
            "flood_ha": round(flood_ha, 2),
            "water_pre_ha": round(pre_ha, 2),
            "water_peak_ha": round(peak_ha, 2),
        })
        print(f"{row['pair_id']}: flood={flood_ha:.0f} pre={pre_ha:.0f} peak={peak_ha:.0f}")

    with (args.out / "submission.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["pair_id", "flood_ha", "water_pre_ha", "water_peak_ha"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nsubmission записан: {args.out / 'submission.csv'}")


if __name__ == "__main__":
    main()
