# -*- coding: utf-8 -*-
"""
generate_qualitative_fig.py

End-to-end qualitative figure generation:
  1. Load a CANAL checkpoint and a uniform checkpoint for a given dataset/epsilon/seed.
  2. Run every validation image through both models and rank by (CANAL - uniform) Dice.
  3. Export the top-N images as 4 separate PNGs each:
       {dataset}_{filename}_eps{eps}_input.png
       {dataset}_{filename}_eps{eps}_groundtruth.png
       {dataset}_{filename}_eps{eps}_uniform_dice{d:.3f}.png
       {dataset}_{filename}_eps{eps}_canal_dice{d:.3f}.png
  4. Print a ranking table.

All outputs go to qualitative_exports/ in the same directory.

Usage:
  python generate_qualitative_fig.py --dataset kvasir --epsilon 2.0 --canal-seed 200 --uniform-seed 200
  python generate_qualitative_fig.py --dataset busi   --epsilon 2.0 --canal-seed 400 --uniform-seed 400
  python generate_qualitative_fig.py --dataset isic   --epsilon 2.0 --canal-seed 200 --uniform-seed 200 --suffix rerun3
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet, vessel_dice

ap = argparse.ArgumentParser()
ap.add_argument("--dataset",      required=True, choices=["busi", "kvasir", "isic"])
ap.add_argument("--epsilon",      required=True, type=str)
ap.add_argument("--canal-seed",   required=True, type=int)
ap.add_argument("--uniform-seed", required=True, type=int)
ap.add_argument("--suffix",       default="",    help="checkpoint filename suffix, e.g. rerun3 for ISIC")
ap.add_argument("--top",          type=int, default=3, help="number of top images to export")
ap.add_argument("--size",         type=int, default=96)
ap.add_argument("--upscale",      type=int, default=6)
args = ap.parse_args()

HERE = Path(__file__).parent
dev  = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_FOLDER = {"busi": "BUSI_HF", "kvasir": "KVASIR_HF", "isic": "ISIC_HF"}[args.dataset]

if args.dataset == "busi":
    from busi_dataset import BUSIDataset
    val_ds = BUSIDataset("val", args.size)
elif args.dataset == "kvasir":
    from kvasir_dataset import KvasirDataset
    val_ds = KvasirDataset("val", args.size)
else:
    from isic_dataset import ISICDataset
    val_ds = ISICDataset("val", args.size)

in_dir = HERE.parent / "data" / DATASET_FOLDER / "val" / "input"
filenames = [Path(p).stem for p in sorted(glob.glob(str(in_dir / "*.png")))]
assert len(filenames) == len(val_ds), (
    f"{len(filenames)} filenames vs {len(val_ds)} dataset items -- mismatch!"
)


def load_model(seed, canal_bool):
    sfx = f"_{args.suffix}" if args.suffix else ""
    ckpt_path = (HERE / "checkpoints"
                 / f"{args.dataset}_K3_canal{canal_bool}_eps{args.epsilon}_seed{seed}{sfx}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = TinyUNet(in_ch=ckpt["in_ch"], num_classes=2, base=ckpt["student_base"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded {ckpt_path.name}  (best_dice={ckpt['best_dice']:.4f}, best_epoch={ckpt['best_epoch']})")
    return model


print(f"\n[{args.dataset.upper()}] device={dev}  eps={args.epsilon}  "
      f"canal_seed={args.canal_seed}  uniform_seed={args.uniform_seed}"
      + (f"  suffix={args.suffix}" if args.suffix else ""))
uniform_model = load_model(args.uniform_seed, "False")
canal_model   = load_model(args.canal_seed,   "True")


@torch.no_grad()
def per_image_dice(model, x, y):
    logits = model(x.unsqueeze(0).to(dev))
    return vessel_dice(logits, y.unsqueeze(0).to(dev))


print(f"\nRanking {len(val_ds)} validation images by (CANAL - uniform) Dice ...")
rows = []
for i in range(len(val_ds)):
    x, y = val_ds[i]
    u_dice = per_image_dice(uniform_model, x, y)
    c_dice = per_image_dice(canal_model,   x, y)
    rows.append((filenames[i], x, y, u_dice, c_dice, c_dice - u_dice))

rows.sort(key=lambda r: r[5], reverse=True)

print(f"\n{'rank':<5} {'filename':<40} {'uniform':>8} {'CANAL':>8} {'diff':>8}")
print("-" * 72)
for rank, (fname, _, _, u, c, d) in enumerate(rows[:10], 1):
    marker = " <-- will export" if rank <= args.top else ""
    print(f"{rank:<5} {fname:<40} {u:>8.4f} {c:>8.4f} {d:>+8.4f}{marker}")


def to_display_rgb(x_tensor):
    arr = x_tensor.cpu().numpy()
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    return (arr.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)


def mask_to_rgb(mask_tensor):
    arr = mask_tensor.cpu().numpy()
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[arr == 1] = (255, 255, 255)
    return rgb


@torch.no_grad()
def predict(model, x):
    logits = model(x.unsqueeze(0).to(dev))
    return logits.argmax(dim=1).squeeze(0)


out_dir = HERE / "qualitative_exports"
out_dir.mkdir(exist_ok=True)
disp_sz = args.size * args.upscale

print(f"\nExporting top {args.top} images to {out_dir}/")
for rank, (fname, x, y_gt, u_dice, c_dice, diff) in enumerate(rows[:args.top], 1):
    uni_pred = predict(uniform_model, x)
    can_pred = predict(canal_model,   x)

    prefix = f"{args.dataset}_{fname}_eps{args.epsilon}"
    Image.fromarray(to_display_rgb(x)).resize(
        (disp_sz, disp_sz), Image.Resampling.NEAREST
    ).save(out_dir / f"{prefix}_input.png")
    Image.fromarray(mask_to_rgb(y_gt)).resize(
        (disp_sz, disp_sz), Image.Resampling.NEAREST
    ).save(out_dir / f"{prefix}_groundtruth.png")
    Image.fromarray(mask_to_rgb(uni_pred)).resize(
        (disp_sz, disp_sz), Image.Resampling.NEAREST
    ).save(out_dir / f"{prefix}_uniform_dice{u_dice:.3f}.png")
    Image.fromarray(mask_to_rgb(can_pred)).resize(
        (disp_sz, disp_sz), Image.Resampling.NEAREST
    ).save(out_dir / f"{prefix}_canal_dice{c_dice:.3f}.png")

    print(f"  rank {rank}: {fname}")
    print(f"    uniform={u_dice:.4f}  CANAL={c_dice:.4f}  diff={diff:+.4f}")
    print(f"    -> {prefix}_*.png")

print(f"\nDone. {args.top * 4} PNGs saved to {out_dir}/")
