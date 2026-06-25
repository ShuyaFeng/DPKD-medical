"""
Visual sanity check: overlay each dataset's mask (semi-transparent red) on
top of its corresponding image, for N random pairs, saved as combined
side-by-side PNGs for manual inspection.

Usage:  python check_overlay.py --dataset kvasir --split train --n 8
        python check_overlay.py --dataset busi --split train --n 8
"""
import argparse, random
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["isic", "kvasir", "busi", "drive"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    folder_name = {"isic": "ISIC_HF", "kvasir": "KVASIR_HF", "busi": "BUSI_HF", "drive": "DRIVE_HF"}[args.dataset]
    base = ROOT / folder_name / args.split
    in_dir, lab_dir = base / "input", base / "label"

    ids = sorted(p.stem for p in in_dir.glob("*.png"))
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    ids = ids[:args.n]

    out_dir = Path(__file__).parent / "overlay_check"
    out_dir.mkdir(exist_ok=True)

    for iid in ids:
        img = Image.open(in_dir / f"{iid}.png").convert("RGB")
        mask = Image.open(lab_dir / f"{iid}.png").convert("L")

        img_arr = np.array(img)
        mask_arr = np.array(mask) > 127  # True where lesion/polyp

        # build a red overlay: where mask is True, blend toward red
        overlay = img_arr.copy()
        red = np.array([255, 0, 0])
        alpha = 0.5
        overlay[mask_arr] = (overlay[mask_arr] * (1 - alpha) + red * alpha).astype(np.uint8)

        # side by side: original | mask | overlay
        mask_rgb = np.stack([mask_arr.astype(np.uint8) * 255] * 3, axis=-1)
        combined = np.concatenate([img_arr, mask_rgb, overlay], axis=1)
        Image.fromarray(combined).save(out_dir / f"{args.dataset}_{iid}.png")
        print(f"  saved {args.dataset}_{iid}.png")

    print(f"\ndone -> {out_dir}  ({len(ids)} images, each: original | mask | overlay)")


if __name__ == "__main__":
    main()
