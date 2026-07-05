"""
Run every validation image through both the uniform and CANAL checkpoints,
compute per-image Dice for each, and rank by (CANAL_dice - uniform_dice)
so you can pick the clearest, most visually convincing examples.

Usage: python rank_qualitative_examples.py --dataset busi --epsilon 2.0 \
           --uniform-seed 500 --canal-seed 200
"""
import argparse, sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet, vessel_dice

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["busi", "kvasir", "isic"])
ap.add_argument("--epsilon", required=True, type=str)
ap.add_argument("--uniform-seed", required=True, type=int)
ap.add_argument("--canal-seed", required=True, type=int)
ap.add_argument("--size", type=int, default=96)
args = ap.parse_args()

HERE = Path(__file__).parent
dev = "cuda" if torch.cuda.is_available() else "cpu"

if args.dataset == "busi":
    from busi_dataset import BUSIDataset
    val_ds = BUSIDataset("val", args.size)
elif args.dataset == "kvasir":
    from kvasir_dataset import KvasirDataset
    val_ds = KvasirDataset("val", args.size)
else:
    from isic_dataset import ISICDataset
    val_ds = ISICDataset("val", args.size)

# recover the actual filenames in the same sorted order the Dataset class used
import glob
in_dir = (HERE.parent / "data" / {"busi": "BUSI_HF", "kvasir": "KVASIR_HF", "isic": "ISIC_HF"}[args.dataset]
          / "val" / "input")
filenames = [Path(p).stem for p in sorted(glob.glob(str(in_dir / "*.png")))]
assert len(filenames) == len(val_ds), f"{len(filenames)} filenames vs {len(val_ds)} dataset items -- mismatch!"

def load_model(seed, canal_bool):
    ckpt_path = HERE / "checkpoints" / f"{args.dataset}_K3_canal{canal_bool}_eps{args.epsilon}_seed{seed}.pt"
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = TinyUNet(in_ch=ckpt["in_ch"], num_classes=2, base=ckpt["student_base"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

uniform_model = load_model(args.uniform_seed, "False")
canal_model = load_model(args.canal_seed, "True")

@torch.no_grad()
def per_image_dice(model, x, y):
    logits = model(x.unsqueeze(0).to(dev))
    return vessel_dice(logits, y.unsqueeze(0).to(dev))

rows = []
for i in range(len(val_ds)):
    x, y = val_ds[i]
    u_dice = per_image_dice(uniform_model, x, y)
    c_dice = per_image_dice(canal_model, x, y)
    rows.append((filenames[i], u_dice, c_dice, c_dice - u_dice))

rows.sort(key=lambda r: r[3], reverse=True)  # biggest CANAL advantage first

print(f"\n{'filename':<25} {'uniform':>8} {'CANAL':>8} {'diff':>8}")
print("-" * 55)
for fname, u, c, d in rows[:15]:
    print(f"{fname:<25} {u:>8.4f} {c:>8.4f} {d:>+8.4f}")

print(f"\n...(showing top 15 of {len(rows)} total)")
print(f"\nWorst 5 (CANAL underperforms most):")
for fname, u, c, d in rows[-5:]:
    print(f"{fname:<25} {u:>8.4f} {c:>8.4f} {d:>+8.4f}")
