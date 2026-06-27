"""
Measure the real per-image bottleneck Frobenius norm scale, per Dr. Tian's
instructions for setting CLIP_CAPS honestly:
  "push a few batches through one teacher, look at the per-image bottleneck
   Frobenius norm (e3.flatten(1).norm()), and set CLIP_CAPS to about its
   95th percentile. Do NOT tune it to the result."

Usage: python diagnose_caps_scale.py --dataset busi
       python diagnose_caps_scale.py --dataset kvasir
       python diagnose_caps_scale.py --dataset isic
"""
import argparse, sys
from pathlib import Path
import torch
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from drive_pate_poc import train_K_teachers
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["busi", "kvasir", "isic"])
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"

if args.dataset == "busi":
    from busi_dataset import BUSIDataset
    train_ds = BUSIDataset("train", 96); in_ch = 1
elif args.dataset == "kvasir":
    from kvasir_dataset import KvasirDataset
    train_ds = KvasirDataset("train", 96); in_ch = 3
else:
    from isic_dataset import ISICDataset
    train_ds = ISICDataset("train", 96); in_ch = 3

tk = train_K_teachers(train_ds, 3, dev, n_epochs=60, in_ch=in_ch)
teacher = tk[0][0]
teacher.eval()

loader = DataLoader(train_ds, batch_size=8, shuffle=False)
all_norms = []
with torch.no_grad():
    for i, (x, _) in enumerate(loader):
        _, _, e3 = teacher.encode(x.to(dev))
        norms = e3.flatten(1).norm(dim=1)
        all_norms.extend(norms.cpu().tolist())
        if i >= 9:  # a few batches, per instructions -- 10 batches here
            break

all_norms = np.array(all_norms)
p95 = np.percentile(all_norms, 95)
print(f"\n[{args.dataset}] per-image bottleneck Frobenius norm:")
print(f"  n={len(all_norms)}  min={all_norms.min():.4f}  max={all_norms.max():.4f}  "
      f"mean={all_norms.mean():.4f}  p95={p95:.4f}")
