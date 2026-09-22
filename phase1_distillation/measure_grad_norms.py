"""
Measure per-example gradient norms to choose clipping threshold C.

Paper (Abadi 2016, Section 3.3) says:
  "A good way to choose C is by taking the median of the norms
   of the unclipped gradients over the course of training."

We measure on the training set (same as paper) using a random init model.
Runs 50 batches and reports median, mean, 25th and 75th percentile.

Usage:
  python measure_grad_norms.py --dataset isic
  python measure_grad_norms.py --dataset kvasir
  python measure_grad_norms.py --dataset busi
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from opacus.validators import ModuleValidator

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BATCHES = 50   # enough to get a stable median
BATCH_SIZE = 1   # one image at a time to get per-example gradients


def get_dataset(name):
    if name == "isic":
        from isic_dataset import ISICDataset
        return ISICDataset("train", 96), 3
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset
        return KvasirDataset("train", 96), 3
    if name == "busi":
        from busi_dataset import BUSIDataset
        return BUSIDataset("train", 96), 1
    raise ValueError(name)


def seg_loss(logits, y):
    ce = F.cross_entropy(logits, y, reduction="mean")
    probs = logits.softmax(1)[:, 1]
    yf = (y == 1).float()
    inter = (probs * yf).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + yf.sum(dim=(1, 2))
    dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
    return ce + 3.0 * dice.mean()


def measure(dataset, in_ch):
    torch.manual_seed(42)
    model = TinyUNet(in_ch=in_ch, num_classes=2, base=16)
    model = ModuleValidator.fix(model)   # BatchNorm → GroupNorm (same as training)
    for mod in model.modules():
        if isinstance(mod, torch.nn.ReLU):
            mod.inplace = False
    model = model.to(DEVICE)
    model.train()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    norms = []

    for i, (x, y) in enumerate(loader):
        if i >= N_BATCHES:
            break
        model.zero_grad()
        loss = seg_loss(model(x.to(DEVICE)), y.to(DEVICE))
        loss.backward()
        # flatten all parameter gradients into one vector and compute L2 norm
        grad_vec = torch.cat([
            p.grad.detach().flatten()
            for p in model.parameters()
            if p.grad is not None
        ])
        norms.append(grad_vec.norm(2).item())

    norms = np.array(norms)
    print(f"\nGradient norm statistics over {len(norms)} examples:")
    print(f"  Median (recommended C) : {np.median(norms):.4f}")
    print(f"  Mean                   : {np.mean(norms):.4f}")
    print(f"  25th percentile        : {np.percentile(norms, 25):.4f}")
    print(f"  75th percentile        : {np.percentile(norms, 75):.4f}")
    print(f"  Min                    : {np.min(norms):.4f}")
    print(f"  Max                    : {np.max(norms):.4f}")
    print(f"\n  → Use C = {np.median(norms):.2f} for this dataset")
    return np.median(norms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["isic", "kvasir", "busi"])
    args = parser.parse_args()

    print(f"Measuring gradient norms on {args.dataset} training set...")
    ds, in_ch = get_dataset(args.dataset)
    print(f"Train size: {len(ds)}  in_ch={in_ch}  device={DEVICE}")
    measure(ds, in_ch)


if __name__ == "__main__":
    main()
