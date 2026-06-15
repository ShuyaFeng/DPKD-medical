"""
DRIVE local noise probe — clean vs uniform vs channel_WF on REAL DRIVE images.

A self-contained reproduction of phase1_drive_channel_noise_v2.py that runs on
CPU/MPS without mmseg, the teacher checkpoint, or the cluster:

  1. Load 20 train + 20 val DRIVE images from data/DRIVE_HF, resized to 96x96.
  2. Train a TinyUNet teacher (base=32, bottleneck = 128 ch) from scratch.
  3. Compute per-channel bottleneck importance via gradient (GKD Eq. 1).
  4. Estimate per-channel L2 caps (90th percentile across train images).
  5. For ε ∈ {0.5, 1, 2, 4, 8, 16}:
       - inject Gaussian noise on the teacher's clean bottleneck for each val image
       - mechanisms compared: NONE (clean), uniform, channel_WF
       - decode → measure vessel Dice
  6. Print a 3-way comparison table + save JSON.

Caveats vs the cluster version:
  - Importance + caps come from DRIVE train (not HRF) — this is the *pre-pivot*
    noise probe; for a public-proxy run see compute_public_proxy.py on cluster.
  - Tiny teacher (~0.5M params vs mmseg's ~7M) so absolute Dice is lower than
    the 0.787 the cluster gets. What matters is the *relative* behaviour of
    uniform vs WF vs clean.
"""

import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from skimage import transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse the model + DP algorithms from synthetic_demo.py
sys.path.insert(0, str(Path(__file__).parent))
from synthetic_demo import (
    TinyUNet,
    eps_to_rho,
    waterfilling_sigma,
    uniform_sigma,
    clip_and_normalise,
    denormalise,
    task_loss,
)


DRIVE_ROOT = Path(__file__).resolve().parent.parent / "data" / "DRIVE_HF"


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class DriveDataset(Dataset):
    """Real DRIVE images, resized to (size, size) RGB + binary vessel mask."""

    def __init__(self, split: str = "train", size: int = 96):
        in_dir = DRIVE_ROOT / split / "input"
        lab_dir = DRIVE_ROOT / split / "label"
        self.items = []
        for tif in sorted(glob.glob(str(in_dir / "*.tif"))):
            stem = Path(tif).stem
            cands = [lab_dir / f"{stem}.png",
                     lab_dir / f"{stem}_manual1.png"]
            png = next((p for p in cands if p.exists()), None)
            if png is None:
                continue
            img = np.array(Image.open(tif), dtype=np.uint8)             # (H, W, 3)
            lab = np.array(Image.open(png), dtype=np.uint8)             # (H, W)
            img_r = (transform.resize(img, (size, size, 3),
                                      preserve_range=True,
                                      anti_aliasing=True).astype(np.float32)
                     / 255.0)
            lab_r = transform.resize(lab.astype(float), (size, size),
                                     preserve_range=True,
                                     anti_aliasing=False, order=0)
            x = img_r.transpose(2, 0, 1)                                 # (3, H, W)
            y = (lab_r > 127).astype(np.int64)                           # (H, W)
            self.items.append((torch.from_numpy(x), torch.from_numpy(y)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# -------------------------------------------------------------------------
# Metric
# -------------------------------------------------------------------------

def vessel_dice(logits: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> float:
    pred = logits.argmax(dim=1)
    pv = (pred == 1)
    gv = (gt == 1)
    inter = (pv & gv).sum(dim=(1, 2)).float()
    denom = pv.sum(dim=(1, 2)).float() + gv.sum(dim=(1, 2)).float()
    return float(((2 * inter + eps) / (denom + eps)).mean().item())


@torch.no_grad()
def evaluate_vessel_dice(model, loader, device):
    model.eval()
    scores = [vessel_dice(model(x.to(device)), y.to(device)) for x, y in loader]
    return float(np.mean(scores))


# -------------------------------------------------------------------------
# Train teacher
# -------------------------------------------------------------------------

def train_teacher(model, loader, n_epochs, lr, device, val_loader=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(n_epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = task_loss(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            tag = ""
            if val_loader is not None:
                tag = f"  val_dice={evaluate_vessel_dice(model, val_loader, device):.4f}"
                model.train()
            print(f"  ep {ep+1:3d}/{n_epochs} loss={total/len(loader):.4f}{tag}")


# -------------------------------------------------------------------------
# Importance + caps
# -------------------------------------------------------------------------

def compute_importance(model, loader, device):
    model.eval()
    Cb = model.base * 4
    grad_abs_sum = torch.zeros(Cb, device=device)
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Enable a grad path through the input so e3 stays in the autograd
        # graph even when the teacher's params are frozen (requires_grad=False),
        # e.g. teachers returned by train_K_teachers. Without this,
        # e3.retain_grad() raises "can't retain_grad on requires_grad=False".
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        e1, e2, e3 = model.encode(x)
        e3.retain_grad()
        loss = task_loss(model.decode(e1, e2, e3), y)
        loss.backward()
        grad_abs_sum += e3.grad.detach().abs().mean(dim=(0, 2, 3))
        n += 1
    return (grad_abs_sum / n).cpu()


@torch.no_grad()
def collect_caps(model, loader, device, q: float = 0.9):
    model.eval()
    norms = []
    for x, _ in loader:
        x = x.to(device)
        _, _, e3 = model.encode(x)
        for b in range(e3.shape[0]):
            norms.append(e3[b].flatten(1).norm(dim=1).cpu())
    return torch.quantile(torch.stack(norms), q, dim=0)


# -------------------------------------------------------------------------
# Noise probe — the 3-way comparison
# -------------------------------------------------------------------------

def noise_probe(model, val_loader, importance, caps, device, epsilons,
                K: int = 1, delta_dp: float = 1e-5):
    """Return mean vessel Dice for {no_noise, uniform[eps], channel_WF[eps]}."""
    Cb = importance.shape[0]
    deltas = torch.full((Cb,), 2.0 / K, device=device)

    sigma_table: Dict[float, Dict[str, torch.Tensor]] = {}
    for eps in epsilons:
        rho = eps_to_rho(eps, delta_dp)
        sigma_table[eps] = {
            "uniform":    uniform_sigma(deltas, rho),
            "channel_WF": waterfilling_sigma(deltas, importance.to(device), rho),
        }

    results = {"no_noise": [],
               "uniform":    {e: [] for e in epsilons},
               "channel_WF": {e: [] for e in epsilons}}

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            e1, e2, e3 = model.encode(x)

            # --- clean (no noise) reference ---
            results["no_noise"].append(vessel_dice(model.decode(e1, e2, e3), y))

            # --- noisy ---
            bn_norm = clip_and_normalise(e3, caps.to(device))
            B, C, H, W = bn_norm.shape
            for eps in epsilons:
                for mech in ["uniform", "channel_WF"]:
                    sigma = sigma_table[eps][mech]
                    noise = (torch.randn(B, C, H, W, device=device)
                             * sigma.view(1, C, 1, 1))
                    bn_noisy = denormalise(bn_norm + noise, caps.to(device))
                    logits = model.decode(e1, e2, bn_noisy)
                    results[mech][eps].append(vessel_dice(logits, y))

    mean = {
        "no_noise":   float(np.mean(results["no_noise"])),
        "uniform":    {e: float(np.mean(results["uniform"][e]))    for e in epsilons},
        "channel_WF": {e: float(np.mean(results["channel_WF"][e])) for e in epsilons},
    }
    return mean, sigma_table


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    print(f"Device: {device}")

    print("\n[1/4] Loading real DRIVE (resize 96x96)...")
    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  size=96x96 RGB")

    print("\n[2/4] Training teacher (TinyUNet base=32, ~0.5M params)...")
    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"  teacher params: {n_params/1e6:.2f}M")
    t0 = time.time()
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3,
                  device=device, val_loader=val_loader)
    print(f"  teacher training time: {time.time()-t0:.1f}s")
    clean_dice = evaluate_vessel_dice(teacher, val_loader, device)
    print(f"  Teacher CLEAN vessel Dice on val: {clean_dice:.4f}  ← upper bound")

    print("\n[3/4] Computing channel importance + per-channel caps...")
    importance = compute_importance(teacher, train_loader, device)
    caps       = collect_caps(teacher, train_loader, device)
    print(f"  importance: C={importance.shape[0]}  "
          f"min={importance.min():.3e}  max={importance.max():.3e}  "
          f"ratio={(importance.max()/importance.min()).item():.1f}x")
    print(f"  caps:  min={caps.min():.3f}  max={caps.max():.3f}  "
          f"mean={caps.mean():.3f}")

    print("\n[4/4] Noise probe across epsilons (channel_WF vs uniform vs clean)...")
    epsilons = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    K_probe  = 10   # PATE-style sensitivity to match phase1_channel_noise_summary_v2.json
    means, sigma_table = noise_probe(teacher, val_loader, importance, caps,
                                     device, epsilons, K=K_probe)

    print(f"\n  CLEAN (no noise, reference):    {means['no_noise']:.4f}")
    print()
    print(f"  {'eps':>5}  {'uniform':>10}  {'channel_WF':>11}  "
          f"{'WF-uniform':>11}  {'drop from clean (WF)':>22}")
    print("  " + "-" * 70)
    lift = {}
    for e in epsilons:
        u = means["uniform"][e]
        w = means["channel_WF"][e]
        lift[e] = w - u
        drop = w - means["no_noise"]
        print(f"  {e:>5.1f}  {u:>10.4f}  {w:>11.4f}  "
              f"{w-u:>+11.4f}  {drop:>+22.4f}")

    print("\n  Sigma allocation diagnostic (top-10 important vs bot-10):")
    top10 = torch.argsort(importance, descending=True)[:10]
    bot10 = torch.argsort(importance, descending=False)[:10]
    print(f"  {'eps':>5}  {'uniform σ':>10}  {'WF σ top':>10}  "
          f"{'WF σ bot':>10}  {'ratio':>8}")
    print("  " + "-" * 52)
    for e in epsilons:
        sw = sigma_table[e]["channel_WF"]
        su = sigma_table[e]["uniform"]
        rat = (sw[bot10].mean() / sw[top10].mean()).item()
        print(f"  {e:>5.1f}  {su[0].item():>10.4f}  "
              f"{sw[top10].mean().item():>10.4f}  "
              f"{sw[bot10].mean().item():>10.4f}  {rat:>7.3f}x")

    out = {
        "device": device,
        "img_size": 96,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "K_probe": K_probe,
        "teacher_params_M": round(n_params / 1e6, 3),
        "teacher_clean_vessel_dice": clean_dice,
        "importance_stats": {
            "min": float(importance.min()),
            "max": float(importance.max()),
            "ratio_max_min": float(importance.max() / importance.min()),
        },
        "caps_stats": {
            "min": float(caps.min()),
            "max": float(caps.max()),
            "mean": float(caps.mean()),
        },
        "noise_probe": {
            "no_noise":   means["no_noise"],
            "uniform":    {str(e): means["uniform"][e]    for e in epsilons},
            "channel_WF": {str(e): means["channel_WF"][e] for e in epsilons},
            "lift_WF_over_uniform": {str(e): lift[e] for e in epsilons},
        },
    }
    out_path = Path(__file__).parent / "drive_local_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.axhline(means["no_noise"], color="#555", ls=":", lw=2,
                   label=f"clean / no noise ({means['no_noise']:.3f})")
        ax.plot(epsilons, [means["uniform"][e] for e in epsilons], "s--",
                color="#d62728", lw=2, ms=8, label="uniform")
        ax.plot(epsilons, [means["channel_WF"][e] for e in epsilons], "o-",
                color="#1f77b4", lw=2, ms=8, label="channel-WF")
        ax.set_xscale("log", base=2)
        ax.set_xticks(epsilons)
        ax.set_xticklabels([str(int(e)) for e in epsilons])
        ax.set_xlabel("privacy budget  $\\epsilon$")
        ax.set_ylabel("vessel Dice")
        ax.set_title(
            f"Real DRIVE — local noise probe (K={K_probe}, 96×96, TinyUNet "
            f"{n_params/1e6:.2f}M params)"
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = Path(__file__).parent / "drive_local_tradeoff.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved: {plot_path}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
