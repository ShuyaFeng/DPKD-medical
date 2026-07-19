# -*- coding: utf-8 -*-
"""
Diagnostic: find the correct CLIP_IMP for importance privatization.

CLIP_IMP must be a valid upper bound on the L2 norm of any single
sample's gradient-energy vector. This script measures exactly that
across all three datasets and reports:

  1. Per-sample gradient-energy L2 norm distribution (min/mean/p90/p99/max)
  2. What sigma_imp would be under ALPHA_IMP=0.1 vs 0.45
     at multiple CLIP_IMP values and epsilon values
  3. How large the noise is relative to the mean importance score

Run on Cheaha:
    cd /home/ab36/DPKD-medical/phase1_distillation
    conda activate mmseg-cu124-240
    python diagnose_clip_imp.py
"""

import math
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from drive_local_demo import task_loss
from drive_pate_poc import train_K_teachers, partition_dataset
from synthetic_demo import eps_to_rho

EPSILONS   = [1.0, 2.0, 4.0, 8.0]
K          = 3
N_EPOCHS   = 10   # short — just for scale diagnosis, not final results
BATCH_SIZE = 8
IMG_SIZE   = 96


def per_sample_grad_energy(teacher, dataset, device):
    """
    For each image in dataset, compute the gradient-energy vector
    (one scalar per bottleneck channel) and return its L2 norm.
    Returns a 1-D tensor of length len(dataset).
    """
    teacher.eval()
    norms = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        x.requires_grad_(True)
        teacher.zero_grad(set_to_none=True)
        e1, e2, e3 = teacher.encode(x)
        e3.retain_grad()
        loss = task_loss(teacher.decode(e1, e2, e3), y)
        loss.backward()
        # per-channel mean squared gradient energy for this one image
        g = e3.grad.detach().pow(2).mean(dim=(0, 2, 3))  # shape [C]
        norms.append(g.norm().item())
    return torch.tensor(norms)


def diagnose_dataset(name, train_ds, in_ch, device):
    print("\n" + "=" * 70)
    print("Dataset: {}  N={}  in_ch={}".format(name.upper(), len(train_ds), in_ch))
    print("=" * 70)

    # Train K teachers briefly (quick, just for scale)
    teachers, _ = train_K_teachers(
        train_ds, K, device, n_epochs=N_EPOCHS, in_ch=in_ch
    )

    # Collect per-sample L2 norms from each teacher on its own partition
    partitions = partition_dataset(len(train_ds), K)
    all_norms = []
    for k, (teacher, idxs) in enumerate(zip(teachers, partitions)):
        subset = Subset(train_ds, idxs)
        norms_k = per_sample_grad_energy(teacher, subset, device)
        all_norms.append(norms_k)
        print("  Teacher {}: n={} | norm min={:.4e} mean={:.4e} "
              "p90={:.4e} max={:.4e}".format(
                  k + 1, len(idxs),
                  norms_k.min().item(),
                  norms_k.mean().item(),
                  norms_k.quantile(0.90).item(),
                  norms_k.max().item()))

    combined = torch.cat(all_norms)
    p99 = combined.quantile(0.99).item()
    pmax = combined.max().item()

    print("\n  Combined across all teachers:")
    print("    min  = {:.4e}".format(combined.min().item()))
    print("    mean = {:.4e}".format(combined.mean().item()))
    print("    p90  = {:.4e}".format(combined.quantile(0.90).item()))
    print("    p99  = {:.4e}".format(p99))
    print("    max  = {:.4e}".format(pmax))

    # Aggregate importance (what CANAL actually uses)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    from drive_pate_pruning_joint import shared_importance
    agg_imp = shared_importance(teachers, train_loader, device)
    print("\n  Aggregate importance (what CANAL uses):")
    print("    min  = {:.4e}".format(agg_imp.min().item()))
    print("    mean = {:.4e}".format(agg_imp.mean().item()))
    print("    max  = {:.4e}".format(agg_imp.max().item()))

    N = len(train_ds)

    # Show sigma_imp for candidate CLIP_IMP values and both ALPHA_IMP settings
    print("\n  --- Noise analysis ---")
    print("  (sigma_imp vs aggregate importance mean = {:.4e})".format(
        agg_imp.mean().item()))
    print("")

    candidates = {
        "1e-8 (current)": 1e-8,
        "p99 of per-sample norms": p99,
        "max of per-sample norms": pmax,
    }

    for label, clip in candidates.items():
        sens = 2.0 * clip / N
        print("  CLIP_IMP = {} ({:.4e})  sens={:.4e}".format(label, clip, sens))
        for alpha_imp in [0.10, 0.45]:
            for eps in EPSILONS:
                rho = eps_to_rho(eps)
                rho_imp = alpha_imp * rho
                sigma_imp = sens / math.sqrt(2.0 * rho_imp)
                ratio = sigma_imp / agg_imp.mean().item() if agg_imp.mean().item() > 0 else float('inf')
                print("    alpha={:.2f} eps={:4.1f}: sigma_imp={:.4e}  "
                      "noise/imp_mean={:.2f}x".format(
                          alpha_imp, eps, sigma_imp, ratio))
        print("")


def main():
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device={}".format(dev))

    datasets = []

    try:
        from isic_dataset import ISICDataset
        datasets.append(("isic", ISICDataset("train", IMG_SIZE), 3))
    except Exception as e:
        print("ISIC not available: {}".format(e))

    try:
        from kvasir_dataset import KvasirDataset
        datasets.append(("kvasir", KvasirDataset("train", IMG_SIZE), 3))
    except Exception as e:
        print("Kvasir not available: {}".format(e))

    try:
        from busi_dataset import BUSIDataset
        datasets.append(("busi", BUSIDataset("train", IMG_SIZE), 1))
    except Exception as e:
        print("BUSI not available: {}".format(e))

    for name, ds, in_ch in datasets:
        diagnose_dataset(name, ds, in_ch, dev)

    print("\n\nDone. Use the p99 or max of per-sample norms as CLIP_IMP.")
    print("If noise/imp_mean >> 1, the importance scores will be destroyed at that CLIP_IMP.")
    print("If noise/imp_mean << 1, the importance scores survive and privatization is valid.")


if __name__ == "__main__":
    main()
