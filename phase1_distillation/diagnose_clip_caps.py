# -*- coding: utf-8 -*-
"""
Diagnostic: measure per-sample activation norm distribution for caps privatization.

For each dataset, reports the distribution of per-sample max per-channel
activation norms (the quantity CLIP_CAPS must bound) and shows what
sigma_caps would be at different candidate values and epsilon values.

Run on Cheaha:
    cd /home/ab36/DPKD-medical/phase1_distillation
    conda activate mmseg-cu124-240
    python diagnose_clip_caps.py | tee clip_caps_diagnostic.txt
"""

import math
import sys
import torch
from pathlib import Path
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).parent))
from drive_pate_poc import train_K_teachers, partition_dataset
from synthetic_demo import eps_to_rho

EPSILONS  = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
K         = 3
N_EPOCHS  = 10
IMG_SIZE  = 96


def per_sample_max_cap_norm(teacher, dataset, device):
    """
    For each image, compute the max per-channel L2 activation norm at the
    bottleneck. This bounds how much one patient can shift the per-channel
    clipping caps (90th-percentile activation norms used for normalization).
    Returns a 1-D tensor of length len(dataset).
    """
    teacher.eval()
    maxnorms = []
    for i in range(len(dataset)):
        x, _ = dataset[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, e3 = teacher.encode(x)
        per_ch = e3[0].flatten(1).norm(dim=1)   # [C] — L2 norm per channel
        maxnorms.append(per_ch.max().item())     # worst channel for this patient
    return torch.tensor(maxnorms)


def diagnose_dataset(name, train_ds, in_ch, device):
    print("\n" + "=" * 70)
    print("Dataset: {}  N={}  in_ch={}".format(name.upper(), len(train_ds), in_ch))
    print("=" * 70)

    teachers, _ = train_K_teachers(
        train_ds, K, device, n_epochs=N_EPOCHS, in_ch=in_ch
    )
    partitions = partition_dataset(len(train_ds), K)
    all_norms = []
    for k, (teacher, idxs) in enumerate(zip(teachers, partitions)):
        subset = Subset(train_ds, idxs)
        norms_k = per_sample_max_cap_norm(teacher, subset, device)
        all_norms.append(norms_k)
        print("  Teacher {}: n={} | min={:.4e}  mean={:.4e}  "
              "p90={:.4e}  max={:.4e}".format(
                  k + 1, len(idxs),
                  norms_k.min().item(), norms_k.mean().item(),
                  norms_k.quantile(0.90).item(), norms_k.max().item()))

    combined = torch.cat(all_norms)
    pmean = combined.mean().item()
    p99   = combined.quantile(0.99).item()
    pmax  = combined.max().item()

    print("\n  Combined across all teachers:")
    print("    min  = {:.4e}".format(combined.min().item()))
    print("    mean = {:.4e}".format(pmean))
    print("    p90  = {:.4e}".format(combined.quantile(0.90).item()))
    print("    p99  = {:.4e}".format(p99))
    print("    max  = {:.4e}".format(pmax))

    N = len(train_ds)
    print("\n  --- Noise analysis (f_caps=0.10) ---")
    for label, clip in [("mean / avg", pmean), ("p99", p99), ("max", pmax)]:
        sens = 2.0 * clip / N
        print("\n  CLIP_CAPS = {} ({:.4e})  sens={:.4e}".format(label, clip, sens))
        for eps in EPSILONS:
            rho      = eps_to_rho(eps)
            rho_caps = 0.10 * rho
            sigma    = sens / math.sqrt(2.0 * rho_caps)
            print("    eps={:5.1f}: sigma_caps={:.4e}".format(eps, sigma))


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

    print("\n\nDone.")
    print("Use mean (avg) as CLIP_CAPS for the average-based approach.")
    print("Use p99 as CLIP_CAPS for a worst-case valid bound.")


if __name__ == "__main__":
    main()
