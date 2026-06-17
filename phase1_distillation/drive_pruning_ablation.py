"""
Comprehensive pruning ablation — isolate where every Dice gain comes from.

Single teacher (cleanest, no channel-alignment confound). Decomposes the
channel-pruning result into its constituent effects so the paper can
attribute each gain honestly.

Conditions
----------
No-noise references (constant across ε):
  student_only   : base=16 student trained on GT only, NO teacher at all.
                   The FLOOR. If noisy methods don't beat this, the teacher
                   contributes nothing (answers the alpha=1 red flag).
  clean_full     : distill from clean (no-noise) teacher, full channels.
                   The CEILING of feature distillation.
  clean_imp10    : clean teacher, importance top-10% only. If this DROPS
                   vs clean_full, pruning's benefit is purely denoising
                   (not information) — exactly what we expect.

Noisy (per ε ∈ {2,8,16}):
  noisy_full     : full channels + noise            (DP baseline)
  imp_keep10     : importance top-10%  + noise
  rand_keep10    : RANDOM 10%          + noise       (isolates: does picking
                                                      the RIGHT channels matter,
                                                      or is it just fewer dims?)
  imp_keep02     : importance top-2%   + noise
  rand_keep02    : RANDOM 2%           + noise

Key contrasts
-------------
  student_only vs noisy methods   → is the teacher useful at all?
  clean_full vs clean_imp10       → pruning hurts when there's no noise (info loss)
  imp_keepX vs rand_keepX         → does importance selection beat random?
  keep10 vs keep02                → how far does aggressive pruning go?

GRID: 3 no-noise + 5 noisy×3ε = 18 conditions × 5 seeds = 90 trainings.

OUTPUT: drive_pruning_ablation_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher, compute_importance,
    collect_caps, evaluate_vessel_dice, evaluate_metrics, summarize_metrics,
)
from synthetic_demo import (
    eps_to_rho, uniform_sigma, clip_and_normalise, denormalise,
)
from drive_student_distill import train_student_distill


def masked_uniform_sigma(deltas, rho, active_mask):
    sigma = torch.zeros_like(deltas)
    if active_mask.any():
        sigma[active_mask] = uniform_sigma(deltas[active_mask], rho)
    return sigma


@torch.no_grad()
def precompute(teacher, train_ds, caps, sigma, active_mask, device, seed):
    """sigma=None → clean (no noise). active_mask=None → full channels."""
    teacher.eval()
    torch.manual_seed(seed)
    cache = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    C = caps.shape[0]
    for x, _ in loader:
        x = x.to(device)
        _, _, e3 = teacher.encode(x)
        bn = clip_and_normalise(e3, caps)
        if active_mask is not None:
            bn = bn * active_mask.view(1, C, 1, 1).float()
        if sigma is not None:
            B, Cc, H, W = bn.shape
            bn = bn + torch.randn(B, Cc, H, W, device=device) * sigma.view(1, Cc, 1, 1)
        bn = denormalise(bn, caps)
        for b in range(bn.shape[0]):
            cache.append(bn[b].detach().cpu())
    return cache


def run_students(train_ds, val_loader, cache, device, seeds, base_T):
    """Return a list of per-seed best-metric dicts."""
    mlist = []
    for s in seeds:
        _, bm = train_student_distill(
            train_ds, val_loader, cache, device,
            student_base=16, teacher_base=base_T,
            n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s)
        mlist.append(bm)
    return mlist


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

    base_T = 32
    Cb_T = base_T * 4
    K = 1
    deltas = torch.full((Cb_T,), 2.0 / K, device=device)
    epsilons = [2.0, 8.0, 16.0]
    seeds = [100, 200, 300, 400, 500]

    print("\n[1/3] Train single teacher...")
    teacher = TinyUNet(in_ch=3, num_classes=2, base=base_T).to(device)
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3, device=device)
    importance = compute_importance(teacher, train_loader, device).to(device)
    caps = collect_caps(teacher, train_loader, device).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    rank = torch.argsort(importance, descending=True)

    def imp_mask(frac):
        m = torch.zeros(Cb_T, dtype=torch.bool, device=device)
        m[rank[:max(1, int(round(frac*Cb_T)))]] = True
        return m

    def rand_mask(frac, seed):
        g = torch.Generator().manual_seed(seed)  # CPU generator (mps-safe)
        m = torch.zeros(Cb_T, dtype=torch.bool, device=device)
        idx = torch.randperm(Cb_T, generator=g)[:max(1, int(round(frac*Cb_T)))].to(device)
        m[idx] = True
        return m

    results = {"C": Cb_T, "K": K, "epsilons": epsilons, "seeds": seeds,
               "no_noise": {}, "noisy": {}}

    # --- no-noise references ---
    print("\n[2/3] No-noise references...")
    print("  student_only (pure GT, NO teacher)...")
    so = []
    for s in seeds:
        torch.manual_seed(s)
        stu = TinyUNet(in_ch=3, num_classes=2, base=16).to(device)
        train_teacher(stu, DataLoader(train_ds, batch_size=4, shuffle=True),
                      n_epochs=40, lr=1e-3, device=device)
        so.append(evaluate_metrics(stu, val_loader, device))
    results["no_noise"]["student_only"] = summarize_metrics(so)
    print(f"    student_only vessel_dice = {summarize_metrics(so)['vessel_dice']['mean']:.4f}")

    for tag, mask in [("clean_full", None), ("clean_imp10", imp_mask(0.10))]:
        cache = precompute(teacher, train_ds, caps, None, mask, device, seed=7)
        ml = run_students(train_ds, val_loader, cache, device, seeds, base_T)
        results["no_noise"][tag] = summarize_metrics(ml)
        print(f"    {tag} vessel_dice = {summarize_metrics(ml)['vessel_dice']['mean']:.4f}")

    # --- noisy conditions ---
    print("\n[3/3] Noisy conditions...")
    for eps in epsilons:
        rho = eps_to_rho(eps)
        results["noisy"][str(eps)] = {}
        print(f"\n  ε={eps}  ρ={rho:.4f}")
        conds = [
            ("noisy_full",  None),
            ("imp_keep10",  imp_mask(0.10)),
            ("rand_keep10", rand_mask(0.10, 11)),
            ("imp_keep02",  imp_mask(0.02)),
            ("rand_keep02", rand_mask(0.02, 22)),
        ]
        for tag, mask in conds:
            sigma = masked_uniform_sigma(deltas, rho, mask) if mask is not None \
                    else uniform_sigma(deltas, rho)
            cache = precompute(teacher, train_ds, caps, sigma, mask, device,
                               seed=42 + int(eps*10) + hash(tag) % 100)
            ml = run_students(train_ds, val_loader, cache, device, seeds, base_T)
            sm = summarize_metrics(ml)
            results["noisy"][str(eps)][tag] = sm
            print(f"    {tag:12s} vd={sm['vessel_dice']['mean']:.4f} "
                  f"mdice={sm['mdice']['mean']:.4f} miou={sm['miou']['mean']:.4f}")

    out = Path(__file__).parent / "drive_pruning_ablation_results.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out}")
    print("\nREAD THE TABLE:")
    print("  student_only is the floor — every noisy method must beat it,")
    print("  else the teacher is useless. imp_keepX vs rand_keepX tells you")
    print("  whether IMPORTANCE selection matters or it's just dimensionality.")


if __name__ == "__main__":
    main()
