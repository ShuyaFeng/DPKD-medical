"""
LOCAL student distillation: WF vs uniform.

Earlier experiments were noise PROBES (teacher decoding its own noised
bottleneck). The paper's actual claim is about a STUDENT trained against the
sample-once noisy targets. We test that here.

Pipeline (mirrors phase1_gkd_distill_v2.py minus mmseg):
  1. Frozen teacher (TinyUNet base=32) trained on DRIVE train.
  2. Per (ε, noise_type) condition:
     a. Sample-once precompute: cache one noisy bottleneck per training image.
     b. Train a SMALLER student (TinyUNet base=16) for N epochs:
        - student encoder → 64-channel bottleneck
        - 1×1 adapter → 128-channel projection
        - feature loss   = MSE(adapter(student_bn), cached_noisy_teacher_bn)
        - task loss      = CE + 3·Dice on student logits vs GT
        - total          = task + λ·feature
     c. Eval student vessel Dice on val.
  3. Compare uniform vs WF at each ε.

We also keep a "student baseline" (task loss only, no distillation, no noise)
as the reference point where the paper's WF vs uniform difference would be
meaningful — student baseline is the floor we're trying to climb above
using the teacher's signal.
"""

import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher, evaluate_vessel_dice,
    compute_importance, collect_caps, vessel_dice,
)
from synthetic_demo import (
    eps_to_rho, waterfilling_sigma, uniform_sigma,
    clip_and_normalise, denormalise, task_loss,
)


class Adapter(nn.Module):
    """1×1 conv: student_ch → teacher_ch."""
    def __init__(self, s_ch: int, t_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(s_ch, t_ch, 1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out",
                                nonlinearity="relu")

    def forward(self, x):
        return self.conv(x)


@torch.no_grad()
def precompute_cache(teacher, train_ds, caps, sigma, device, seed: int):
    """
    Sample-once precompute. Returns a list keyed by dataset index.
    cache[i] = (C, H_b, W_b) noisy teacher bottleneck for train_ds[i].
    """
    teacher.eval()
    torch.manual_seed(seed)
    cache: List[torch.Tensor] = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    for x, _ in loader:
        x = x.to(device)
        _, _, e3 = teacher.encode(x)
        bn_norm = clip_and_normalise(e3, caps)
        B, C, H, W = bn_norm.shape
        noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
        bn_noisy = denormalise(bn_norm + noise, caps)
        for b in range(B):
            cache.append(bn_noisy[b].detach().cpu())
    return cache


def train_student_distill(train_ds, val_loader, cache, device,
                          student_base=16, teacher_base=32,
                          n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=2):
    """Train student from scratch on (cached noisy features + GT labels)."""
    torch.manual_seed(seed)
    student = TinyUNet(in_ch=3, num_classes=2, base=student_base).to(device)
    adapter = Adapter(student_base * 4, teacher_base * 4).to(device)
    opt = torch.optim.Adam(
        list(student.parameters()) + list(adapter.parameters()), lr=lr,
    )

    N = len(train_ds)
    batch_size = 4
    best_dice = 0.0
    best_ep = 0
    history = []

    for ep in range(n_epochs):
        student.train(); adapter.train()
        perm = torch.randperm(N)
        tot_task, tot_feat = 0.0, 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idxs = perm[i:i+batch_size].tolist()
            xs = torch.stack([train_ds[idx][0] for idx in idxs]).to(device)
            ys = torch.stack([train_ds[idx][1] for idx in idxs]).to(device)
            t_targets = torch.stack([cache[idx] for idx in idxs]).to(device)

            opt.zero_grad()
            e1, e2, e3 = student.encode(xs)
            s_proj = adapter(e3)
            f_loss = F.mse_loss(s_proj, t_targets)
            logits = student.decode(e1, e2, e3)
            t_loss = task_loss(logits, ys)
            loss = t_loss + lambda_feat * f_loss
            loss.backward()
            opt.step()
            tot_task += t_loss.item(); tot_feat += f_loss.item()
            n_batches += 1

        val_dice = evaluate_vessel_dice(student, val_loader, device)
        if val_dice > best_dice:
            best_dice = val_dice
            best_ep = ep + 1
        history.append({"ep": ep + 1, "task": tot_task / n_batches,
                        "feat": tot_feat / n_batches, "val_dice": val_dice})

        if (ep + 1) % 5 == 0 or ep == 0 or ep == n_epochs - 1:
            print(f"    ep {ep+1:3d}/{n_epochs}  task={tot_task/n_batches:.4f}  "
                  f"feat={tot_feat/n_batches:.4f}  val={val_dice:.4f}  "
                  f"best_so_far={best_dice:.4f} (ep {best_ep})")

    return best_dice, history


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    # -- data --
    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    # -- teacher (frozen post-train) --
    print("\n[1/3] Training teacher (TinyUNet base=32)...")
    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    teacher_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    train_teacher(teacher, teacher_loader, n_epochs=60, lr=1e-3, device=device)
    clean_dice = evaluate_vessel_dice(teacher, val_loader, device)
    importance = compute_importance(teacher, teacher_loader, device).to(device)
    caps       = collect_caps(teacher, teacher_loader, device).to(device)
    Cb_T = importance.shape[0]
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    print(f"  teacher clean Dice = {clean_dice:.4f}")
    print(f"  imp ratio = {(importance.max()/importance.min()).item():.2f}×, "
          f"caps mean={caps.mean().item():.2f}")

    # -- student baseline (NO distillation, NO noisy targets) --
    print("\n[2/3] Student baseline (task loss only, no distillation)...")
    torch.manual_seed(1)
    student_baseline = TinyUNet(in_ch=3, num_classes=2, base=16).to(device)
    s_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    train_teacher(student_baseline, s_loader, n_epochs=60, lr=1e-3, device=device)
    baseline_dice = evaluate_vessel_dice(student_baseline, val_loader, device)
    print(f"  student baseline Dice = {baseline_dice:.4f}")

    # -- distillation sweep --
    print("\n[3/3] Student distillation: WF vs uniform at each ε...")
    epsilons = [2.0, 8.0, 16.0]
    K = 10
    deltas = torch.full((Cb_T,), 2.0 / K, device=device)

    results = {
        "clean_teacher": clean_dice,
        "student_baseline": baseline_dice,
        "K": K, "lambda_feat": 0.4, "n_epochs": 40,
        "sweep": {},
    }

    for eps in epsilons:
        rho = eps_to_rho(eps)
        sigma_uni = uniform_sigma(deltas, rho)
        sigma_wf  = waterfilling_sigma(deltas, importance, rho)

        # σ inspection
        top10 = torch.argsort(importance, descending=True)[:10]
        bot10 = torch.argsort(importance, descending=False)[:10]
        print(f"\n--- ε={eps}  ρ={rho:.4f} ---")
        print(f"   σ uniform = {sigma_uni[0].item():.4f}")
        print(f"   σ WF top10 = {sigma_wf[top10].mean().item():.4f}, "
              f"bot10 = {sigma_wf[bot10].mean().item():.4f}, "
              f"ratio = {(sigma_wf[bot10].mean()/sigma_wf[top10].mean()).item():.3f}×")

        results["sweep"][eps] = {}
        for mech, sigma in [("uniform", sigma_uni), ("channel_WF", sigma_wf)]:
            print(f"\n  >>> mechanism = {mech}")
            cache = precompute_cache(teacher, train_ds, caps, sigma, device,
                                     seed=42 + int(eps * 10))
            t0 = time.time()
            best, hist = train_student_distill(
                train_ds, val_loader, cache, device,
                student_base=16, teacher_base=32,
                n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=2 + int(eps),
            )
            print(f"  → best Dice = {best:.4f}  (time {time.time()-t0:.1f}s)")
            results["sweep"][eps][mech] = {
                "best_dice": best,
                "history": hist,
                "sigma_top10": sigma_wf[top10].mean().item() if mech == "channel_WF" else sigma_uni[0].item(),
                "sigma_bot10": sigma_wf[bot10].mean().item() if mech == "channel_WF" else sigma_uni[0].item(),
            }

    # -- summary --
    print("\n" + "=" * 84)
    print(f"clean teacher           {clean_dice:.4f}")
    print(f"student baseline (no DP){baseline_dice:.4f}")
    print(f"{'ε':>5}  {'uniform':>10}  {'channel_WF':>11}  {'WF − uni':>10}  "
          f"{'WF − baseline':>14}")
    print("-" * 84)
    for eps in epsilons:
        u = results["sweep"][eps]["uniform"]["best_dice"]
        w = results["sweep"][eps]["channel_WF"]["best_dice"]
        print(f"{eps:>5.1f}  {u:>10.4f}  {w:>11.4f}  {w-u:>+10.4f}  "
              f"{w-baseline_dice:>+14.4f}")
    print("=" * 84)

    # -- save --
    out_path = Path(__file__).parent / "drive_student_distill_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out_path}")

    # -- plot --
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhline(clean_dice, color="#555", ls=":", lw=2,
               label=f"teacher clean = {clean_dice:.3f}")
    ax.axhline(baseline_dice, color="#ff7f0e", ls="--", lw=2,
               label=f"student baseline (no distill) = {baseline_dice:.3f}")
    uni = [results["sweep"][e]["uniform"]["best_dice"]   for e in epsilons]
    wf  = [results["sweep"][e]["channel_WF"]["best_dice"] for e in epsilons]
    ax.plot(epsilons, uni, "s--", color="#d62728", lw=2, ms=9, label="uniform (DP distill)")
    ax.plot(epsilons, wf,  "o-",  color="#1f77b4", lw=2, ms=9, label="channel-WF (DP distill)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget  ε")
    ax.set_ylabel("vessel Dice")
    ax.set_title(f"DRIVE — ACTUAL student distillation (TinyUNet teacher 32 → student 16)\n"
                 f"task loss + λ·feature loss, K={K}, λ=0.4, 40 epochs")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = Path(__file__).parent / "drive_student_distill.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
