"""
Student adapter for WF+thr-released features, NO GT supervision.

Goal: show whether a downstream adaptation step (trained only on (noisy
thr-released bn, clean bn) feature pairs, i.e. pure feature matching) can
push the Dice further toward the clean upper bound.

This is the WORKFLOW.md §0.5.1 option 3 ("drop task loss entirely") made
operational. In the real pipeline this adapter is trained on the public
HRF proxy; here, for the LOCAL demo, we use DRIVE train as an oracle and
say so explicitly. The mechanism question (does adaptation help?) does not
change with that substitution.

Architecture:
    adapter(x) = x + conv(x)   (residual; predicts the correction)

Pipeline at inference:
    image → teacher_encoder → clean_bn → clip+normalise + threshold + noise
          → adapter (trainable, no GT)
          → de-normalise
          → teacher_decoder (frozen)   ← post-processing
          → vessel mask
"""

import json
import sys
from pathlib import Path

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
from drive_wf_threshold import thresholded_wf_sigma
from synthetic_demo import (
    eps_to_rho, uniform_sigma, clip_and_normalise, denormalise,
)


class FeatureAdapter(nn.Module):
    """Residual 3-conv adapter. Predicts a CORRECTION to the noisy thr bn."""
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden,   hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    # ---- teacher (frozen post-train) ----
    print("\n[Setup] Loading DRIVE + training teacher...")
    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3, device=device)
    clean_dice = evaluate_vessel_dice(teacher, val_loader, device)
    importance = compute_importance(teacher, train_loader, device).to(device)
    caps       = collect_caps(teacher, train_loader, device).to(device)
    Cb = importance.shape[0]
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    print(f"  clean Dice: {clean_dice:.4f}")

    # ---- mechanism: keep top-10% with WF on active set ----
    K = 10
    keep_fraction = 0.1
    deltas = torch.full((Cb,), 2.0 / K, device=device)
    rank_order = torch.argsort(importance, descending=True)
    kk = max(1, int(round(keep_fraction * Cb)))
    mask = torch.zeros(Cb, dtype=torch.bool, device=device)
    mask[rank_order[:kk]] = True
    print(f"\nMechanism: WF+thr keep {keep_fraction*100:.0f}%  "
          f"→ {kk}/{Cb} active channels   K={K}")

    eps_list = [8.0, 16.0, 32.0]
    results = {"clean": clean_dice, "eps_list": eps_list}

    for eps in eps_list:
        rho = eps_to_rho(eps)
        sigma = thresholded_wf_sigma(deltas, importance, rho, mask)
        print(f"\n========== ε={eps} ==========")
        print(f"  σ on active set (mean): {sigma[mask].mean().item():.4f}")

        # ---------- BASELINE: probe with no adapter ----------
        baseline_scores = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                e1, e2, e3 = teacher.encode(x)
                torch.manual_seed(42 + int(eps * 100))
                bn_norm = clip_and_normalise(e3, caps)
                B, C, H, W = bn_norm.shape
                noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                bn_thr = bn_norm + noise
                bn_thr[:, ~mask, :, :] = 0
                bn_out = denormalise(bn_thr, caps)
                baseline_scores.append(vessel_dice(teacher.decode(e1, e2, bn_out), y))
        baseline_dice = float(np.mean(baseline_scores))
        print(f"  [no adapter]    Dice = {baseline_dice:.4f}")

        # ---------- TRAIN ADAPTER (pure feature matching) ----------
        adapter = FeatureAdapter(Cb).to(device)
        opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        n_epochs = 40
        adapter.train()
        for ep in range(n_epochs):
            total = 0.0
            for x, _ in train_loader:                       # NOTE: y unused — no GT
                x = x.to(device)
                with torch.no_grad():
                    e1, e2, e3 = teacher.encode(x)
                    bn_clean = clip_and_normalise(e3, caps)          # target (oracle in demo)
                B, C, H, W = bn_clean.shape
                noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                bn_thr = bn_clean + noise
                bn_thr = bn_thr.detach().clone()
                bn_thr[:, ~mask, :, :] = 0

                opt.zero_grad()
                bn_pred = adapter(bn_thr)
                loss = F.mse_loss(bn_pred, bn_clean.detach())
                loss.backward()
                opt.step()
                total += loss.item()
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"    ep {ep+1:3d}/{n_epochs}  feat_loss={total/len(train_loader):.4f}")

        # ---------- EVAL WITH ADAPTER ----------
        adapter.eval()
        adapter_scores = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                e1, e2, e3 = teacher.encode(x)
                torch.manual_seed(42 + int(eps * 100))
                bn_clean = clip_and_normalise(e3, caps)
                B, C, H, W = bn_clean.shape
                noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                bn_thr = bn_clean + noise
                bn_thr[:, ~mask, :, :] = 0
                bn_pred = adapter(bn_thr)
                bn_out = denormalise(bn_pred, caps)
                adapter_scores.append(vessel_dice(teacher.decode(e1, e2, bn_out), y))
        adapter_dice = float(np.mean(adapter_scores))
        lift = adapter_dice - baseline_dice
        print(f"  [with adapter]  Dice = {adapter_dice:.4f}   "
              f"(adapter lift {lift:+.4f},  clean = {clean_dice:.4f})")

        results[f"eps_{eps}"] = {
            "baseline_no_adapter": baseline_dice,
            "with_adapter":        adapter_dice,
            "adapter_lift":        lift,
            "clean_gap_closed":    lift / max(1e-9, clean_dice - baseline_dice),
        }

    out_path = Path(__file__).parent / "drive_student_adapter_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axhline(clean_dice, color="#555", ls=":", lw=2.5,
               label=f"clean (no noise) = {clean_dice:.3f}")
    base = [results[f"eps_{e}"]["baseline_no_adapter"] for e in eps_list]
    adap = [results[f"eps_{e}"]["with_adapter"]        for e in eps_list]
    ax.plot(eps_list, base, "s--", color="#ff7f0e", lw=2,   ms=8,
            label="WF+thr keep 10%  (no adapter, decoder probed directly)")
    ax.plot(eps_list, adap, "o-",  color="#2ca02c", lw=2.5, ms=10,
            label="WF+thr keep 10%  +  adapter  (pure feature matching, no GT)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(eps_list)
    ax.set_xticklabels([str(int(e)) for e in eps_list])
    ax.set_xlabel("privacy budget  $\\epsilon$  (K=10)")
    ax.set_ylabel("vessel Dice")
    ax.set_title(
        "Student adaptation on thr-released features  —  "
        "no GT, pure MSE feature matching\n"
        "(adapter trained on DRIVE train as oracle stand-in for HRF "
        "public-proxy)"
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = Path(__file__).parent / "drive_student_adapter.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
