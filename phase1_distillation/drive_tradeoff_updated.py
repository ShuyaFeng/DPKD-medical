"""
Updated DRIVE tradeoff plot — adds the WF+threshold variants to the original
3-line plot. Overwrites phase1_distillation/drive_local_tradeoff.png.

Mechanisms on the same axes:
  - clean (no noise reference)
  - uniform Gaussian
  - plain channel-WF (= keep 100%)
  - WF + threshold keep 25%   (32 / 128 channels active)
  - WF + threshold keep 10%   (13 / 128 channels active)

Full ε sweep at K=10 to match the original plot exactly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
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


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    # ---- data + teacher ----
    print("\n[1/3] Loading DRIVE + training teacher...")
    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3, device=device)
    clean_dice = evaluate_vessel_dice(teacher, val_loader, device)
    print(f"  clean: {clean_dice:.4f}")

    # ---- importance + caps ----
    print("\n[2/3] Importance + caps...")
    importance = compute_importance(teacher, train_loader, device).to(device)
    caps       = collect_caps(teacher, train_loader, device).to(device)
    Cb = importance.shape[0]
    print(f"  C={Cb}  ratio={(importance.max()/importance.min()).item():.2f}×")

    # ---- full sweep ----
    print("\n[3/3] Sweeping ε across all mechanisms...")
    epsilons       = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    keep_fractions = [1.0, 0.25, 0.1]   # plain WF + 2 threshold variants
    K = 10

    deltas = torch.full((Cb,), 2.0 / K, device=device)
    rank_order = torch.argsort(importance, descending=True)

    masks = {}
    for f in keep_fractions:
        kk = max(1, int(round(f * Cb)))
        m = torch.zeros(Cb, dtype=torch.bool, device=device)
        m[rank_order[:kk]] = True
        masks[f] = m

    sigma_tab = {}
    for eps in epsilons:
        rho = eps_to_rho(eps)
        sigma_tab[eps] = {"uniform": uniform_sigma(deltas, rho)}
        for f in keep_fractions:
            sigma_tab[eps][f] = thresholded_wf_sigma(deltas, importance, rho, masks[f])

    res = {"clean": [], "uniform": {e: [] for e in epsilons}}
    for f in keep_fractions:
        res[f] = {e: [] for e in epsilons}

    teacher.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            e1, e2, e3 = teacher.encode(x)
            res["clean"].append(vessel_dice(teacher.decode(e1, e2, e3), y))

            bn_norm = clip_and_normalise(e3, caps)
            B, C, H, W = bn_norm.shape

            for eps in epsilons:
                # uniform
                torch.manual_seed(42 + int(eps * 100))
                sigma = sigma_tab[eps]["uniform"]
                noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                bn_noisy = denormalise(bn_norm + noise, caps)
                res["uniform"][eps].append(
                    vessel_dice(teacher.decode(e1, e2, bn_noisy), y)
                )

                # plain WF (f=1.0) and threshold variants
                for f in keep_fractions:
                    torch.manual_seed(42 + int(eps * 100))
                    sigma = sigma_tab[eps][f]
                    noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                    bn_thr = bn_norm + noise
                    inactive = ~masks[f]
                    if inactive.any():
                        bn_thr[:, inactive, :, :] = 0
                    bn_noisy = denormalise(bn_thr, caps)
                    res[f][eps].append(vessel_dice(teacher.decode(e1, e2, bn_noisy), y))

    # ---- aggregate ----
    m_clean = float(np.mean(res["clean"]))
    m_uni   = {e: float(np.mean(res["uniform"][e])) for e in epsilons}
    m_f     = {f: {e: float(np.mean(res[f][e])) for e in epsilons}
               for f in keep_fractions}

    # ---- print ----
    print(f"\nCLEAN: {m_clean:.4f}")
    hdr = f"{'mech':<24s}" + "".join(f"  ε={e:>4.1f}" for e in epsilons)
    print(hdr)
    print("-" * len(hdr))
    print(f"{'uniform':<24s}" + "".join(f"  {m_uni[e]:>7.4f}" for e in epsilons))
    for f in keep_fractions:
        lbl = "plain channel-WF" if f == 1.0 else f"WF+thr keep {int(f*100):>3d}%"
        print(f"{lbl:<24s}" + "".join(f"  {m_f[f][e]:>7.4f}" for e in epsilons))

    # ---- save JSON (companion to figure) ----
    out_json = {
        "clean": m_clean,
        "K": K,
        "C": Cb,
        "epsilons": epsilons,
        "uniform":      {str(e): m_uni[e]    for e in epsilons},
        "plain_WF":     {str(e): m_f[1.0][e] for e in epsilons},
        "WF_thr_25":    {str(e): m_f[0.25][e] for e in epsilons},
        "WF_thr_10":    {str(e): m_f[0.1][e]  for e in epsilons},
    }
    json_path = Path(__file__).parent / "drive_tradeoff_updated_results.json"
    json_path.write_text(json.dumps(out_json, indent=2))
    print(f"\nSaved results: {json_path}")

    # ---- plot (overwrite drive_local_tradeoff.png) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhline(m_clean, color="#555", ls=":", lw=2.5,
               label=f"clean / no noise = {m_clean:.3f}")
    ax.plot(epsilons, [m_uni[e] for e in epsilons], "s--",
            color="#d62728", lw=2, ms=8, label="uniform")

    palette = {1.0: "#1f77b4", 0.25: "#9467bd", 0.1: "#ff7f0e"}
    for f in keep_fractions:
        lbl = "plain channel-WF  (keep 100%)" if f == 1.0 \
              else f"WF+thr keep {int(f*100):>2d}%  ({int(round(f*Cb))}/{Cb} ch.)"
        ax.plot(epsilons, [m_f[f][e] for e in epsilons],
                "o-", color=palette[f], lw=2.2, ms=8, label=lbl)

    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget  $\\epsilon$")
    ax.set_ylabel("vessel Dice")
    ax.set_title(
        f"Real DRIVE — noise probe (K={K}, 96×96, TinyUNet 0.47M params)\n"
        f"plain WF ≈ uniform; WF+threshold pulls toward clean as more channels are dropped"
    )
    ax.legend(loc="center right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_png = Path(__file__).parent / "drive_local_tradeoff.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved (overwrote): {out_png}")


if __name__ == "__main__":
    main()
