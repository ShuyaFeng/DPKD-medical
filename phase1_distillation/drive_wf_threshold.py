"""
WF + threshold variant probe on real DRIVE.

For each keep-fraction f, the top f×C channels (by gradient importance) are the
"active set". The full ρ budget is allocated via WF only across the active set;
non-active channels are NOT released (σ→∞), and at probe time they're replaced
with 0 in the normalised feature space (a data-independent constant).

Compares:    clean / uniform / plain WF / WF+thr at keep ∈ {0.75, 0.5, 0.25, 0.1}
at ε ∈ {4, 8, 16}.

Important caveat: the decoder is NOT retrained for the threshold scheme. §13.2's
clean win for WF+thr was against a closed-form Bayes-LDA classifier that
optimally ignores destroyed channels. A trained CNN decoder cannot do this
without adaptation. This probe tells us how badly that mismatch hurts in the
feature-release setting.
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
from synthetic_demo import (
    eps_to_rho, waterfilling_sigma, uniform_sigma,
    clip_and_normalise, denormalise,
)


def thresholded_wf_sigma(deltas, importance, rho, active_mask):
    """WF allocation on the active subset; σ=0 for inactive (they will be zeroed)."""
    sigma = torch.zeros_like(deltas)
    if active_mask.any():
        sigma[active_mask] = waterfilling_sigma(
            deltas[active_mask], importance[active_mask], rho,
        )
    return sigma


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
    print(f"  C={Cb}  importance ratio={(importance.max()/importance.min()).item():.2f}×")

    # ---- threshold probe ----
    print("\n[3/3] Probe: clean / uniform / plain-WF / WF+thr at multiple keep-fractions...")
    epsilons = [4.0, 8.0, 16.0]
    keep_fractions = [1.0, 0.75, 0.5, 0.25, 0.1]
    K = 10

    deltas = torch.full((Cb,), 2.0 / K, device=device)
    rank_order = torch.argsort(importance, descending=True)

    masks = {}
    for f in keep_fractions:
        keep_k = max(1, int(round(f * Cb)))
        mask = torch.zeros(Cb, dtype=torch.bool, device=device)
        mask[rank_order[:keep_k]] = True
        masks[f] = mask

    sigma_table = {}
    for eps in epsilons:
        rho = eps_to_rho(eps)
        sigma_table[eps] = {"uniform": uniform_sigma(deltas, rho)}
        for f in keep_fractions:
            sigma_table[eps][f] = thresholded_wf_sigma(
                deltas, importance, rho, masks[f],
            )

    results = {"clean": [], "uniform": {e: [] for e in epsilons}}
    for f in keep_fractions:
        results[f] = {e: [] for e in epsilons}

    teacher.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            e1, e2, e3 = teacher.encode(x)
            results["clean"].append(vessel_dice(teacher.decode(e1, e2, e3), y))

            bn_norm = clip_and_normalise(e3, caps)
            B, C, H, W = bn_norm.shape

            for eps in epsilons:
                # uniform (no threshold)
                torch.manual_seed(42 + int(eps * 100))
                sigma = sigma_table[eps]["uniform"]
                noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                bn_noisy = denormalise(bn_norm + noise, caps)
                results["uniform"][eps].append(
                    vessel_dice(teacher.decode(e1, e2, bn_noisy), y)
                )

                # plain WF (f=1.0) and WF+thr variants
                for f in keep_fractions:
                    torch.manual_seed(42 + int(eps * 100))
                    sigma = sigma_table[eps][f]
                    noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                    bn_thr = bn_norm + noise
                    inactive = ~masks[f]
                    if inactive.any():
                        bn_thr[:, inactive, :, :] = 0.0      # zero out the unreleased channels
                    bn_noisy = denormalise(bn_thr, caps)
                    results[f][eps].append(
                        vessel_dice(teacher.decode(e1, e2, bn_noisy), y)
                    )

    means = {"clean": float(np.mean(results["clean"]))}
    means["uniform"] = {e: float(np.mean(results["uniform"][e])) for e in epsilons}
    for f in keep_fractions:
        means[f] = {e: float(np.mean(results[f][e])) for e in epsilons}

    # ---- print ----
    print()
    print(f"CLEAN vessel Dice (upper bound):  {means['clean']:.4f}")
    print()
    hdr = f"{'mechanism':<24s}" + "".join(f"  ε={e:>4.1f}" for e in epsilons)
    print(hdr)
    print("-" * len(hdr))
    print(f"{'uniform':<24s}" + "".join(f"  {means['uniform'][e]:>7.4f}" for e in epsilons))
    for f in keep_fractions:
        label = "plain WF (keep 100%)" if f == 1.0 else f"WF+thr keep {int(f*100):>3d}%"
        print(f"{label:<24s}" + "".join(f"  {means[f][e]:>7.4f}" for e in epsilons))

    # σ inspection
    print("\nσ on the active set at ε=8 (smaller = cleaner active channels):")
    for f in keep_fractions:
        sigma = sigma_table[8.0][f]
        active = masks[f]
        n_active = int(active.sum())
        mean_sigma_active = sigma[active].mean().item()
        label = "plain WF" if f == 1.0 else f"WF+thr keep {int(f*100)}%"
        print(f"  {label:<22s}  active={n_active:>4d} / {Cb}  mean σ_active={mean_sigma_active:.4f}")

    # ---- save + plot ----
    out = {
        "clean": means["clean"],
        "uniform":     {str(e): means["uniform"][e]     for e in epsilons},
        "plain_WF":    {str(e): means[1.0][e]           for e in epsilons},
        "WF_thr_75":   {str(e): means[0.75][e]          for e in epsilons},
        "WF_thr_50":   {str(e): means[0.5][e]           for e in epsilons},
        "WF_thr_25":   {str(e): means[0.25][e]          for e in epsilons},
        "WF_thr_10":   {str(e): means[0.1][e]           for e in epsilons},
        "epsilons":      epsilons,
        "keep_fractions": keep_fractions,
        "K":              K,
        "C":              Cb,
    }
    out_path = Path(__file__).parent / "drive_wf_threshold_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1.0: "#1f77b4", 0.75: "#ff7f0e", 0.5: "#2ca02c",
              0.25: "#9467bd", 0.1: "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axhline(means["clean"], color="#555", ls=":", lw=2.5,
               label=f"clean (no noise) = {means['clean']:.3f}")
    ax.plot(epsilons, [means["uniform"][e] for e in epsilons],
            "s--", color="black", lw=2, ms=7, label="uniform (no thr)")
    for f in keep_fractions:
        label = "plain WF (keep 100%)" if f == 1.0 else f"WF+thr keep {int(f*100):>2d}%"
        ys = [means[f][e] for e in epsilons]
        ax.plot(epsilons, ys, "o-", color=colors[f], lw=2, ms=8, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget  $\\epsilon$")
    ax.set_ylabel("vessel Dice")
    ax.set_title(
        "DRIVE — WF + threshold variant probe\n"
        "(decoder NOT retrained; inactive channels set to 0 in normalised space)"
    )
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = Path(__file__).parent / "drive_wf_threshold.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
