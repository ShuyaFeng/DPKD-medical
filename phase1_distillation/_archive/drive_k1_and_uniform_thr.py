"""
Two ablations in one run, both probing the teacher (decoder NOT retrained):

(E1) K=1 verification
    Re-run WF+thr at K=1 (sample-once accounting from WORKFLOW §6.2 step 4).
    K=1 means Δ_c = 2 (vs Δ_c = 0.2 at K=10), so all σ are 10× larger.
    Question: does WF+thr's advantage survive under the current DP accounting?

(E2) uniform+thr vs WF+thr
    Same threshold (keep top 10% / 25%), but on the *active set* use:
      - WF allocation (σ_c by waterfilling on the subset)
      - uniform allocation (single σ across the active subset)
    Question: is the win from thresholding alone, or does the WF formula
    on the active set also contribute?

Plot: 2-panel figure.
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


def thresholded_sigma(deltas, importance, rho, active_mask, scheme: str):
    """
    Allocate σ on the active set under `scheme` ('wf' or 'uniform'); set σ=0 on
    inactive (they'll be zeroed before decoder).
    """
    sigma = torch.zeros_like(deltas)
    if not active_mask.any():
        return sigma
    if scheme == "wf":
        sigma[active_mask] = waterfilling_sigma(
            deltas[active_mask], importance[active_mask], rho,
        )
    elif scheme == "uniform":
        sigma[active_mask] = uniform_sigma(deltas[active_mask], rho)
    else:
        raise ValueError(scheme)
    return sigma


def probe(teacher, val_loader, importance, caps, device, epsilons, K,
          keep_fractions, schemes):
    """
    Returns mean Dice for clean / uniform-no-thr / and every
    (scheme, keep_fraction, epsilon) combination.
    """
    Cb = importance.shape[0]
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
        sigma_tab[eps] = {"uniform_full": uniform_sigma(deltas, rho)}
        for scheme in schemes:
            for f in keep_fractions:
                sigma_tab[eps][(scheme, f)] = thresholded_sigma(
                    deltas, importance, rho, masks[f], scheme,
                )

    res = {"clean": [], "uniform_full": {e: [] for e in epsilons}}
    for scheme in schemes:
        for f in keep_fractions:
            res[(scheme, f)] = {e: [] for e in epsilons}

    teacher.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            e1, e2, e3 = teacher.encode(x)
            res["clean"].append(vessel_dice(teacher.decode(e1, e2, e3), y))

            bn_norm = clip_and_normalise(e3, caps)
            B, C, H, W = bn_norm.shape

            for eps in epsilons:
                # uniform (no threshold)
                torch.manual_seed(42 + int(eps * 100))
                sigma = sigma_tab[eps]["uniform_full"]
                noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                bn_noisy = denormalise(bn_norm + noise, caps)
                res["uniform_full"][eps].append(
                    vessel_dice(teacher.decode(e1, e2, bn_noisy), y)
                )

                for scheme in schemes:
                    for f in keep_fractions:
                        torch.manual_seed(42 + int(eps * 100))
                        sigma = sigma_tab[eps][(scheme, f)]
                        noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                        bn_thr = bn_norm + noise
                        inactive = ~masks[f]
                        if inactive.any():
                            bn_thr[:, inactive, :, :] = 0
                        bn_noisy = denormalise(bn_thr, caps)
                        res[(scheme, f)][eps].append(
                            vessel_dice(teacher.decode(e1, e2, bn_noisy), y)
                        )

    means = {"clean": float(np.mean(res["clean"])),
             "uniform_full": {e: float(np.mean(res["uniform_full"][e])) for e in epsilons}}
    for scheme in schemes:
        for f in keep_fractions:
            means[(scheme, f)] = {e: float(np.mean(res[(scheme, f)][e]))
                                  for e in epsilons}
    return means


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

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
    print(f"  clean Dice: {clean_dice:.4f}")
    print(f"  importance ratio: {(importance.max()/importance.min()).item():.2f}×")

    # ----------------------------------------------------------------------
    # E1: K=1 verification (sample-once accounting)
    # ----------------------------------------------------------------------
    print("\n[E1] K=1 sweep (sample-once accounting, σ 10× larger than K=10)...")
    eps_k1 = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    means_k1 = probe(teacher, val_loader, importance, caps, device,
                     eps_k1, K=1,
                     keep_fractions=[1.0, 0.25, 0.1],
                     schemes=["wf"])

    print(f"  CLEAN: {clean_dice:.4f}")
    hdr = f"  {'mech':<22s}" + "".join(f"  ε={e:>5.1f}" for e in eps_k1)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'uniform':<22s}" + "".join(f"  {means_k1['uniform_full'][e]:>7.4f}" for e in eps_k1))
    for f in [1.0, 0.25, 0.1]:
        lbl = "plain WF" if f == 1.0 else f"WF+thr keep {int(f*100):>3d}%"
        print(f"  {lbl:<22s}" + "".join(f"  {means_k1[('wf', f)][e]:>7.4f}" for e in eps_k1))

    # ----------------------------------------------------------------------
    # E2: uniform+thr vs WF+thr (K=10 to keep continuity with main plot)
    # ----------------------------------------------------------------------
    print("\n[E2] uniform+thr vs WF+thr at K=10 (where threshold won big)...")
    eps_e2 = [4.0, 8.0, 16.0, 32.0]
    means_e2 = probe(teacher, val_loader, importance, caps, device,
                     eps_e2, K=10,
                     keep_fractions=[0.25, 0.1],
                     schemes=["wf", "uniform"])

    hdr = f"  {'mech':<24s}" + "".join(f"  ε={e:>4.1f}" for e in eps_e2)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'uniform (no thr)':<24s}" + "".join(f"  {means_e2['uniform_full'][e]:>7.4f}" for e in eps_e2))
    for f in [0.25, 0.1]:
        print(f"  {'uniform + thr keep ' + str(int(f*100)) + '%':<24s}" +
              "".join(f"  {means_e2[('uniform', f)][e]:>7.4f}" for e in eps_e2))
        print(f"  {'WF + thr keep ' + str(int(f*100)) + '%':<24s}" +
              "".join(f"  {means_e2[('wf', f)][e]:>7.4f}" for e in eps_e2))

    # ----------------------------------------------------------------------
    # Save + plot
    # ----------------------------------------------------------------------
    out = {
        "clean": clean_dice,
        "K1_sweep": {
            "epsilons": eps_k1,
            "uniform":    {str(e): means_k1["uniform_full"][e]    for e in eps_k1},
            "plain_WF":   {str(e): means_k1[("wf", 1.0)][e]       for e in eps_k1},
            "WF_thr_25":  {str(e): means_k1[("wf", 0.25)][e]      for e in eps_k1},
            "WF_thr_10":  {str(e): means_k1[("wf", 0.1)][e]       for e in eps_k1},
        },
        "uniform_vs_wf_on_active_set_K10": {
            "epsilons": eps_e2,
            "uniform_no_thr":  {str(e): means_e2["uniform_full"][e]    for e in eps_e2},
            "uniform_thr_25":  {str(e): means_e2[("uniform", 0.25)][e] for e in eps_e2},
            "WF_thr_25":       {str(e): means_e2[("wf", 0.25)][e]      for e in eps_e2},
            "uniform_thr_10":  {str(e): means_e2[("uniform", 0.1)][e]  for e in eps_e2},
            "WF_thr_10":       {str(e): means_e2[("wf", 0.1)][e]       for e in eps_e2},
        },
    }
    json_path = Path(__file__).parent / "drive_k1_and_uniform_thr_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved JSON: {json_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("DRIVE — K=1 verification (left)  +  uniform-vs-WF on active set (right)",
                 fontsize=13, fontweight="bold")

    # --- Panel a: K=1 sweep ---
    ax = axes[0]
    ax.axhline(clean_dice, color="#555", ls=":", lw=2.5,
               label=f"clean = {clean_dice:.3f}")
    palette = {1.0: "#1f77b4", 0.25: "#9467bd", 0.1: "#ff7f0e"}
    ax.plot(eps_k1, [means_k1["uniform_full"][e] for e in eps_k1],
            "s--", color="#d62728", lw=2, ms=7, label="uniform")
    for f in [1.0, 0.25, 0.1]:
        ys = [means_k1[("wf", f)][e] for e in eps_k1]
        lbl = "plain channel-WF" if f == 1.0 else f"WF+thr keep {int(f*100)}%"
        ax.plot(eps_k1, ys, "o-", color=palette[f], lw=2.2, ms=8, label=lbl)
    ax.set_xscale("log", base=2)
    ax.set_xticks(eps_k1)
    ax.set_xticklabels([str(int(e)) for e in eps_k1])
    ax.set_xlabel("privacy budget  $\\epsilon$  (K=1, sample-once)")
    ax.set_ylabel("vessel Dice")
    ax.set_title("(a) K=1 — does WF+thr survive sample-once accounting?")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel b: uniform vs WF on active set (K=10) ---
    ax = axes[1]
    ax.axhline(clean_dice, color="#555", ls=":", lw=2.5,
               label=f"clean = {clean_dice:.3f}")
    ax.plot(eps_e2, [means_e2["uniform_full"][e] for e in eps_e2],
            "s--", color="#d62728", lw=2, ms=7, label="uniform (no thr)")
    colors_e2 = {0.25: "#9467bd", 0.1: "#ff7f0e"}
    for f in [0.25, 0.1]:
        ax.plot(eps_e2, [means_e2[("uniform", f)][e] for e in eps_e2],
                "^--", color=colors_e2[f], lw=1.8, ms=8,
                label=f"uniform+thr keep {int(f*100)}%")
        ax.plot(eps_e2, [means_e2[("wf", f)][e] for e in eps_e2],
                "o-",  color=colors_e2[f], lw=2.2, ms=8,
                label=f"WF+thr keep {int(f*100)}%")
    ax.set_xscale("log", base=2)
    ax.set_xticks(eps_e2)
    ax.set_xticklabels([str(int(e)) for e in eps_e2])
    ax.set_xlabel("privacy budget  $\\epsilon$  (K=10)")
    ax.set_ylabel("vessel Dice")
    ax.set_title("(b) Threshold vs WF: who contributes the lift?")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = Path(__file__).parent / "drive_k1_and_uniform_thr.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
