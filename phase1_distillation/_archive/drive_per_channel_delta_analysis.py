"""
Analytical estimate: per-channel Δ_c (raw-space sensitivity) vs Δ_c constant
(normalized-space sensitivity, current code) on real DRIVE.

Both settings give the same DP guarantee at the same ρ; what differs is the
σ allocation and therefore the importance-weighted reconstruction error J
(the surrogate Theorem 1 minimizes).

Setting A (current code, phase1_gkd_distill_v2.py):
    - clip + per-channel L2 normalize → unit ball
    - Δ_c = 2/K  (constant)
    - WF in normalized space; denormalize after
    - effective raw-space noise std:  τ_A_c = σ_A_c × cap_c

Setting B (per-channel Δ, no normalize):
    - clip only
    - Δ_c = 2 cap_c / K   (per-channel)
    - WF in raw space directly
    - effective raw-space noise std:  τ_B_c = σ_B_c

We compute and compare:
  (i)   σ spread between unimportant/important channels
  (ii)  surrogate objective  J = Σ s_c τ_c²  for WF vs uniform under each setting
  (iii) WF advantage  (J_uni − J_WF) / J_uni

If even the analytical advantage is tiny, no point running a noise probe.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher,
    compute_importance, collect_caps,
)
from synthetic_demo import eps_to_rho


# -------------------------------------------------------------------------
# WF / uniform with per-channel Δ
# -------------------------------------------------------------------------

def wf_sigma(deltas: torch.Tensor,
             importance: torch.Tensor,
             rho: float) -> torch.Tensor:
    """σ_c = κ √Δ_c / s_c^{1/4},  κ = √((1/(2ρ)) Σ Δ_c √s_c)."""
    s = importance.clamp(min=1e-12)
    kappa = ((deltas * s.sqrt()).sum() / (2.0 * rho)).sqrt()
    return kappa * deltas.sqrt() / s.pow(0.25)


def uniform_sigma(deltas: torch.Tensor, rho: float) -> torch.Tensor:
    """σ_c = const,  Σ Δ_c²/(2σ²) = ρ → σ = √(Σ Δ_c² / (2ρ))."""
    sigma_val = (deltas.pow(2).sum() / (2.0 * rho)).sqrt()
    return sigma_val.expand(deltas.shape[0]).clone()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    # ---- teacher + per-channel caps & importance ----
    print("\n[Setup] Loading DRIVE + training teacher...")
    train_ds = DriveDataset("train", size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3, device=device)

    importance = compute_importance(teacher, train_loader, device).cpu()
    caps       = collect_caps(teacher, train_loader, device).cpu()
    C = importance.shape[0]

    cap_ratio = (caps.max() / caps.min()).item()
    imp_ratio = (importance.max() / importance.min()).item()
    pearson = float(np.corrcoef(caps.numpy(), importance.numpy())[0, 1])
    spearman = float(np.corrcoef(
        np.argsort(np.argsort(caps.numpy())),
        np.argsort(np.argsort(importance.numpy())),
    )[0, 1])

    print(f"\nC = {C}")
    print(f"caps:       min={caps.min():.3f}  max={caps.max():.3f}  "
          f"mean={caps.mean():.3f}  ratio={cap_ratio:.2f}×")
    print(f"importance: min={importance.min():.3e}  max={importance.max():.3e}  "
          f"ratio={imp_ratio:.2f}×")
    print(f"cap–importance correlation:  Pearson={pearson:+.3f}  "
          f"Spearman={spearman:+.3f}")

    # ---- sensitivities ----
    K = 1                                          # honest single teacher
    deltas_A = torch.full((C,), 2.0 / K)            # constant (current code)
    deltas_B = 2.0 * caps / K                       # per-channel raw

    print(f"\nK = {K}")
    print(f"Δ_A constant = {2.0/K:.4f}  for every channel")
    print(f"Δ_B per-channel: min={deltas_B.min():.3f}  max={deltas_B.max():.3f}  "
          f"mean={deltas_B.mean():.3f}")

    # ---- sweep ε ----
    epsilons = [2.0, 4.0, 8.0, 16.0, 32.0]
    rows = []
    print("\n" + "=" * 96)
    print(f"{'ε':>5}  {'τ spread (max/min)':^28}  {'σ_bot/σ_top':^22}  {'WF advantage J':^20}")
    print(f"{'':5}  {'Setting A':>13}  {'Setting B':>13}  {'A':>10}  {'B':>10}  {'A':>8}  {'B':>8}")
    print("=" * 96)
    for eps in epsilons:
        rho = eps_to_rho(eps)

        # ===== Setting A: Δ=2/K constant, WF in normalized space =====
        sigma_A_norm   = wf_sigma(deltas_A, importance, rho)
        sigma_A_norm_u = uniform_sigma(deltas_A, rho)
        # raw-space noise std (after denormalize)
        tau_A     = sigma_A_norm   * caps
        tau_A_uni = sigma_A_norm_u * caps

        # ===== Setting B: Δ_c=2 cap_c/K per-channel, WF in raw space =====
        tau_B     = wf_sigma(deltas_B, importance, rho)
        tau_B_uni = uniform_sigma(deltas_B, rho)

        # surrogate: J = Σ s_c τ_c²
        J_A_WF  = (importance * tau_A.pow(2)).sum().item()
        J_A_uni = (importance * tau_A_uni.pow(2)).sum().item()
        J_B_WF  = (importance * tau_B.pow(2)).sum().item()
        J_B_uni = (importance * tau_B_uni.pow(2)).sum().item()

        # σ spread (max/min and bot10/top10)
        spread_A_WF = (tau_A.max() / tau_A.min()).item()
        spread_B_WF = (tau_B.max() / tau_B.min()).item()

        top10 = torch.argsort(importance, descending=True)[:10]
        bot10 = torch.argsort(importance, descending=False)[:10]
        bot_top_A = (tau_A[bot10].mean() / tau_A[top10].mean()).item()
        bot_top_B = (tau_B[bot10].mean() / tau_B[top10].mean()).item()

        # WF advantage as percent
        adv_A = (J_A_uni - J_A_WF) / J_A_uni * 100
        adv_B = (J_B_uni - J_B_WF) / J_B_uni * 100

        print(f"{eps:>5.1f}  {spread_A_WF:>13.3f}  {spread_B_WF:>13.3f}  "
              f"{bot_top_A:>10.3f}  {bot_top_B:>10.3f}  "
              f"{adv_A:>+7.2f}%  {adv_B:>+7.2f}%")

        rows.append({
            "epsilon": eps,
            "rho": rho,
            "spread_A_WF":  spread_A_WF,
            "spread_B_WF":  spread_B_WF,
            "bot_top_A":    bot_top_A,
            "bot_top_B":    bot_top_B,
            "J_A_WF":  J_A_WF, "J_A_uni": J_A_uni, "advantage_A_pct": adv_A,
            "J_B_WF":  J_B_WF, "J_B_uni": J_B_uni, "advantage_B_pct": adv_B,
        })
    print("=" * 96)

    # ---- per-channel sigma plot for one mid ε ----
    eps_focus = 8.0
    rho_focus = eps_to_rho(eps_focus)
    tau_A_focus     = wf_sigma(deltas_A, importance, rho_focus) * caps
    tau_A_uni_focus = uniform_sigma(deltas_A, rho_focus) * caps
    tau_B_focus     = wf_sigma(deltas_B, importance, rho_focus)
    tau_B_uni_focus = uniform_sigma(deltas_B, rho_focus)

    out = {
        "C": C, "K": K,
        "cap_stats": {"min": float(caps.min()), "max": float(caps.max()),
                      "mean": float(caps.mean()), "ratio": cap_ratio},
        "imp_stats": {"min": float(importance.min()), "max": float(importance.max()),
                      "mean": float(importance.mean()), "ratio": imp_ratio},
        "cap_imp_pearson": pearson,
        "cap_imp_spearman": spearman,
        "rows": rows,
    }
    out_path = Path(__file__).parent / "drive_per_channel_delta_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved JSON: {out_path}")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    eps_arr = [r["epsilon"] for r in rows]

    # Panel a: WF advantage (J) under each setting
    ax = axes[0]
    ax.plot(eps_arr, [r["advantage_A_pct"] for r in rows], "s-",
            color="#1f77b4", lw=2, ms=9,
            label="Setting A: Δ=2/K (current code)")
    ax.plot(eps_arr, [r["advantage_B_pct"] for r in rows], "o-",
            color="#ff7f0e", lw=2, ms=9,
            label="Setting B: Δ_c=2·cap_c/K (per-channel)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(eps_arr)
    ax.set_xticklabels([str(int(e)) for e in eps_arr])
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("WF advantage over uniform  (%)")
    ax.set_title("(a) Surrogate gain  (J_uni − J_WF) / J_uni")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel b: σ spread (bot10/top10)
    ax = axes[1]
    ax.plot(eps_arr, [r["bot_top_A"] for r in rows], "s-",
            color="#1f77b4", lw=2, ms=9,
            label="Setting A")
    ax.plot(eps_arr, [r["bot_top_B"] for r in rows], "o-",
            color="#ff7f0e", lw=2, ms=9,
            label="Setting B")
    ax.axhline(1, color="black", lw=0.5, ls=":", label="uniform = 1.0")
    ax.set_xscale("log", base=2)
    ax.set_xticks(eps_arr)
    ax.set_xticklabels([str(int(e)) for e in eps_arr])
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("σ ratio:  bot-10 / top-10  (raw space)")
    ax.set_title("(b) σ spread between unimportant / important channels")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel c: per-channel σ at ε=8, sorted by importance
    ax = axes[2]
    order = torch.argsort(importance, descending=True)
    x = np.arange(C)
    ax.plot(x, tau_A_focus[order].numpy(),     "-", color="#1f77b4", lw=1.5,
            label="A WF τ")
    ax.plot(x, tau_A_uni_focus[order].numpy(), "--", color="#1f77b4", lw=1.5, alpha=0.6,
            label="A uniform τ")
    ax.plot(x, tau_B_focus[order].numpy(),     "-", color="#ff7f0e", lw=1.5,
            label="B WF τ")
    ax.plot(x, tau_B_uni_focus[order].numpy(), "--", color="#ff7f0e", lw=1.5, alpha=0.6,
            label="B uniform τ")
    ax.set_xlabel("channel rank by importance (high → low)")
    ax.set_ylabel("raw-space noise std  τ")
    ax.set_title(f"(c) per-channel τ at ε={eps_focus:.0f}  (lower = less noise)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"DRIVE — Setting A (Δ const) vs Setting B (Δ per-channel)   "
        f"|   C={C},  K={K},  cap ratio={cap_ratio:.2f}×,  "
        f"imp ratio={imp_ratio:.2f}×,  Pearson(cap,imp)={pearson:+.3f}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = Path(__file__).parent / "drive_per_channel_delta.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
