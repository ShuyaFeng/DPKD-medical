"""
Empirical privacy audit #1 — Membership Inference Attack (MIA) on the
sample-once feature release.

Threat model (conservative / worst-case adversary): the attacker HOLDS the
teacher, so it can compute the clean normalized bottleneck of any candidate
image. Given the released (noisy) features of the 20 training members, it
decides membership of a candidate by the distance to the nearest release —
the near-optimal likelihood-ratio test for a Gaussian mechanism.

Members = 20 DRIVE train, non-members = 20 DRIVE val. We sweep the noise from
σ=0 (NO NOISE) to the σ that realizes ε∈{16,8,4,2,1,0.5} on the K=1 release,
and report attack AUC + TPR@FPR=0.1, averaged over many fresh noise draws.
The no-noise point shows total membership leakage; the DP bound TPR≤e^ε·FPR+δ
is overlaid on the ROC.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import DriveDataset, TinyUNet, train_teacher, collect_caps
from synthetic_demo import eps_to_rho, clip_and_normalise
from drive_pate_poc import correct_uniform_sigma

HERE = Path(__file__).parent
DELTA = 1e-5


def auc_mw(pos, neg):
    """Mann–Whitney AUC = P(score_member > score_nonmember)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (pos.size * neg.size)


@torch.no_grad()
def features(teacher, ds, caps, device):
    out = []
    for x, _ in DataLoader(ds, batch_size=4):
        _, _, e3 = teacher.encode(x.to(device))
        out.append(clip_and_normalise(e3, caps).cpu())
    return torch.cat(out).reshape(len(ds), -1).numpy()   # (N, D)


def nearest_release_score(release, cand):
    """score(x) = -min_j ||release_j - feat(x)||^2  (higher = more 'member-like')."""
    d = ((release[:, None, :] - cand[None, :, :]) ** 2).sum(-1)   # (Nm, M)
    return -d.min(0)


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    train_ds, val_ds = DriveDataset("train", 96), DriveDataset("val", 96)
    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, DataLoader(train_ds, batch_size=4, shuffle=True),
                  n_epochs=60, lr=1e-3, device=device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    caps = collect_caps(teacher, DataLoader(train_ds, batch_size=4), device).to(device)

    Fm = features(teacher, train_ds, caps, device)   # members  (20, D)
    Fn = features(teacher, val_ds,   caps, device)    # non-mem  (20, D)
    C = teacher.base * 4
    deltas = torch.full((C,), 2.0)                    # Δ = 2/K, K=1
    sig = lambda e: float(correct_uniform_sigma(deltas, eps_to_rho(e))[0])

    epsilons = [16.0, 8.0, 4.0, 2.0, 1.0, 0.5]
    configs = [("no-noise", None, 0.0)] + [(f"ε={e}", e, sig(e)) for e in epsilons]
    rng = np.random.default_rng(0)
    N_TRIAL = 200
    FPR_GRID = np.linspace(0, 1, 51)

    results, roc = {}, {}
    print(f"\n{'config':>10} {'σ':>9} {'AUC':>8} {'TPR@FPR.1':>11}")
    for name, eps, sigma in configs:
        aucs, tpr01, tpr_grid = [], [], []
        for _ in range(N_TRIAL):
            noise = rng.normal(0, sigma, Fm.shape) if sigma > 0 else np.zeros_like(Fm)
            rel = Fm + noise
            s_m = nearest_release_score(rel, Fm)
            s_n = nearest_release_score(rel, Fn)
            aucs.append(auc_mw(s_m, s_n))
            thr = np.quantile(s_n, 0.9)               # FPR = 0.1
            tpr01.append((s_m >= thr).mean())
            # ROC: TPR at each FPR grid point
            thr_grid = np.quantile(s_n, 1 - FPR_GRID)
            tpr_grid.append([(s_m >= t).mean() for t in thr_grid])
        results[name] = {"epsilon": eps, "sigma": sigma,
                         "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                         "tpr_at_fpr0.1": float(np.mean(tpr01))}
        roc[name] = np.mean(tpr_grid, 0)
        print(f"{name:>10} {sigma:>9.3f} {np.mean(aucs):>8.3f} {np.mean(tpr01):>11.3f}")

    (HERE / "drive_mia_attack_results.json").write_text(json.dumps(results, indent=2))

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) AUC vs ε  (no-noise shown at far right)
    xs = [e for *_, e in [(n, c["epsilon"]) for n, c in results.items()] if e is not None]
    ax1.axhline(0.5, color="gray", ls="--", lw=1.2, label="random guess (AUC=0.5)")
    ax1.plot(epsilons, [results[f"ε={e}"]["auc"] for e in epsilons], "o-",
             color="#d62728", lw=2, ms=9, label="MIA AUC (with noise)")
    ax1.scatter([max(epsilons) * 2], [results["no-noise"]["auc"]], color="black",
                s=120, marker="*", zorder=5, label=f"NO NOISE = {results['no-noise']['auc']:.2f}")
    ax1.set_xscale("log"); ax1.set_xlabel("privacy budget ε"); ax1.set_ylabel("membership-inference AUC")
    ax1.set_ylim(0.45, 1.02); ax1.set_title("(a) MIA AUC vs ε — noise kills membership signal")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # (b) ROC at a few ε + no-noise + DP bound
    for name, col in [("no-noise", "black"), ("ε=16.0", "#ff7f0e"),
                      ("ε=8.0", "#2ca02c"), ("ε=2.0", "#1f77b4")]:
        ax2.plot(FPR_GRID, roc[name], "-", color=col, lw=2, label=f"{name} (AUC {results[name]['auc']:.2f})")
        e = results[name]["epsilon"]
        if e is not None:
            ax2.plot(FPR_GRID, np.minimum(1, np.exp(e) * FPR_GRID + DELTA), ":",
                     color=col, lw=1, alpha=0.6)
    ax2.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax2.set_xlabel("false-positive rate"); ax2.set_ylabel("true-positive rate")
    ax2.set_title("(b) ROC  (dotted = theoretical (ε,δ)-DP bound)")
    ax2.legend(fontsize=8.5, loc="lower right"); ax2.grid(alpha=0.3)

    fig.suptitle("Membership inference on the sample-once feature release (K=1, worst-case adversary)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(HERE / "fig_mia.png", dpi=145)
    print("saved fig_mia.png + drive_mia_attack_results.json")


if __name__ == "__main__":
    main()
