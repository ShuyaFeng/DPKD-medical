"""
Output-Space DP Baseline — PATE adaptation for medical image segmentation.

Adapts the PATE framework (Papernot et al., ICLR 2017, "Semi-supervised
Knowledge Transfer for Deep Learning from Private Training Data") to
segmentation. Discrete class voting (LNMax) is replaced with soft-mask
averaging: each teacher produces a foreground probability map, maps are
clipped and averaged across K teachers, and Gaussian noise is added.
We call this "Output-Space DP" and cite PATE as the foundational paradigm.

Departures from original PATE (all disclosed in paper):
  1. K=3 teachers instead of n=250.
     Original PATE relies on large n for the vote gap (gap ≈ 60% of n).
     With K=3, max gap = 3; q ≈ 0.46 even at unanimous agreement (Lemma 4).
     Poor performance at small K is an expected and reported result.

  2. Continuous soft masks instead of discrete hard labels.
     PATE outputs argmax over vote counts — one integer label per image.
     Segmentation requires H×W predictions; argmax is not applicable.
     Each teacher outputs softmax(logits)[:,1] ∈ [0,1]^(H×W) instead.
     Student trains with BCE + soft-Dice on clamped noisy masks (not CE on
     hard labels as in the original).

  3. Gaussian mechanism instead of LNMax (Laplacian on argmax).
     Original: f(x) = argmax_j { n_j(x) + Lap(1/γ) }  (Eq. 1, Theorem 2).
     Ours: noisy_mask = mean(clipped_masks) + N(0, σ²I).
     Laplacian + argmax cannot operate on continuous H×W vectors.

  4. zCDP (Gaussian) instead of moments accountant (Laplacian).
     Original PATE uses the moments accountant (Abadi et al. 2016) for
     tight data-dependent bounds under Laplacian noise.
     We use zCDP (ρ = eps_to_rho(ε, δ)) with Gaussian noise for consistency
     with CANAL, enabling a direct apples-to-apples comparison.

  5. Student queries teachers on PRIVATE training data (not public data).
     Original PATE trains the student on public/unlabeled non-sensitive data
     labeled by the noisy vote mechanism.
     No public medical imaging data is available in our setting, so we query
     teachers on the private training images to generate soft-mask supervision.

  6. Supervised student instead of semi-supervised GAN training (PATE-G).
     The GAN-based semi-supervised approach requires public unlabeled data
     and is not applicable here.

DP accounting (zCDP, same framework as CANAL):
  C  = 90th-percentile of soft-mask L2 norms (all teachers × all train images)
  Δ  = 2C / K   (replace-one sensitivity: one record change shifts the
                  mean by ≤ 2C/K in L2 norm, since only one teacher changes)
  ρ  = eps_to_rho(ε, δ=1e-5)
  σ  = Δ / sqrt(2ρ)

  Accounting simplification (same as CANAL): the precomputed cache contains
  one noisy release per training image (N total). We account for ε as a
  single release (sample-once principle), identical to how CANAL accounts for
  its N cached bottleneck features. The relative comparison between methods
  is therefore fair.

Why this baseline underperforms CANAL:
  C (soft-mask L2 norm on a 96×96 image) ≈ 20–40, so Δ = 2C/K >> 2/K
  (CANAL's sensitivity in unit-norm feature space). Proportionally more
  Gaussian noise is required at every ε, leading to lower Dice scores.
  This directly demonstrates CANAL's advantage from feature-space compression.

Pipeline:
  1. Train K=3 TinyUNet(base=32) teachers on disjoint patient cohorts (60 ep)
  2. Compute C = 90th-percentile of soft-mask L2 norms across all teachers
  3. For each ε: compute σ, cache one noisy soft mask per training image
  4. Train TinyUNet(base=16) student for 40 ep with BCE + soft-Dice loss

Usage:
  python pate_baseline.py --dataset isic
  python pate_baseline.py --dataset kvasir
  python pate_baseline.py --dataset busi
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet, evaluate_vessel_dice
from drive_pate_poc import train_K_teachers
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K               = 3
EPSILONS        = [1.0, 2.0, 4.0, 8.0]
SEEDS           = [100, 200, 300, 400, 500]
DELTA           = 1e-5
CAPS_PERCENTILE = 90
TEACHER_EPOCHS  = 60
STUDENT_EPOCHS  = 40

DATASET_CFG = {
    "isic":   dict(in_ch=3, bs=45),
    "kvasir": dict(in_ch=3, bs=28),
    "busi":   dict(in_ch=1, bs=23),
}


# =============================================================================
# Dataset helper
# =============================================================================

def get_dataset(name, split):
    if name == "isic":
        from isic_dataset import ISICDataset
        return ISICDataset(split, 96)
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset
        return KvasirDataset(split, 96)
    if name == "busi":
        from busi_dataset import BUSIDataset
        return BUSIDataset(split, 96)
    raise ValueError(f"Unknown dataset: {name}. Choose from isic, kvasir, busi.")


# =============================================================================
# DP machinery
# =============================================================================

@torch.no_grad()
def compute_C(teachers, train_ds, device):
    """90th-percentile L2 norm cap over all (teacher, image) pairs."""
    all_norms = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    for teacher in teachers:
        teacher.eval()
        for x, _ in loader:
            x = x.to(device)
            soft_mask = teacher(x).softmax(1)[:, 1]         # B×H×W
            norms = soft_mask.flatten(1).norm(dim=1)         # B
            all_norms.extend(norms.tolist())
    return float(np.percentile(all_norms, CAPS_PERCENTILE))


@torch.no_grad()
def precompute_output_cache(teachers, train_ds, C, sigma, device, seed):
    """
    Sample-once PATE soft-mask cache. For each training image:
      1. Each teacher produces soft_mask_k = softmax(teacher_k(x))[:,1] ∈ [0,1]^(H×W)
      2. Clip L2 norm of each soft_mask_k to C
      3. Average across K teachers
      4. Add isotropic Gaussian noise N(0, σ²I)
    Returns a list of (H×W) CPU tensors indexed by training-set position.
    """
    torch.manual_seed(seed)
    cache = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    for x, _ in loader:
        x = x.to(device)
        clipped_masks = []
        for teacher in teachers:
            sm    = teacher(x).softmax(1)[:, 1]                # B×H×W
            norms = sm.flatten(1).norm(dim=1)                  # B
            scale = (C / norms.clamp(min=1e-8)).clamp(max=1.0)
            clipped_masks.append(sm * scale.view(-1, 1, 1))
        avg   = torch.stack(clipped_masks).mean(0)             # B×H×W
        noisy = avg + torch.randn_like(avg) * sigma
        for b in range(avg.shape[0]):
            cache.append(noisy[b].detach().cpu())
    return cache


# =============================================================================
# Student training
# =============================================================================

class CachedOutputDataset(Dataset):
    """Wraps a base segmentation dataset with pre-computed noisy soft masks."""

    def __init__(self, base_ds, cache):
        self.base_ds = base_ds
        self.cache   = cache

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, y, self.cache[idx]


def soft_mask_loss(logits, noisy_mask):
    """BCE + soft-Dice loss against a noisy soft-mask target (clamped to [0,1])."""
    target = noisy_mask.clamp(0.0, 1.0)
    prob   = logits.softmax(1)[:, 1]                       # B×H×W
    bce    = F.binary_cross_entropy(prob, target)
    inter  = (prob * target).sum(dim=(1, 2))
    denom  = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice   = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return bce + dice


def train_student(base_ds, cache, val_loader, in_ch, bs, device):
    """
    Train a TinyUNet(base=16) student using cached noisy soft masks.
    Returns best val Dice across all epochs.
    """
    ds      = CachedOutputDataset(base_ds, cache)
    loader  = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    student = TinyUNet(in_ch=in_ch, num_classes=2, base=16).to(device)
    opt     = torch.optim.Adam(student.parameters(), lr=1e-3)

    best_dice = 0.0
    for ep in range(STUDENT_EPOCHS):
        student.train()
        for x, _y, nm in loader:
            x  = x.to(device)
            nm = nm.to(device)
            opt.zero_grad()
            soft_mask_loss(student(x), nm).backward()
            opt.step()

        val_dice = evaluate_vessel_dice(student, val_loader, device)
        if val_dice > best_dice:
            best_dice = val_dice
        if (ep + 1) % 10 == 0:
            print(f"    student ep {ep+1:2d}/{STUDENT_EPOCHS}"
                  f"  val_dice={val_dice:.4f}  best={best_dice:.4f}")

    return best_dice


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["isic", "kvasir", "busi"])
    args = parser.parse_args()

    cfg   = DATASET_CFG[args.dataset]
    in_ch = cfg["in_ch"]
    bs    = cfg["bs"]

    train_ds   = get_dataset(args.dataset, "train")
    val_ds     = get_dataset(args.dataset, "val")
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    print(f"PATE baseline (Papernot et al. 2017) — {args.dataset} | device={DEVICE}")
    print(f"  in_ch={in_ch}  bs={bs}  K={K}")
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}")
    print(f"  Teacher epochs={TEACHER_EPOCHS}  Student epochs={STUDENT_EPOCHS}")

    results = {
        "method":          "Output-Space DP (PATE — Papernot et al. 2017, adapted for segmentation)",
        "dataset":         args.dataset,
        "K":               K,
        "teacher_epochs":  TEACHER_EPOCHS,
        "student_epochs":  STUDENT_EPOCHS,
        "delta":           DELTA,
        "caps_percentile": CAPS_PERCENTILE,
        "epsilons":        EPSILONS,
        "seeds":           SEEDS,
        "sweep":           {},
    }

    all_dices_by_eps = {str(e): [] for e in EPSILONS}
    sigma_by_eps     = {}

    for s_idx, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n[Seed {seed} ({s_idx+1}/{len(SEEDS)})]"
              f" Training K={K} teachers ({TEACHER_EPOCHS} epochs)...")
        teachers, _ = train_K_teachers(
            train_ds, K, DEVICE, n_epochs=TEACHER_EPOCHS, in_ch=in_ch
        )

        C   = compute_C(teachers, train_ds, DEVICE)
        sen = 2.0 * C / K
        print(f"  C (p{CAPS_PERCENTILE} soft-mask L2) = {C:.4f}   Δ = 2C/K = {sen:.4f}")

        for eps in EPSILONS:
            rho   = eps_to_rho(eps, delta=DELTA)
            sigma = sen / math.sqrt(2.0 * rho)
            print(f"\n  ε={eps}  ρ={rho:.5f}  σ={sigma:.4f}")

            cache = precompute_output_cache(
                teachers, train_ds, C, sigma, DEVICE,
                seed=seed + int(eps * 100),
            )

            torch.manual_seed(seed)
            np.random.seed(seed)
            dice = train_student(train_ds, cache, val_loader, in_ch, bs, DEVICE)
            all_dices_by_eps[str(eps)].append(dice)
            sigma_by_eps[str(eps)] = {"sigma": float(sigma), "C": float(C), "delta": float(sen)}
            print(f"  ε={eps}  seed={seed}:  Dice={dice:.4f}  (σ={sigma:.4f}  C={C:.4f})")

    print("\n" + "=" * 60)
    for eps in EPSILONS:
        dices = all_dices_by_eps[str(eps)]
        mean  = float(np.mean(dices))
        std   = float(np.std(dices))
        results["sweep"][str(eps)] = {
            "dices": [float(d) for d in dices],
            "mean":  mean,
            "std":   std,
            "sem":   std / math.sqrt(len(dices)),
            "sigma": sigma_by_eps[str(eps)]["sigma"],
            "C":     sigma_by_eps[str(eps)]["C"],
            "delta": sigma_by_eps[str(eps)]["delta"],
        }
        print(f"ε={eps}: {mean:.4f} ± {std:.4f}  (σ={sigma_by_eps[str(eps)]['sigma']:.4f})")
    print("=" * 60)

    out = HERE / "results" / f"{args.dataset}_pate_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
