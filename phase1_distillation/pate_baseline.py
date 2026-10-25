"""
Output-Space DP Baseline — PATE with GNMax aggregator for segmentation.

Implements the PATE framework with the GNMax aggregator and Rényi DP
accounting from the follow-up paper:
  Papernot et al., "Scalable Private Learning with PATE"
  ICLR 2018, arXiv:1802.08908

The original PATE (Papernot et al., ICLR 2017, arXiv:1610.05755) used
LNMax (Laplace on vote counts) with the moments accountant. We use GNMax
(Gaussian on vote counts) with Rényi DP accounting so that epsilon is
directly comparable to CANAL's Gaussian/zCDP setup.

Aggregation (GNMax, Algorithm 1 of Papernot et al. 2018):
  Each teacher produces a hard binary segmentation mask (argmax of logits).
  For each pixel (h,w), vote counts n_1(h,w) and n_0(h,w) = K − n_1(h,w)
  are formed. Independent Gaussian noise N(0,σ²) is added to each count,
  and argmax selects the winning class. The result is a hard binary mask
  cached once per training image (sample-once, matching CANAL).

Sensitivity (per Dr. Tian):
  Replacing one training record changes one teacher. In the worst case that
  teacher's mask flips at every pixel, changing vote counts by ±1 at all
  H×W pixels. The L2 sensitivity of the (H×W × 2)-dimensional vote vector
  is therefore Δ_2 = √(2HW). For 96×96 images: Δ_2 ≈ 135.8.

DP accounting (Rényi DP / zCDP — same framework as CANAL):
  For the Gaussian mechanism with L2 sensitivity Δ_2, the optimal
  RDP → (ε,δ)-DP conversion is equivalent to the zCDP formula:
    ρ = eps_to_rho(ε, δ=1e-5)
    σ = Δ_2 / √(2ρ) = √(HW / ρ)
  Privacy unit: one release per training image (sample-once, same as CANAL).
  Both methods have N releases; both report per-release ε.

Why this baseline underperforms CANAL:
  CANAL normalises each feature channel to the unit ball, giving sensitivity
  Δ_c = 2/K per channel and σ ≈ 5 at ε=1. PATE/GNMax works in pixel space
  where one teacher's entire H×W mask can flip, giving Δ_2 = √(2HW) ≈ 136
  and σ ≈ 2000 at ε=1. CANAL's feature-space normalisation reduces
  sensitivity by ~70× relative to PATE's pixel-space operation.

Adaptations for segmentation (disclosed):
  1. GNMax applied pixel-by-pixel (binary: background vs foreground).
     Original PATE/GNMax aggregates a single class label per image.
     Segmentation requires H×W predictions; GNMax is applied independently
     at each pixel (h,w) with the same σ.
  2. K=3 teachers (vs n≥100 in the paper).
     With K=3, vote gaps are small (max gap=3) so data-dependent privacy
     bounds are not materially tighter than worst-case. We use worst-case σ.
  3. Student queries private training images (no public data available).
     Original PATE labels a public unlabeled set via the noisy mechanism.

Pipeline:
  1. Train K=3 TinyUNet(base=32) teachers on disjoint patient cohorts (60 ep)
  2. For each ε: compute σ = √(HW/ρ), cache one GNMax hard mask per image
  3. Train TinyUNet(base=16) student for 40 ep with CE + Dice on hard masks

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

K              = 3
EPSILONS       = [1.0, 2.0, 4.0, 8.0]
SEEDS          = [100, 200, 300, 400, 500]
DELTA          = 1e-5
IMG_SIZE       = 96       # H = W
TEACHER_EPOCHS = 60
STUDENT_EPOCHS = 40

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
        return ISICDataset(split, IMG_SIZE)
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset
        return KvasirDataset(split, IMG_SIZE)
    if name == "busi":
        from busi_dataset import BUSIDataset
        return BUSIDataset(split, IMG_SIZE)
    raise ValueError(f"Unknown dataset: {name}")


# =============================================================================
# DP machinery — GNMax with Rényi DP / zCDP
# =============================================================================

def compute_sigma(eps: float, delta: float) -> float:
    """
    Gaussian noise scale for GNMax with sensitivity Δ_2 = √(2HW).

    RDP for Gaussian mechanism: RDP(α) = α·Δ_2²/(2σ²).
    Optimal RDP→(ε,δ)-DP conversion is equivalent to zCDP:
      ρ = eps_to_rho(ε, δ),  σ = Δ_2 / √(2ρ) = √(HW/ρ)
    """
    rho = eps_to_rho(eps, delta)
    return math.sqrt(IMG_SIZE * IMG_SIZE / rho)   # = sqrt(2HW) / sqrt(2rho)


@torch.no_grad()
def precompute_gnmax_cache(teachers, train_ds, sigma, device, seed):
    """
    GNMax cache — Papernot et al. 2018, Algorithm 1, adapted for segmentation.

    For each training image x:
      1. Each teacher k produces a hard binary mask m_k ∈ {0,1}^(H×W)
         via argmax of its logits.
      2. Vote counts: n_1(h,w) = Σ_k m_k(h,w),  n_0 = K − n_1.
      3. GNMax: add independent N(0,σ²) to n_0 and n_1 at each pixel,
         take argmax → hard binary label in {0,1}^(H×W).
      4. Cache the hard mask (LongTensor H×W) keyed by training-set index.
    """
    torch.manual_seed(seed)
    cache  = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)

    for x, _ in loader:
        x = x.to(device)
        hard_masks = []
        for teacher in teachers:
            teacher.eval()
            hard_masks.append(teacher(x).argmax(1))   # B×H×W, {0,1}

        n1      = torch.stack(hard_masks).sum(0).float()   # B×H×W
        n0      = K - n1
        noise_1 = torch.randn_like(n1) * sigma
        noise_0 = torch.randn_like(n0) * sigma
        label   = (n1 + noise_1 > n0 + noise_0).long()    # B×H×W, {0,1}

        for b in range(x.shape[0]):
            cache.append(label[b].cpu())

    return cache


# =============================================================================
# Student training
# =============================================================================

class CachedMaskDataset(Dataset):
    """Wraps a base segmentation dataset with pre-computed GNMax hard masks."""

    def __init__(self, base_ds, cache):
        self.base_ds = base_ds
        self.cache   = cache

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, _ = self.base_ds[idx]
        return x, self.cache[idx]


def seg_loss(logits, hard_mask):
    """CE + Dice against a hard binary mask target."""
    ce    = F.cross_entropy(logits, hard_mask, reduction="mean")
    prob  = logits.softmax(1)[:, 1]
    yf    = (hard_mask == 1).float()
    inter = (prob * yf).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + yf.sum(dim=(1, 2))
    dice  = (1.0 - (2 * inter + 1.0) / (denom + 1.0)).mean()
    return ce + dice


def train_student(base_ds, cache, val_loader, in_ch, bs, device):
    """
    Train TinyUNet(base=16) student on cached GNMax hard masks.
    Returns best val Dice across all epochs.
    """
    ds      = CachedMaskDataset(base_ds, cache)
    loader  = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    student = TinyUNet(in_ch=in_ch, num_classes=2, base=16).to(device)
    opt     = torch.optim.Adam(student.parameters(), lr=1e-3)

    best_dice = 0.0
    for ep in range(STUDENT_EPOCHS):
        student.train()
        for x, hm in loader:
            x  = x.to(device)
            hm = hm.to(device)
            opt.zero_grad()
            seg_loss(student(x), hm).backward()
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

    delta_2 = math.sqrt(2.0 * IMG_SIZE * IMG_SIZE)

    print(f"PATE/GNMax (Papernot et al. 2018) — {args.dataset} | device={DEVICE}")
    print(f"  in_ch={in_ch}  bs={bs}  K={K}  H=W={IMG_SIZE}")
    print(f"  Sensitivity Δ_2 = sqrt(2HW) = {delta_2:.2f}")
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}")
    print(f"  Teacher epochs={TEACHER_EPOCHS}  Student epochs={STUDENT_EPOCHS}")
    print(f"\n  Sigma values per epsilon:")
    for eps in EPSILONS:
        print(f"    ε={eps}  σ={compute_sigma(eps, DELTA):.2f}")

    results = {
        "method":          "PATE/GNMax (Papernot et al. 2018, adapted for segmentation)",
        "paper":           "Scalable Private Learning with PATE, ICLR 2018, arXiv:1802.08908",
        "aggregator":      "GNMax (Gaussian Noisy Max, per-pixel binary voting)",
        "accounting":      "Renyi DP / zCDP, sensitivity=sqrt(2HW)",
        "dataset":         args.dataset,
        "K":               K,
        "img_size":        IMG_SIZE,
        "sensitivity_l2":  delta_2,
        "teacher_epochs":  TEACHER_EPOCHS,
        "student_epochs":  STUDENT_EPOCHS,
        "delta":           DELTA,
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

        for eps in EPSILONS:
            sigma = compute_sigma(eps, DELTA)
            print(f"\n  ε={eps}  σ={sigma:.2f}  (Δ_2={delta_2:.2f})")

            cache = precompute_gnmax_cache(
                teachers, train_ds, sigma, DEVICE,
                seed=seed + int(eps * 100),
            )

            torch.manual_seed(seed)
            np.random.seed(seed)
            dice = train_student(train_ds, cache, val_loader, in_ch, bs, DEVICE)
            all_dices_by_eps[str(eps)].append(dice)
            sigma_by_eps[str(eps)] = float(sigma)
            print(f"  ε={eps}  seed={seed}:  Dice={dice:.4f}  σ={sigma:.2f}")

    print("\n" + "=" * 60)
    for eps in EPSILONS:
        dices = all_dices_by_eps[str(eps)]
        mean  = float(np.mean(dices))
        std   = float(np.std(dices))
        sigma = sigma_by_eps[str(eps)]
        results["sweep"][str(eps)] = {
            "dices": [float(d) for d in dices],
            "mean":  mean,
            "std":   std,
            "sem":   std / math.sqrt(len(dices)),
            "sigma": sigma,
        }
        print(f"ε={eps}: {mean:.4f} ± {std:.4f}  σ={sigma:.2f}")
    print("=" * 60)

    out = HERE / "results" / f"{args.dataset}_pate_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
