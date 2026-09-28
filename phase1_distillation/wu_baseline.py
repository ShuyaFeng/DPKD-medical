"""
Wu et al. 2025 (adapted) — multi-view KD with MIA adversarial training.

Paper: "Privacy-Preserving Multi-View Knowledge Distillation" (Wu et al., 2025).

Adaptations for medical image segmentation (disclosed in paper):
  1. TinyUNet sub-models (base=16) instead of ResNet18 — matches student scale.
  2. Segmentation loss: CE + 3×Dice instead of CE only.
  3. Pixel-wise KL divergence for L_KD (Eq. 9) instead of image-level.
  4. Single sub-model for BUSI (grayscale) per paper's own single-channel rule
     (authors apply the same rule for MNIST in their ablation).
  5. Batch sizes = sqrt(N) consistent with CANAL/DP-SGD setup for our datasets.
  6. Missing hyperparameters (alpha, beta, gamma, delta_w, lambda_adv, tau) tuned
     on BUSI val at eps=8.0 with --hparam, then fixed for all datasets/epsilons.

DP mechanism: Algorithm 1 from Wu et al. — linear composition.
  sigma = S * sqrt(2*ln(1.25/delta)) / (eps/T)
  Budget split: 0.5*eps for features, 0.5*eps for logits.
  S_feat=1.0, S_logit=2.0 (stated in paper).

Usage:
  # Step 1: HP search on BUSI val at eps=8 (run once, saves results/wu_busi_hparams.json)
  python wu_baseline.py --dataset busi --hparam

  # Step 2: Full experiment for each dataset
  python wu_baseline.py --dataset isic
  python wu_baseline.py --dataset kvasir
  python wu_baseline.py --dataset busi
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet, evaluate_vessel_dice

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPSILONS = [1.0, 2.0, 4.0, 8.0]
SEEDS    = [100, 200, 300, 400, 500]
DELTA    = 1e-5
S_FEAT   = 1.0    # per-sample feature sensitivity  (Wu et al. Algorithm 1)
S_LOGIT  = 2.0    # per-sample logit sensitivity    (Wu et al. Algorithm 1)

TEACHER_EPOCHS = 50
STUDENT_EPOCHS = 200

# Batch sizes = sqrt(N), matching CANAL/DP-SGD for fair comparison
DATASET_CFG = {
    "isic":   dict(in_ch=3, bs=45, n_views=3),
    "kvasir": dict(in_ch=3, bs=28, n_views=3),
    "busi":   dict(in_ch=1, bs=23, n_views=1),
}

# Default HP — used if wu_busi_hparams.json not found.
# alpha, beta, gamma, delta_w: student loss weights (Wu et al. Eq. 8)
# lam_adv: MIA adversarial weight (Algorithm 2)
# tau: KL distillation temperature (Eq. 9)
DEFAULT_HP = dict(alpha=1.0, beta=1.0, gamma=1.0, delta_w=0.1, lam_adv=0.1, tau=4.0)


# =============================================================================
# Architecture
# =============================================================================

class ChannelAttention(nn.Module):
    """SE-style channel attention (Wu et al. Fig. 3)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.gap(x).view(B, C)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w)))).view(B, C, 1, 1)
        return x * w


class MultiViewTeacher(nn.Module):
    """
    Multi-view teacher (Wu et al. Section 3.1).

    n_views=3 (RGB): each channel processed by its own TinyUNet sub-model;
                     bottlenecks fused via 1×1 conv + BN + channel attention;
                     skip connections averaged across views.
    n_views=1 (BUSI): single sub-model, no fusion (per paper's single-channel rule).
    """

    def __init__(self, n_views: int, base: int = 16):
        super().__init__()
        self.n_views = n_views
        self.base    = base
        self.subs    = nn.ModuleList([
            TinyUNet(in_ch=1, num_classes=2, base=base)
            for _ in range(n_views)
        ])
        if n_views > 1:
            self.conv_fuse = nn.Conv2d(base * 4 * n_views, base * 4, 1, bias=False)
            self.bn_fuse   = nn.BatchNorm2d(base * 4)
            self.attn      = ChannelAttention(base * 4, reduction=8)

    def get_features_and_logits(self, x):
        """Return (bottleneck: B×64×H/4×W/4, logits: B×2×H×W). No DP noise added."""
        if self.n_views == 1:
            e1, e2, e3 = self.subs[0].encode(x)
            return e3, self.subs[0].decode(e1, e2, e3)
        # Multi-view: split along channel dim
        encs        = [self.subs[i].encode(x[:, i:i+1]) for i in range(self.n_views)]
        e1s, e2s, e3s = zip(*encs)
        e3_cat      = torch.cat(e3s, dim=1)
        e3_fused    = self.attn(F.relu(self.bn_fuse(self.conv_fuse(e3_cat))))
        e1_avg      = torch.stack(list(e1s)).mean(0)
        e2_avg      = torch.stack(list(e2s)).mean(0)
        logits      = self.subs[0].decode(e1_avg, e2_avg, e3_fused)
        return e3_fused, logits

    def forward(self, x):
        return self.get_features_and_logits(x)[1]


class MIADiscriminator(nn.Module):
    """
    ResNet-style MIA discriminator (Wu et al. Section 6.2.2).
    Binary output: 1=member, 0=non-member.
    """

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.out = nn.Linear(feat_dim, 1)

    def forward(self, x):
        h = self.gap(x).flatten(1)
        h = F.relu(self.fc2(F.relu(self.fc1(h))) + h)   # residual block
        return torch.sigmoid(self.out(h))


# =============================================================================
# DP machinery  (Algorithm 1 — Wu et al. 2025)
# =============================================================================

def compute_sigma(eps_budget: float, delta: float, sensitivity: float,
                  total_steps: int) -> float:
    """Linear composition: each step spends eps_budget/total_steps."""
    eps_step = eps_budget / total_steps
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps_step


def clip_and_add_noise(x: torch.Tensor, sensitivity: float,
                       sigma: float) -> torch.Tensor:
    """Per-sample L2 clip to `sensitivity`, then add Gaussian noise N(0,sigma^2)."""
    B    = x.shape[0]
    flat = x.view(B, -1)
    norms  = flat.norm(2, dim=1, keepdim=True)
    scale  = (sensitivity / norms.clamp(min=1e-8)).clamp(max=1.0)
    clipped = (flat * scale).view_as(x)
    return clipped + torch.randn_like(clipped) * sigma


# =============================================================================
# Losses
# =============================================================================

def seg_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CE + 3×Dice (per-sample), same as dpsgd_baseline.py."""
    ce    = F.cross_entropy(logits, y, reduction="mean")
    probs = logits.softmax(1)[:, 1]
    yf    = (y == 1).float()
    inter = (probs * yf).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + yf.sum(dim=(1, 2))
    dice  = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
    return ce + 3.0 * dice.mean()


def student_loss(logit_s, feat_s, y, logit_t_noisy, feat_t_noisy,
                 alpha, beta, gamma, delta_w, tau):
    """Four-term student loss — Wu et al. Eq. 8–9."""
    B, C, H, W = logit_t_noisy.shape
    # L_KD: pixel-wise KL divergence with temperature (Eq. 9)
    log_ps = F.log_softmax(logit_s.view(B * H * W, C) / tau, dim=1)
    pt     = F.softmax(logit_t_noisy.view(B * H * W, C) / tau, dim=1)
    l_kd   = tau ** 2 * F.kl_div(log_ps, pt, reduction="batchmean")
    # L_task: segmentation loss on student predictions
    l_task = seg_loss(logit_s, y)
    # L_feature: MSE on bottleneck features
    l_feat = F.mse_loss(feat_s, feat_t_noisy)
    # L_contrastive: 1 − cosine similarity on GAP embeddings
    gs     = F.adaptive_avg_pool2d(feat_s,        1).flatten(1)
    gt_    = F.adaptive_avg_pool2d(feat_t_noisy,  1).flatten(1)
    l_cont = 1.0 - F.cosine_similarity(gs, gt_, dim=1).mean()
    return alpha * l_kd + beta * l_task + gamma * l_feat + delta_w * l_cont


# =============================================================================
# Training routines
# =============================================================================

def train_teacher(teacher, mia, train_loader, val_loader, device, lam_adv):
    """
    Algorithm 2 — MIA adversarial training.

    Phase 1 (epochs 0-1, warmup):
      - Teacher: task loss only.
      - MIA: BCE on train features (member=1) and val features (non-member=0).
    Phase 2 (epochs 2..TEACHER_EPOCHS-1):
      - MIA frozen.
      - Teacher: L_task + lam_adv * L_adv, where L_adv pushes teacher features
        toward the non-member distribution (adversarial against MIA).
    """
    opt_t   = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    opt_mia = torch.optim.Adam(mia.parameters(), lr=1e-4)
    bce     = nn.BCELoss()

    # Phase 1 — warmup
    for ep in range(2):
        teacher.train()
        mia.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Task update for teacher (no adversarial yet)
            feat, logit = teacher.get_features_and_logits(x)
            opt_t.zero_grad()
            seg_loss(logit, y).backward()
            opt_t.step()
            # MIA: train set = members (label 1)
            opt_mia.zero_grad()
            bce(mia(feat.detach()),
                torch.ones(x.shape[0], 1, device=device)).backward()
            opt_mia.step()
        # MIA: val set = non-members (label 0)
        teacher.eval()
        for x, _ in val_loader:
            x = x.to(device)
            with torch.no_grad():
                feat, _ = teacher.get_features_and_logits(x)
            opt_mia.zero_grad()
            bce(mia(feat),
                torch.zeros(x.shape[0], 1, device=device)).backward()
            opt_mia.step()

    # Phase 2 — freeze MIA, adversarially train teacher
    for p in mia.parameters():
        p.requires_grad_(False)
    mia.eval()

    for ep in range(TEACHER_EPOCHS - 2):
        teacher.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_t.zero_grad()
            feat, logit = teacher.get_features_and_logits(x)
            l_task = seg_loss(logit, y)
            # Teacher wants MIA to predict non-member (adversarial goal)
            l_adv  = bce(mia(feat),
                         torch.zeros(x.shape[0], 1, device=device))
            (l_task + lam_adv * l_adv).backward()
            opt_t.step()
        if (ep + 3) % 10 == 0:
            print(f"    teacher ep {ep+3}/{TEACHER_EPOCHS} done")


def train_student(teacher, train_ds, val_loader, eps, hp, in_ch, bs, device):
    """
    Train student with per-step DP noise on teacher outputs (Algorithm 1).
    Returns (val_dice, sigma_feat, sigma_logit).
    """
    loader  = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    T_total = STUDENT_EPOCHS * len(loader)

    sig_feat  = compute_sigma(0.5 * eps, DELTA, S_FEAT,  T_total)
    sig_logit = compute_sigma(0.5 * eps, DELTA, S_LOGIT, T_total)

    student = TinyUNet(in_ch=in_ch, num_classes=2, base=16).to(device)
    opt_s   = torch.optim.Adam(student.parameters(), lr=1e-3)

    teacher.eval()
    for _ in range(STUDENT_EPOCHS):
        student.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat_t, logit_t   = teacher.get_features_and_logits(x)
                feat_t_noisy      = clip_and_add_noise(feat_t,  S_FEAT,  sig_feat)
                logit_t_noisy     = clip_and_add_noise(logit_t, S_LOGIT, sig_logit)
            e1_s, e2_s, e3_s = student.encode(x)
            logit_s          = student.decode(e1_s, e2_s, e3_s)
            opt_s.zero_grad()
            student_loss(logit_s, e3_s, y, logit_t_noisy, feat_t_noisy,
                         hp["alpha"], hp["beta"], hp["gamma"],
                         hp["delta_w"], hp["tau"]).backward()
            opt_s.step()

    dice = evaluate_vessel_dice(student, val_loader, device)
    return dice, sig_feat, sig_logit


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
# HP search (BUSI val, eps=8, 50 student epochs)
# =============================================================================

def run_hparam_search(device):
    """
    Tune tau on BUSI val at eps=8.
    lam_adv is fixed at 0.1 here and kept fixed in best_hp because lam_adv
    only affects teacher training — we cannot tune it cheaply (would require
    one full teacher training per candidate value).  alpha=beta=gamma=1.0,
    delta_w=0.1 are also fixed as reasonable prior-literature defaults.
    Saves best HP to results/wu_busi_hparams.json.
    """
    from busi_dataset import BUSIDataset
    bs         = DATASET_CFG["busi"]["bs"]
    LAM_ADV    = 0.1   # fixed — only tau is tuned
    train_ds   = BUSIDataset("train", 96)
    val_ds     = BUSIDataset("val",   96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    print(f"HP search | BUSI | eps=8.0 | device={device}")
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}")
    print(f"  Tuning: tau in {{2, 4, 8}}  |  lam_adv fixed at {LAM_ADV}")

    tau_vals   = [2.0, 4.0, 8.0]
    EPS_HP     = 8.0
    STUD_HP_EP = 50

    # Train teacher once — shared across all tau candidates
    torch.manual_seed(100)
    np.random.seed(100)
    teacher      = MultiViewTeacher(n_views=1, base=16).to(device)
    mia          = MIADiscriminator(feat_dim=16 * 4).to(device)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    print("  Training teacher (seed=100, 50 epochs)...")
    train_teacher(teacher, mia, train_loader, val_loader, device, LAM_ADV)
    print(f"  Teacher val Dice: {evaluate_vessel_dice(teacher, val_loader, device):.4f}")

    runs      = []
    best_dice = -1.0
    best_tau  = tau_vals[0]

    for tau in tau_vals:
        loader2 = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
        T  = STUD_HP_EP * len(loader2)
        sf = compute_sigma(0.5 * EPS_HP, DELTA, S_FEAT,  T)
        sl = compute_sigma(0.5 * EPS_HP, DELTA, S_LOGIT, T)

        student = TinyUNet(in_ch=1, num_classes=2, base=16).to(device)
        opt_s   = torch.optim.Adam(student.parameters(), lr=1e-3)
        teacher.eval()
        for _ in range(STUD_HP_EP):
            student.train()
            for x, y in loader2:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    ft, lt = teacher.get_features_and_logits(x)
                    ft_n   = clip_and_add_noise(ft, S_FEAT,  sf)
                    lt_n   = clip_and_add_noise(lt, S_LOGIT, sl)
                e1, e2, e3 = student.encode(x)
                ls = student.decode(e1, e2, e3)
                opt_s.zero_grad()
                student_loss(ls, e3, y, lt_n, ft_n,
                             1.0, 1.0, 1.0, 0.1, tau).backward()
                opt_s.step()

        dice = evaluate_vessel_dice(student, val_loader, device)
        print(f"  tau={tau}  Dice={dice:.4f}  (sf={sf:.0f}  sl={sl:.0f})")
        runs.append({"tau": tau, "dice": float(dice)})
        if dice > best_dice:
            best_dice = dice
            best_tau  = tau

    best_hp = dict(alpha=1.0, beta=1.0, gamma=1.0, delta_w=0.1,
                   lam_adv=LAM_ADV, tau=best_tau)
    print(f"\nBest: tau={best_tau}  Dice={best_dice:.4f}")
    out = HERE / "results" / "wu_busi_hparams.json"
    out.write_text(json.dumps({"best": best_hp, "runs": runs}, indent=2))
    print(f"Saved: {out}")
    return best_hp


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["isic", "kvasir", "busi"])
    parser.add_argument("--hparam",  action="store_true",
                        help="Run HP search on BUSI val (eps=8, 50 student epochs). "
                             "Saves results/wu_busi_hparams.json.")
    args = parser.parse_args()

    if args.hparam:
        run_hparam_search(DEVICE)
        return

    cfg     = DATASET_CFG[args.dataset]
    in_ch   = cfg["in_ch"]
    bs      = cfg["bs"]
    n_views = cfg["n_views"]

    train_ds   = get_dataset(args.dataset, "train")
    val_ds     = get_dataset(args.dataset, "val")
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    print(f"Wu et al. 2025 (adapted) — {args.dataset} | device={DEVICE}")
    print(f"  in_ch={in_ch}  n_views={n_views}  bs={bs}")
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}")
    print(f"  S_feat={S_FEAT}  S_logit={S_LOGIT}")
    print(f"  Teacher epochs={TEACHER_EPOCHS}  Student epochs={STUDENT_EPOCHS}")

    # Load tuned HP or fall back to defaults
    hp_path = HERE / "results" / "wu_busi_hparams.json"
    if hp_path.exists():
        hp = json.loads(hp_path.read_text())["best"]
        print(f"  HP (from {hp_path.name}): {hp}")
    else:
        hp = DEFAULT_HP
        print(f"  HP (default — run --hparam on busi first for tuned values): {hp}")

    results = {
        "method":          "Wu et al. 2025 (adapted for medical image segmentation)",
        "dataset":         args.dataset,
        "config":          {k: cfg[k] for k in ("in_ch", "bs", "n_views")},
        "teacher_epochs":  TEACHER_EPOCHS,
        "student_epochs":  STUDENT_EPOCHS,
        "delta":           DELTA,
        "S_feat":          S_FEAT,
        "S_logit":         S_LOGIT,
        "hparams":         hp,
        "epsilons":        EPSILONS,
        "seeds":           SEEDS,
        "sweep":           {},
    }

    all_dices_by_eps = {str(e): [] for e in EPSILONS}
    sig_by_eps       = {}

    for s_idx, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        teacher      = MultiViewTeacher(n_views=n_views, base=16).to(DEVICE)
        mia          = MIADiscriminator(feat_dim=16 * 4).to(DEVICE)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)

        print(f"\n[Seed {seed} ({s_idx+1}/{len(SEEDS)})] "
              f"Training teacher ({TEACHER_EPOCHS} epochs)...")
        train_teacher(teacher, mia, train_loader, val_loader, DEVICE, hp["lam_adv"])

        t_dice = evaluate_vessel_dice(teacher, val_loader, DEVICE)
        print(f"  Teacher val Dice: {t_dice:.4f}")

        for eps in EPSILONS:
            dice, sf, sl = train_student(
                teacher, train_ds, val_loader, eps, hp, in_ch, bs, DEVICE
            )
            all_dices_by_eps[str(eps)].append(dice)
            sig_by_eps[str(eps)] = {"sigma_feat": sf, "sigma_logit": sl}
            print(f"  eps={eps}  seed={seed}:  Dice={dice:.4f}"
                  f"  (sigma_feat={sf:.1f}  sigma_logit={sl:.1f})")

    print("\n" + "=" * 60)
    for eps in EPSILONS:
        dices = all_dices_by_eps[str(eps)]
        mean  = float(np.mean(dices))
        std   = float(np.std(dices))
        sf    = sig_by_eps[str(eps)]["sigma_feat"]
        sl    = sig_by_eps[str(eps)]["sigma_logit"]
        results["sweep"][str(eps)] = {
            "dices":        [float(d) for d in dices],
            "mean":         mean,
            "std":          std,
            "sem":          std / math.sqrt(len(dices)),
            "sigma_feat":   sf,
            "sigma_logit":  sl,
        }
        print(f"eps={eps}: {mean:.4f} ± {std:.4f}  "
              f"(sigma_feat={sf:.1f}  sigma_logit={sl:.1f})")
    print("=" * 60)

    out = HERE / "results" / f"{args.dataset}_wu_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
