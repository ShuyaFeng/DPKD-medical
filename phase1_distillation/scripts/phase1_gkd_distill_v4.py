#!/usr/bin/env python
"""
GKD Student Distillation v4 — DP-Honest Pipeline with Normalised Alpha Loss.

Built on top of Dr. Feng's phase1_gkd_distill_v2.py. Keeps her correct
DP structure and adds the normalised-alpha loss from our v3.

What Dr. Feng's v2 fixed (all carried forward here)
----------------------------------------------------
  1. Public-proxy caps: caps loaded from CSV computed on HRF/STARE/CHASE
     (ZERO privacy cost on DRIVE).
  2. Sample-once threat model (--precompute-noise flag): each DRIVE training
     image's noisy bottleneck is computed ONCE before training, cached, and
     reused. Every patient contributes exactly ONE release → reported epsilon
     equals the true user-level epsilon (vs 400-1800x blowup in prior work).
  3. Factor-of-2 fix in sigma: kappa uses 2*rho in denominator (Theorem 1).
  4. Proper _img_id() cache keying for reliable sample-once lookup.

What v3 / v4 adds on top of Dr. Feng's v2
-------------------------------------------
  - Normalised alpha loss (TRUE percentage split between task and teacher):

        scale         = t_loss.detach() / (f_loss.detach() + 1e-8)
        f_loss_scaled = f_loss * scale
        total         = (1 - alpha) * t_loss + alpha * f_loss_scaled

    alpha = 0.7 means exactly 70% of gradient signal from noisy teacher,
    30% from GT — regardless of raw magnitudes of t_loss and f_loss.
    (Dr. Feng's v2 uses lambda_feat which is NOT a true percentage.)

  - Extended validation metrics: mDice, vessel_Dice, mIoU, pixel_acc, mean_acc.
  - Alpha range extended to [0.0, 1.0] inclusive (allows alpha=1.0 ablation).

NOTE: Budget split (Eq. 7 from the paper) is NOT implemented in the
training pipeline. budget_split_analysis.py handles that analytically
for the paper. Importance and caps are used CLEAN from the public CSV.

Pipeline
--------
  PRE-TRAINING (sample-once, once per unique training image):
    image → teacher encoder → clean bottleneck
    → clip + normalise (using public caps)
    → add Gaussian noise (sigma from clean public importance, full epsilon)
    → denormalise → store in cache dict keyed by img_path

  TRAINING LOOP (40,000 iterations):
    load cached noisy bottleneck (no new noise, no new teacher query)
    student encoder → student bottleneck
    adapter (1x1 conv) → projected student (512→1024)
    feature_loss = MSE(projected_student, cached_noisy_bn)
    student decoder → logits
    task_loss = CE + 3×Dice (vs GT)
    total = (1-alpha)*task_loss + alpha*(feature_loss*scale)
    backward + update student + adapter

Run:
  python phase1_gkd_distill_v4.py \
    --noise-type channel_WF --epsilon 8 --alpha 0.7 --seed 0 \
    --public-caps-csv .../public_caps.csv \
    --importance-csv  .../public_importance.csv \
    --precompute-noise

Total sweep: 2 noise types × 4 epsilons × 5 alphas × 3 seeds = 120 jobs
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import PixelData

try:
    from mmengine.model import revert_sync_batchnorm
except Exception:
    revert_sync_batchnorm = None

from mmseg.registry import DATASETS, MODELS


# ---------------------------------------------------------------------------
# Privacy accounting
# ---------------------------------------------------------------------------

def eps_to_rho(eps: float, delta: float = 1e-5) -> float:
    """Bun-Steinke 2016: (eps, delta)-DP → rho-zCDP."""
    log_inv_d = math.log(1.0 / delta)
    b = 2.0 * math.sqrt(log_inv_d)
    discriminant = b * b + 4.0 * eps
    sqrt_rho = (-b + math.sqrt(discriminant)) / 2.0
    return sqrt_rho * sqrt_rho


# ---------------------------------------------------------------------------
# Noise allocation — FACTOR-OF-2 FIXED (Dr. Feng's fix, carried forward)
# ---------------------------------------------------------------------------

def waterfilling_sigma(
    deltas: torch.Tensor,
    importances: torch.Tensor,
    rdp_budget: float,
    eps_s: float = 1e-12,
) -> torch.Tensor:
    """
    Channel-WF allocation (Theorem 1).
        sigma_c = kappa * sqrt(delta_c) / s_c^{1/4}
        kappa   = sqrt( sum_c(delta_c * sqrt(s_c)) / (2 * rdp_budget) )
    Factor-of-2 in denominator fixed vs older scripts.
    """
    s = importances.clamp(min=eps_s)
    kappa = ((deltas * s.sqrt()).sum() / (2.0 * rdp_budget)).sqrt()
    return kappa * deltas.sqrt() / s.pow(0.25)


def uniform_sigma(deltas: torch.Tensor, rdp_budget: float) -> torch.Tensor:
    """
    Uniform: same sigma for every channel.
    Factor-of-2 in denominator fixed vs older scripts.
    """
    sigma = (deltas.pow(2).sum() / (2.0 * rdp_budget)).sqrt()
    return sigma.expand(deltas.shape[0]).clone()


# ---------------------------------------------------------------------------
# MMSeg utilities
# ---------------------------------------------------------------------------

def set_data_root(cfg: Config, data_root: str) -> None:
    if hasattr(cfg, "data_root"):
        cfg.data_root = data_root
    for key in ["val_dataloader", "test_dataloader", "train_dataloader"]:
        if not hasattr(cfg, key):
            continue
        dl = cfg[key]
        ds = dl.get("dataset", None)
        if ds is None:
            continue
        if ds.get("type", None) == "RepeatDataset" and "dataset" in ds:
            ds["dataset"]["data_root"] = data_root
        elif "data_root" in ds:
            ds["data_root"] = data_root


def build_model(cfg: Config, checkpoint: str, device: str, frozen: bool = False):
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = MODELS.build(cfg.model)
    if revert_sync_batchnorm is not None:
        model = revert_sync_batchnorm(model)
    load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device)
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        model.train()
    return model


def build_loader(cfg: Config, split: str, num_workers: int,
                 shuffle: Optional[bool] = None) -> DataLoader:
    key = "train_dataloader" if split == "train" else "val_dataloader"
    dataset = DATASETS.build(cfg[key]["dataset"])
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=cfg[key].get("batch_size", 4),
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=pseudo_collate,
        drop_last=(split == "train"),
    )


def move_to_device(model, data: Dict, device: str, training: bool) -> Dict:
    data = model.data_preprocessor(data, training=training)
    data["inputs"] = data["inputs"].to(device)
    return data


def pad_to_divisor(data: Dict, divisor: int = 16,
                   ignore_index: int = 255) -> Dict:
    inputs = data["inputs"]
    _, _, h, w = inputs.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return data
    new_h, new_w = h + pad_h, w + pad_w
    data["inputs"] = F.pad(inputs, (0, pad_w, 0, pad_h), value=0)
    for sample in data["data_samples"]:
        padded_gt = F.pad(
            sample.gt_sem_seg.data, (0, pad_w, 0, pad_h), value=ignore_index)
        sample.gt_sem_seg = PixelData(data=padded_gt)
        sample.set_metainfo({"pad_shape": (new_h, new_w),
                              "img_shape": (new_h, new_w)})
    return data


def get_gt(data_samples: Sequence, device: str) -> torch.Tensor:
    return torch.stack([
        s.gt_sem_seg.data.squeeze(0).long().to(device)
        for s in data_samples
    ])


# ---------------------------------------------------------------------------
# U-Net manual forward
# ---------------------------------------------------------------------------

def unet_encoder(backbone, x: torch.Tensor) -> List[torch.Tensor]:
    enc_outs = []
    for enc in backbone.encoder:
        x = enc(x)
        enc_outs.append(x)
    return enc_outs


def unet_decoder(backbone, enc_outs: Sequence[torch.Tensor],
                 bottleneck: torch.Tensor) -> List[torch.Tensor]:
    x = bottleneck
    dec_outs = [x]
    for i in reversed(range(len(backbone.decoder))):
        x = backbone.decoder[i](enc_outs[i], x)
        dec_outs.append(x)
    return dec_outs


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def clip_and_normalise(bottleneck: torch.Tensor,
                       caps: torch.Tensor) -> torch.Tensor:
    """Per-channel L2 clipping + normalisation → every channel norm ≤ 1."""
    B, C, H, W = bottleneck.shape
    norms = bottleneck.flatten(2).norm(dim=2)
    scale = (caps.view(1, C) / norms.clamp(min=1e-12)).clamp(max=1.0)
    clipped = bottleneck * scale.view(B, C, 1, 1)
    return clipped / caps.view(1, C, 1, 1).clamp(min=1e-12)


def denormalise(normalised: torch.Tensor, caps: torch.Tensor) -> torch.Tensor:
    _, C, H, W = normalised.shape
    return normalised * caps.view(1, C, 1, 1)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BottleneckAdapter(nn.Module):
    """1×1 conv: student_channels → teacher_channels."""
    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(student_channels, teacher_channels,
                              kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out",
                                nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss_fn(logits: torch.Tensor, gt: torch.Tensor,
                 ignore_index: int = 255, eps: float = 1e-7) -> torch.Tensor:
    valid = (gt != ignore_index)
    prob  = torch.softmax(logits, dim=1)[:, 1]
    gt_f  = (gt == 1).float()
    prob  = prob * valid.float()
    gt_f  = gt_f * valid.float()
    inter = (prob * gt_f).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + gt_f.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def task_loss_fn(logits: torch.Tensor, gt: torch.Tensor,
                 ignore_index: int = 255) -> torch.Tensor:
    """CE + 3×Dice — matches teacher training config."""
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:],
                               mode="bilinear", align_corners=False)
    ce = F.cross_entropy(logits, gt, ignore_index=ignore_index)
    di = dice_loss_fn(logits, gt, ignore_index)
    return ce + 3.0 * di


def feature_matching_loss(projected_student: torch.Tensor,
                           noisy_teacher_bn: torch.Tensor) -> torch.Tensor:
    """MSE between adapter-projected student and cached noisy teacher bn."""
    return F.mse_loss(projected_student, noisy_teacher_bn.detach(),
                      reduction="mean")


def alpha_loss(t_loss: torch.Tensor, f_loss: torch.Tensor,
               alpha: float) -> torch.Tensor:
    """
    Normalised alpha loss — TRUE percentage split.

    scale = t_loss / f_loss brings feature loss to task loss magnitude,
    so alpha genuinely controls the percentage contribution:
      alpha=0.7 → 70% noisy teacher, 30% GT
      alpha=1.0 → 100% teacher (ablation)
      alpha=0.0 → task-only baseline
    """
    if alpha == 0.0:
        return t_loss
    if alpha == 1.0:
        scale = t_loss.detach() / (f_loss.detach() + 1e-8)
        return f_loss * scale
    scale = t_loss.detach() / (f_loss.detach() + 1e-8)
    return (1.0 - alpha) * t_loss + alpha * (f_loss * scale)


# ---------------------------------------------------------------------------
# Extended metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(logits: torch.Tensor, gt: torch.Tensor,
                        num_classes: int = 2, ignore_index: int = 255,
                        eps: float = 1e-7) -> Dict[str, float]:
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:],
                               mode="bilinear", align_corners=False)
    pred  = logits.argmax(dim=1)
    valid = gt != ignore_index
    dices, ious, class_accs = [], [], []
    for cls in range(num_classes):
        pred_c = (pred == cls) & valid
        gt_c   = (gt   == cls) & valid
        dims   = tuple(range(1, gt.ndim))
        inter  = (pred_c & gt_c).sum(dim=dims).float()
        union  = (pred_c | gt_c).sum(dim=dims).float()
        denom_d = pred_c.sum(dim=dims).float() + gt_c.sum(dim=dims).float()
        dices.append(((2.0 * inter + eps) / (denom_d + eps)).mean().item())
        ious.append(((inter + eps) / (union + eps)).mean().item())
        gt_count = gt_c.sum(dim=dims).float()
        class_accs.append(((inter + eps) / (gt_count + eps)).mean().item())
    correct   = ((pred == gt) & valid).sum().float()
    total     = valid.sum().float()
    return {
        "mDice":       float(np.mean(dices)),
        "vessel_Dice": dices[1],
        "mIoU":        float(np.mean(ious)),
        "pixel_acc":   (correct / (total + eps)).item(),
        "mean_acc":    float(np.mean(class_accs)),
    }


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_importances(csv_path: str) -> torch.Tensor:
    """Load importance scores — column: gradient_abs_mean."""
    vals = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["gradient_abs_mean"]))
    return torch.tensor(vals, dtype=torch.float32).clamp(min=1e-12)


def load_caps_from_csv(csv_path: str) -> torch.Tensor:
    """Load per-channel caps — column: cap_norm."""
    vals = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["cap_norm"]))
    return torch.tensor(vals, dtype=torch.float32).clamp(min=1e-12)


# ---------------------------------------------------------------------------
# Sample-once cache (Dr. Feng's implementation from v2)
# ---------------------------------------------------------------------------

def _img_id(data_sample) -> str:
    """Stable per-image identifier from an mmseg DataSample's metainfo."""
    meta = getattr(data_sample, "metainfo", {}) or {}
    for key in ("img_path", "img_id", "filename", "ori_filename"):
        v = meta.get(key)
        if v:
            return str(v)
    return f"obj_{id(data_sample)}"


def _unwrap_dataset(ds):
    """Strip RepeatDataset/ConcatDataset wrappers to count unique items."""
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def precompute_noisy_teacher_cache(
    teacher,
    train_loader,
    caps: torch.Tensor,
    sigma: torch.Tensor,
    device: str,
    ignore_index: int,
    pad_divisor: int,
    max_passes: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Run teacher ONCE per unique DRIVE training image, add noise, cache on CPU.
    Multi-pass to handle RepeatDataset / shuffled loaders.
    """
    n_unique = len(_unwrap_dataset(train_loader.dataset))
    cache: Dict[str, torch.Tensor] = {}
    teacher.eval()

    print(f"\n[sample-once] precomputing noisy teacher bottleneck for "
          f"{n_unique} unique training images...")

    with torch.no_grad():
        for pass_idx in range(max_passes):
            cache_before = len(cache)
            for raw_data in train_loader:
                data = move_to_device(teacher, raw_data, device, training=True)
                data = pad_to_divisor(data, pad_divisor, ignore_index)
                enc_outs   = unet_encoder(teacher.backbone, data["inputs"])
                t_bn_clean = enc_outs[-1]
                t_bn_norm  = clip_and_normalise(t_bn_clean, caps)
                B, C, H, W = t_bn_norm.shape
                noise      = torch.randn(B, C, H, W, device=device) \
                             * sigma.view(1, C, 1, 1)
                t_bn_noisy = denormalise(t_bn_norm + noise, caps)
                for b, ds in enumerate(data["data_samples"]):
                    key = _img_id(ds)
                    if key not in cache:
                        cache[key] = t_bn_noisy[b].detach().cpu().clone()
                if len(cache) >= n_unique:
                    break
            if len(cache) >= n_unique:
                break
            if len(cache) == cache_before:
                print(f"[sample-once] WARNING: cache stopped growing at "
                      f"{len(cache)} (< {n_unique}).")
                break

    print(f"[sample-once] cached {len(cache)} tensors  "
          f"(shape={next(iter(cache.values())).shape if cache else 'n/a'}).")
    return cache


def lookup_cached(data_samples, cache: Dict[str, torch.Tensor],
                  device: str) -> torch.Tensor:
    """Assemble a batch of cached noisy bottlenecks by image id."""
    tensors = []
    for ds in data_samples:
        key = _img_id(ds)
        if key not in cache:
            raise KeyError(f"[sample-once] {key!r} not in cache.")
        tensors.append(cache[key].to(device, non_blocking=True))
    return torch.stack(tensors, dim=0)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GKD v4: Dr. Feng's DP structure + normalised alpha loss."
    )
    # Model paths
    p.add_argument("--teacher-config",     required=True)
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--student-config",     required=True)
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--data-root",          required=True)

    # Public proxy CSVs
    p.add_argument("--public-caps-csv",  required=True,
                   help="CSV from compute_public_proxy.py — column: cap_norm")
    p.add_argument("--importance-csv",   required=True,
                   help="CSV from compute_public_proxy.py — column: gradient_abs_mean")

    # Output
    p.add_argument("--out-dir",          required=True)

    # Noise
    p.add_argument("--noise-type",       required=True,
                   choices=["uniform", "channel_WF"])
    p.add_argument("--epsilon",          type=float, required=True)
    p.add_argument("--alpha",            type=float, required=True,
                   help="Teacher signal fraction [0.0, 1.0]. "
                        "0.7 = 70%% noisy teacher, 30%% GT.")

    # Sample-once (use this flag for DP-honest runs)
    p.add_argument("--precompute-noise", action="store_true",
                   help="Sample-once: compute noisy bottleneck once per image "
                        "before training. Makes reported epsilon = true user "
                        "epsilon. Recommended for all paper experiments.")

    # DP parameters
    p.add_argument("--K",                type=int,   default=1)
    p.add_argument("--delta-dp",         type=float, default=1e-5)

    # Training
    p.add_argument("--max-iters",        type=int,   default=40000)
    p.add_argument("--val-interval",     type=int,   default=4000)
    p.add_argument("--device",           default="cuda:0")
    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--pad-divisor",      type=int,   default=16)
    p.add_argument("--ignore-index",     type=int,   default=255)
    p.add_argument("--seed",             type=int,   default=0)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {args.alpha}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    # ------------------------------------------------------------------
    # Build teacher (frozen) and student
    # ------------------------------------------------------------------
    t_cfg = Config.fromfile(args.teacher_config)
    s_cfg = Config.fromfile(args.student_config)
    set_data_root(t_cfg, args.data_root)
    set_data_root(s_cfg, args.data_root)

    print("Building teacher (frozen)...")
    teacher = build_model(t_cfg, args.teacher_checkpoint, device, frozen=True)

    print("Building student (from baseline checkpoint)...")
    student = build_model(s_cfg, args.student_checkpoint, device, frozen=False)

    student_bn_channels = s_cfg.model.backbone.base_channels * (2 ** 4)  # 512
    teacher_bn_channels = t_cfg.model.backbone.base_channels * (2 ** 4)  # 1024

    adapter = BottleneckAdapter(student_bn_channels, teacher_bn_channels).to(device)
    print(f"Adapter: {student_bn_channels} → {teacher_bn_channels} channels "
          f"({sum(p.numel() for p in adapter.parameters())} params)")

    # ------------------------------------------------------------------
    # Load public caps and importance (ZERO DRIVE privacy cost)
    # ------------------------------------------------------------------
    print(f"\nLoading public caps from:       {args.public_caps_csv}")
    print(f"Loading public importance from: {args.importance_csv}")

    caps       = load_caps_from_csv(args.public_caps_csv).to(device)
    importance = load_importances(args.importance_csv).to(device)

    print(f"Caps       : min={caps.min():.3f}  max={caps.max():.3f}  "
          f"mean={caps.mean():.3f}")
    print(f"Importance : min={importance.min():.3e}  max={importance.max():.3e}  "
          f"mean={importance.mean():.3e}")
    print(f"Importance max/min ratio = {importance.max()/importance.min():.1f}")
    print("[public-proxy] caps and importance cost ZERO DRIVE privacy budget.")

    # ------------------------------------------------------------------
    # Sensitivity and sigma (full epsilon goes to bottleneck release)
    # ------------------------------------------------------------------
    # After per-channel L2 normalisation the replace-one sensitivity is
    # the data-independent constant 2/K (Bun-Steinke 2016, paper §3.2).
    # Caps come from PUBLIC data → not charged to DRIVE budget.
    # Importance comes from PUBLIC data → not charged to DRIVE budget.
    # The full epsilon is therefore spent on the bottleneck release.
    deltas = torch.full_like(caps, 2.0 / args.K)

    rho = eps_to_rho(args.epsilon, args.delta_dp)
    if args.noise_type == "channel_WF":
        sigma = waterfilling_sigma(deltas, importance, rho)
    else:
        sigma = uniform_sigma(deltas, rho)

    print(f"\nNoise type : {args.noise_type}")
    print(f"Epsilon    : {args.epsilon}  rho={rho:.5f}  "
          f"(full budget on bottleneck release)")
    print(f"Sigma      : min={sigma.min():.4f}  max={sigma.max():.4f}  "
          f"mean={sigma.mean():.4f}")

    if args.noise_type == "channel_WF":
        top20 = torch.argsort(importance, descending=True)[:20]
        bot20 = torch.argsort(importance, descending=False)[:20]
        print(f"  WF top-20 important sigma  : {sigma[top20].mean():.4f}  ← SMALLER")
        print(f"  WF bot-20 unimportant sigma: {sigma[bot20].mean():.4f}  ← LARGER")
        if sigma[top20].mean() < sigma[bot20].mean():
            print("  ✅ Water-filling correct")
        else:
            print("  ⚠️  Water-filling inverted — check importance scores")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    train_loader = build_loader(s_cfg, "train", args.num_workers)
    val_loader   = build_loader(s_cfg, "val",   args.num_workers)

    # ------------------------------------------------------------------
    # Sample-once cache (Dr. Feng's approach)
    # ------------------------------------------------------------------
    noisy_cache: Optional[Dict[str, torch.Tensor]] = None
    if args.precompute_noise:
        print("\n" + "=" * 70)
        print("SAMPLE-ONCE: precomputing noisy bottleneck cache.")
        print("Each DRIVE training image released exactly ONCE.")
        print("Reported epsilon = true user-level epsilon.")
        print("=" * 70)
        noisy_cache = precompute_noisy_teacher_cache(
            teacher=teacher,
            train_loader=train_loader,
            caps=caps,
            sigma=sigma,
            device=device,
            ignore_index=args.ignore_index,
            pad_divisor=args.pad_divisor,
        )
    else:
        print("\n[per-iteration] WARNING: fresh noise added every iteration.")
        print("  True user-level epsilon >> reported epsilon.")
        print("  Use --precompute-noise for DP-honest runs.")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    all_params = list(student.parameters()) + list(adapter.parameters())
    optimizer  = torch.optim.SGD(
        all_params, lr=0.01, momentum=0.9, weight_decay=0.0005)

    def poly_lr(iteration: int) -> float:
        return (1.0 - iteration / args.max_iters) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    release_model = ("sample-once" if args.precompute_noise
                     else "per-iteration (NOT DP-honest)")

    print("\n" + "=" * 70)
    print(f"v4: DP-honest pipeline | noise={args.noise_type} | "
          f"eps={args.epsilon} | seed={args.seed}")
    print(f"alpha={args.alpha} → {args.alpha*100:.0f}% noisy teacher, "
          f"{(1-args.alpha)*100:.0f}% GT")
    print(f"release model: {release_model}")
    print(f"max_iters={args.max_iters} | val_interval={args.val_interval}")
    print("=" * 70)

    best_mdice = 0.0
    best_iter  = 0
    history    = []

    train_iter = iter(train_loader)
    iteration  = 0

    student.train()
    adapter.train()
    teacher.eval()

    while iteration < args.max_iters:

        try:
            raw_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            raw_data   = next(train_iter)

        data   = move_to_device(student, raw_data, device, training=True)
        data   = pad_to_divisor(data, args.pad_divisor, args.ignore_index)
        inputs = data["inputs"]
        gt     = get_gt(data["data_samples"], device)

        # ==============================================================
        # TEACHER PATH
        # ==============================================================
        if noisy_cache is not None:
            # Sample-once: load from cache (no new noise, no teacher query)
            t_bn_noisy = lookup_cached(data["data_samples"], noisy_cache, device)
        else:
            # Per-iteration: add fresh noise every step
            with torch.no_grad():
                t_enc      = unet_encoder(teacher.backbone, inputs)
                t_bn_clean = t_enc[-1]
                t_bn_norm  = clip_and_normalise(t_bn_clean, caps)
                B, C_t, H_b, W_b = t_bn_norm.shape
                noise      = (torch.randn(B, C_t, H_b, W_b, device=device)
                              * sigma.view(1, C_t, 1, 1))
                t_bn_noisy = denormalise(t_bn_norm + noise, caps)

        # ==============================================================
        # STUDENT PATH
        # ==============================================================
        optimizer.zero_grad()

        s_enc        = unet_encoder(student.backbone, inputs)
        s_bottleneck = s_enc[-1]              # (B, 512, H_b, W_b)
        s_projected  = adapter(s_bottleneck)  # (B, 1024, H_b, W_b)

        f_loss = feature_matching_loss(s_projected, t_bn_noisy)

        s_feats  = unet_decoder(student.backbone, s_enc, s_bottleneck)
        s_logits = student.decode_head.forward(tuple(s_feats))
        t_loss   = task_loss_fn(s_logits, gt, args.ignore_index)

        total = alpha_loss(t_loss, f_loss, args.alpha)

        total.backward()
        optimizer.step()
        scheduler.step()
        iteration += 1

        if iteration % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            with torch.no_grad():
                if 0.0 < args.alpha < 1.0:
                    scale_v  = t_loss.detach() / (f_loss.detach() + 1e-8)
                    f_sc_v   = f_loss.detach() * scale_v
                    contrib  = (args.alpha * f_sc_v) / (total.detach() + 1e-8)
                    c_str    = f"teacher%={contrib.item()*100:.1f}%"
                elif args.alpha == 1.0:
                    c_str = "teacher%=100% (ablation)"
                else:
                    c_str = "teacher%=0% (task-only)"
            print(f"Iter {iteration:5d}/{args.max_iters} | "
                  f"loss={total.item():.4f}  "
                  f"task={t_loss.item():.4f}  "
                  f"feat={f_loss.item():.4f}  "
                  f"{c_str} | lr={lr_now:.5f}")

        # ==============================================================
        # VALIDATION
        # ==============================================================
        if iteration % args.val_interval == 0 or iteration == args.max_iters:
            student.eval()
            adapter.eval()

            all_m: Dict[str, List[float]] = {
                "mDice": [], "vessel_Dice": [], "mIoU": [],
                "pixel_acc": [], "mean_acc": [],
            }

            with torch.no_grad():
                for raw_val in val_loader:
                    dv    = move_to_device(student, raw_val, device, training=False)
                    dv    = pad_to_divisor(dv, args.pad_divisor, args.ignore_index)
                    gt_v  = get_gt(dv["data_samples"], device)
                    enc_v = unet_encoder(student.backbone, dv["inputs"])
                    fts_v = unet_decoder(student.backbone, enc_v, enc_v[-1])
                    log_v = student.decode_head.forward(tuple(fts_v))
                    m     = compute_all_metrics(log_v, gt_v,
                                               ignore_index=args.ignore_index)
                    for k, v in m.items():
                        all_m[k].append(v)

            avg       = {k: float(np.mean(v)) for k, v in all_m.items()}
            avg_mdice = avg["mDice"]

            print(f"\n>>> Val iter {iteration}:")
            print(f"    mDice      = {avg['mDice']*100:.2f}%")
            print(f"    vessel_Dice= {avg['vessel_Dice']*100:.2f}%")
            print(f"    mIoU       = {avg['mIoU']*100:.2f}%")
            print(f"    pixel_acc  = {avg['pixel_acc']*100:.2f}%")
            print(f"    mean_acc   = {avg['mean_acc']*100:.2f}%")

            history.append({
                "iter":        iteration,
                "mDice":       round(avg["mDice"],        6),
                "vessel_Dice": round(avg["vessel_Dice"],  6),
                "mIoU":        round(avg["mIoU"],         6),
                "pixel_acc":   round(avg["pixel_acc"],    6),
                "mean_acc":    round(avg["mean_acc"],     6),
            })

            if avg_mdice > best_mdice:
                best_mdice = avg_mdice
                best_iter  = iteration
                torch.save(student.state_dict(), out_dir / "best_student.pth")
                torch.save(adapter.state_dict(), out_dir / "best_adapter.pth")
                print(f"    ✅ New best mDice={best_mdice*100:.2f}% saved.")

            torch.save(student.state_dict(), out_dir / "latest_student.pth")
            torch.save(adapter.state_dict(), out_dir / "latest_adapter.pth")
            print()
            student.train()
            adapter.train()

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    best_entry = next(
        (h for h in history if h["iter"] == best_iter),
        history[-1] if history else {}
    )

    summary = {
        "version":              "v4_dp_honest_alpha",
        "noise_type":           args.noise_type,
        "epsilon":              args.epsilon,
        "alpha":                args.alpha,
        "teacher_pct":          f"{args.alpha*100:.0f}%",
        "task_pct":             f"{(1-args.alpha)*100:.0f}%",
        "seed":                 args.seed,
        "K":                    args.K,
        "release_model":        "sample-once" if args.precompute_noise
                                else "per-iteration",
        "caps_source":          args.public_caps_csv,
        "importance_source":    args.importance_csv,
        "max_iters":            args.max_iters,
        "best_iter":            best_iter,
        "best_mDice":           round(best_mdice, 6),
        "best_vessel_Dice":     round(best_entry.get("vessel_Dice", 0), 6),
        "best_mIoU":            round(best_entry.get("mIoU",        0), 6),
        "best_pixel_acc":       round(best_entry.get("pixel_acc",   0), 6),
        "best_mean_acc":        round(best_entry.get("mean_acc",    0), 6),
        "student_baseline_mDice": 0.8794,
        "lift_over_baseline":   round(best_mdice - 0.8794, 6),
        "adapter_channels":     f"{student_bn_channels}→{teacher_bn_channels}",
        "history":              history,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("v4 Training complete.")
    print(f"Noise type    : {args.noise_type}")
    print(f"Epsilon       : {args.epsilon}")
    print(f"Alpha         : {args.alpha} ({args.alpha*100:.0f}% teacher / "
          f"{(1-args.alpha)*100:.0f}% task)")
    print(f"Release model : {release_model}")
    print(f"Seed          : {args.seed}")
    print(f"Best mDice    : {best_mdice*100:.2f}%  at iter {best_iter}")
    print(f"Best mIoU     : {best_entry.get('mIoU', 0)*100:.2f}%")
    print(f"Baseline      : 87.94%")
    print(f"Lift          : {(best_mdice - 0.8794)*100:+.2f}%")
    print(f"Saved         : {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()