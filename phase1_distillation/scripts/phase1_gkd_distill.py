#!/usr/bin/env python
"""
GKD Student Distillation — Noisy Teacher Bottleneck on DRIVE.

Implements Gradient-Guided Knowledge Distillation (Lan & Tian, WACV 2024)
adapted for segmentation with privacy noise on the teacher bottleneck.

Pipeline (every training iteration)
-------------------------------------
1. Teacher encoder  → clean bottleneck (1024, 37, 36)  [frozen]
2. Add noise to teacher bottleneck (uniform or channel_WF)
3. Build teacher target map M^T using precomputed gradient weights w_k^T
   (GKD Eq 1, 2, 3 from paper)
4. Student full forward → student bottleneck (512, 37, 36) + student logits
5. Compute student task loss (CE + 3×Dice vs GT)
6. Use torch.autograd.grad to get student bottleneck gradients
   → compute student gradient weights w_k^S on the fly (GKD Eq 1)
   → build student map M^S (GKD Eq 2, 3)
7. GKD loss = (1/HW) × Σ|M^T - M^S|  (L1, GKD Eq 4)
8. Total loss = task_loss + lambda_gkd × GKD_loss
9. Backward + update student only

Two experiments (run separately via --noise-type):
  --noise-type uniform     → same sigma for all 1024 teacher channels
  --noise-type channel_WF  → sigma_c ∝ sqrt(Δc) / s_c^{1/4}
                             less noise on important channels

Run one job per epsilon per noise type:
  python phase1_gkd_distill.py --noise-type uniform    --epsilon 2
  python phase1_gkd_distill.py --noise-type channel_WF --epsilon 2
  ... etc for epsilon in {2, 4, 8, 16}
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
# Privacy accounting  (same as all previous scripts)
# ---------------------------------------------------------------------------

def eps_to_rho(eps: float, delta: float = 1e-5) -> float:
    """Bun-Steinke: (eps, delta)-DP → rho-zCDP."""
    log_inv_d = math.log(1.0 / delta)
    b = 2.0 * math.sqrt(log_inv_d)
    discriminant = b * b + 4.0 * eps
    sqrt_rho = (-b + math.sqrt(discriminant)) / 2.0
    return sqrt_rho * sqrt_rho


# ---------------------------------------------------------------------------
# Noise allocation  (same as all previous scripts)
# ---------------------------------------------------------------------------

def waterfilling_sigma(
    deltas: torch.Tensor,
    importances: torch.Tensor,
    rdp_budget: float,
    eps_s: float = 1e-12,
) -> torch.Tensor:
    """
    Channel-WF: sigma_c = kappa * sqrt(delta_c) / s_c^{1/4}.
    Important channels (high s_c) get less noise.
    """
    s = importances.clamp(min=eps_s)
    kappa = ((deltas * s.sqrt()).sum() / rdp_budget).sqrt()
    return kappa * deltas.sqrt() / s.pow(0.25)


def uniform_sigma(deltas: torch.Tensor, rdp_budget: float) -> torch.Tensor:
    """Uniform: same sigma for all channels."""
    sigma = (deltas.pow(2).sum() / rdp_budget).sqrt()
    return sigma.expand(deltas.shape[0]).clone()


# ---------------------------------------------------------------------------
# MMSeg model building
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
    """Build one mmseg model, optionally freeze all parameters."""
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


def build_loader(cfg: Config, split: str, num_workers: int) -> DataLoader:
    key = "train_dataloader" if split == "train" else "val_dataloader"
    dataset = DATASETS.build(cfg[key]["dataset"])
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


def move_to_device(model, data: Dict, device: str) -> Dict:
    data = model.data_preprocessor(data, training=True)
    data["inputs"] = data["inputs"].to(device)
    return data


def move_to_device_val(model, data: Dict, device: str) -> Dict:
    data = model.data_preprocessor(data, training=False)
    data["inputs"] = data["inputs"].to(device)
    return data


def pad_to_divisor(
    data: Dict,
    divisor: int = 16,
    ignore_index: int = 255,
) -> Dict:
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
            sample.gt_sem_seg.data,
            (0, pad_w, 0, pad_h),
            value=ignore_index,
        )
        sample.gt_sem_seg = PixelData(data=padded_gt)
        sample.set_metainfo({
            "pad_shape": (new_h, new_w),
            "img_shape": (new_h, new_w),
        })
    return data


def get_gt(data_samples: Sequence, device: str) -> torch.Tensor:
    return torch.stack([
        s.gt_sem_seg.data.squeeze(0).long().to(device)
        for s in data_samples
    ])


# ---------------------------------------------------------------------------
# U-Net manual forward (same as all previous scripts)
# ---------------------------------------------------------------------------

def unet_encoder(backbone, x: torch.Tensor) -> List[torch.Tensor]:
    enc_outs = []
    for enc in backbone.encoder:
        x = enc(x)
        enc_outs.append(x)
    return enc_outs


def unet_decoder(
    backbone,
    enc_outs: Sequence[torch.Tensor],
    bottleneck: torch.Tensor,
) -> List[torch.Tensor]:
    x = bottleneck
    dec_outs = [x]
    for i in reversed(range(len(backbone.decoder))):
        x = backbone.decoder[i](enc_outs[i], x)
        dec_outs.append(x)
    return dec_outs


# ---------------------------------------------------------------------------
# Normalisation helpers (same as Phase 1 v2)
# ---------------------------------------------------------------------------

def clip_and_normalise(
    bottleneck: torch.Tensor,
    caps: torch.Tensor,
) -> torch.Tensor:
    """
    Per-channel L2 clipping then normalisation.
    After this call, every channel has L2 norm in [0, 1].
    """
    B, C, H, W = bottleneck.shape
    norms = bottleneck.flatten(2).norm(dim=2)          # (1, C)
    scale = (caps.view(1, C) / norms.clamp(min=1e-12)).clamp(max=1.0)
    clipped = bottleneck * scale.view(B, C, 1, 1)
    normalised = clipped / caps.view(1, C, 1, 1).clamp(min=1e-12)
    return normalised


def denormalise(normalised: torch.Tensor, caps: torch.Tensor) -> torch.Tensor:
    _, C, H, W = normalised.shape
    return normalised * caps.view(1, C, 1, 1)


# ---------------------------------------------------------------------------
# GKD map computation  (GKD paper Eq 1, 2, 3)
# ---------------------------------------------------------------------------

def build_gkd_map(
    bottleneck: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Build the gradient-weighted attention map M from a bottleneck tensor.

    Implements GKD Eq 1, 2, 3 from Lan & Tian (WACV 2024):
      Eq 2: weighted_k = w_k * A_k
      Eq 3: M = Norm(|sum_k weighted_k|)

    Args:
        bottleneck: (1, C, H, W) — feature map (teacher noisy OR student)
        weights:    (C,)          — per-channel importance weights w_k
                                   For teacher: precomputed gradient_abs_mean
                                   For student: computed on the fly each iter

    Returns:
        M: (1, 1, H, W) — normalised attention map, values in [0, 1]
    """
    C = bottleneck.shape[1]
    # Eq 2: multiply each channel by its weight
    weighted = bottleneck * weights.view(1, C, 1, 1)   # (1, C, H, W)
    # Eq 3: sum over channels, take absolute value
    summed = weighted.sum(dim=1, keepdim=True).abs()   # (1, 1, H, W)
    # Min-max normalisation (Norm in Eq 3)
    b_min = summed.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1)
    b_max = summed.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
    M = (summed - b_min) / (b_max - b_min + eps)
    return M


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(
    logits: torch.Tensor,
    gt: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Soft Dice loss for binary segmentation (vessel class = 1)."""
    valid = (gt != ignore_index)
    prob = torch.softmax(logits, dim=1)[:, 1]     # vessel probability
    gt_f = (gt == 1).float()
    prob = prob * valid.float()
    gt_f = gt_f * valid.float()
    inter = (prob * gt_f).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + gt_f.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def ce_loss(
    logits: torch.Tensor,
    gt: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Standard cross-entropy segmentation loss."""
    # Resize logits to match gt if needed
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear",
                               align_corners=False)
    return F.cross_entropy(logits, gt, ignore_index=ignore_index)


def task_loss(
    logits: torch.Tensor,
    gt: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """CE + 3×Dice — matches teacher training config."""
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear",
                               align_corners=False)
    return ce_loss(logits, gt, ignore_index) + 3.0 * dice_loss(logits, gt, ignore_index)


def gkd_loss(M_T: torch.Tensor, M_S: torch.Tensor) -> torch.Tensor:
    """
    GKD distillation loss (GKD paper Eq 4).
    L1 loss between teacher and student attention maps, averaged over HW.
    """
    return F.l1_loss(M_S, M_T.detach(), reduction="mean")


# ---------------------------------------------------------------------------
# Dice metric for validation
# ---------------------------------------------------------------------------

def dice_metric(
    logits: torch.Tensor,
    gt: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-7,
) -> float:
    """Hard Dice on vessel class (class=1) for validation reporting."""
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear",
                               align_corners=False)
    pred = logits.argmax(dim=1)
    valid = gt != ignore_index
    pred_v = (pred == 1) & valid
    gt_v = (gt == 1) & valid
    dims = tuple(range(1, gt.ndim))
    inter = (pred_v & gt_v).sum(dim=dims).float()
    denom = pred_v.sum(dim=dims).float() + gt_v.sum(dim=dims).float()
    return float(((2.0 * inter + eps) / (denom + eps)).mean().item())


def mdice_metric(
    logits: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = 255,
    eps: float = 1e-7,
) -> float:
    """
    Mean Dice across all classes (matches mmengine IoUMetric mDice).
    This is what your baseline results report.
    """
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear",
                               align_corners=False)
    pred = logits.argmax(dim=1)
    valid = gt != ignore_index
    dice_per_class = []
    for cls in range(num_classes):
        pred_c = (pred == cls) & valid
        gt_c = (gt == cls) & valid
        dims = tuple(range(1, gt.ndim))
        inter = (pred_c & gt_c).sum(dim=dims).float()
        denom = pred_c.sum(dim=dims).float() + gt_c.sum(dim=dims).float()
        dice_c = ((2.0 * inter + eps) / (denom + eps)).mean().item()
        dice_per_class.append(dice_c)
    return float(np.mean(dice_per_class))


# ---------------------------------------------------------------------------
# Importance CSV loader
# ---------------------------------------------------------------------------

def load_importances(csv_path: str) -> torch.Tensor:
    """Load teacher gradient_abs_mean from Phase 1 Step B CSV."""
    vals = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["gradient_abs_mean"]))
    return torch.tensor(vals, dtype=torch.float32).clamp(min=1e-12)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GKD distillation with noisy teacher bottleneck on DRIVE."
    )
    p.add_argument("--teacher-config",     required=True)
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--student-config",     required=True)
    p.add_argument("--student-checkpoint", required=True,
                   help="Student baseline checkpoint to initialise from.")
    p.add_argument("--data-root",          required=True)
    p.add_argument("--importance-csv",     required=True,
                   help="channel_importance_scores_bottleneck_gradient.csv")
    p.add_argument("--out-dir",            required=True)
    p.add_argument("--noise-type",         required=True,
                   choices=["uniform", "channel_WF"])
    p.add_argument("--epsilon",            type=float, required=True,
                   help="Privacy budget epsilon (e.g. 2, 4, 8, 16)")
    p.add_argument("--K",                  type=int, default=1,
                   help="Number of teachers for sensitivity (K=1 = single teacher)")
    p.add_argument("--delta-dp",           type=float, default=1e-5)
    p.add_argument("--lambda-gkd",         type=float, default=0.4,
                   help="Weight of GKD loss in total loss.")
    p.add_argument("--max-iters",          type=int, default=40000)
    p.add_argument("--val-interval",       type=int, default=4000)
    p.add_argument("--device",             default="cuda:0")
    p.add_argument("--num-workers",        type=int, default=4)
    p.add_argument("--pad-divisor",        type=int, default=16)
    p.add_argument("--ignore-index",       type=int, default=255)
    p.add_argument("--cap-quantile",       type=float, default=0.9)
    p.add_argument("--seed",               type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    # ------------------------------------------------------------------
    # Build teacher (frozen) and student (trainable)
    # ------------------------------------------------------------------
    t_cfg = Config.fromfile(args.teacher_config)
    s_cfg = Config.fromfile(args.student_config)
    set_data_root(t_cfg, args.data_root)
    set_data_root(s_cfg, args.data_root)

    print("Building teacher (frozen)...")
    teacher = build_model(t_cfg, args.teacher_checkpoint, device, frozen=True)

    print("Building student (from baseline checkpoint)...")
    student = build_model(s_cfg, args.student_checkpoint, device, frozen=False)

    # ------------------------------------------------------------------
    # Load precomputed teacher channel importance  (w_k^T)
    # ------------------------------------------------------------------
    teacher_importance = load_importances(args.importance_csv).to(device)
    C_T = teacher_importance.shape[0]
    print(f"Teacher importance scores loaded: C={C_T} channels")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    train_loader = build_loader(s_cfg, "train", args.num_workers)
    val_loader   = build_loader(s_cfg, "val",   args.num_workers)

    # ------------------------------------------------------------------
    # PASS 0: estimate per-channel caps from teacher on training data
    # (needed for noise normalisation)
    # ------------------------------------------------------------------
    print("\nPass 0: estimating teacher bottleneck caps from training data...")
    all_norms: List[torch.Tensor] = []
    n_cap = 0
    teacher.eval()
    with torch.no_grad():
        for raw_data in train_loader:
            data = move_to_device(teacher, raw_data, device)
            data = pad_to_divisor(data, args.pad_divisor, args.ignore_index)
            enc_outs = unet_encoder(teacher.backbone, data["inputs"])
            bottleneck = enc_outs[-1]                  # (B, C, H, W)
            # Per-sample, per-channel L2 norm
            for b in range(bottleneck.shape[0]):
                norms = bottleneck[b].flatten(1).norm(dim=1)   # (C,)
                all_norms.append(norms.cpu())
                n_cap += 1
            if n_cap >= 100:   # enough samples for stable cap estimate
                break

    norms_tensor = torch.stack(all_norms, dim=0)       # (N, C)
    caps = torch.quantile(norms_tensor, args.cap_quantile, dim=0).to(device)
    # SENSITIVITY FIX (same bug as v2): noise is added in the normalised
    # space where clip_and_normalise() forces every channel L2 norm <= 1,
    # so the replace-one per-channel L2 sensitivity is the data-independent
    # constant 2/K, NOT 2*caps/K. The old 2*caps/K double-counted `caps`
    # and inflated sigma by ~caps, drowning the teacher signal at every
    # epsilon. denormalise() (*caps after noise) is pure post-processing.
    deltas = torch.full_like(caps, 2.0 / args.K)        # sensitivity (fixed)

    print(f"Cap stats: min={caps.min():.3f} max={caps.max():.3f} "
          f"mean={caps.mean():.3f}  (clip/normalise only, NOT sensitivity)")
    print(f"K={args.K}, delta_c = 2/K (normalised-space sensitivity) "
          f"= {2.0 / args.K:.4f}  (data-independent)")

    # ------------------------------------------------------------------
    # Pre-compute noise sigma for the chosen epsilon
    # ------------------------------------------------------------------
    rho = eps_to_rho(args.epsilon, args.delta_dp)
    if args.noise_type == "channel_WF":
        sigma = waterfilling_sigma(deltas, teacher_importance, rho)
    else:
        sigma = uniform_sigma(deltas, rho)

    print(f"\nNoise type: {args.noise_type}")
    print(f"Epsilon={args.epsilon}  rho={rho:.5f}")
    print(f"Sigma: min={sigma.min():.4f} max={sigma.max():.4f} "
          f"mean={sigma.mean():.4f}")

    if args.noise_type == "channel_WF":
        top20 = torch.argsort(teacher_importance, descending=True)[:20]
        bot20 = torch.argsort(teacher_importance, descending=False)[:20]
        print(f"  WF sigma top-20 important channels: "
              f"{sigma[top20].mean():.4f}  (should be SMALLER)")
        print(f"  WF sigma bot-20 unimportant channels: "
              f"{sigma[bot20].mean():.4f}")

    # ------------------------------------------------------------------
    # Optimiser and LR scheduler  (matches student training config)
    # ------------------------------------------------------------------
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # Polynomial LR decay: lr * (1 - iter/max_iters)^0.9
    def poly_lr(iteration: int) -> float:
        return (1.0 - iteration / args.max_iters) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Training: noise={args.noise_type}  eps={args.epsilon}  "
          f"lambda_gkd={args.lambda_gkd}")
    print(f"Max iters: {args.max_iters}  Val every: {args.val_interval}")
    print("=" * 70)

    best_mdice = 0.0
    best_iter  = 0
    history    = []

    # Infinite training iterator
    train_iter = iter(train_loader)
    iteration  = 0

    student.train()
    teacher.eval()

    while iteration < args.max_iters:
        # --- fetch next training batch ---
        try:
            raw_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            raw_data = next(train_iter)

        # --- prepare inputs ---
        data = move_to_device(student, raw_data, device)
        inputs = data["inputs"]
        gt = get_gt(data["data_samples"], device)

        # ==============================================================
        # TEACHER PATH  (no gradients for teacher parameters)
        # ==============================================================
        with torch.no_grad():
            t_enc = unet_encoder(teacher.backbone, inputs)
            t_bottleneck_clean = t_enc[-1]              # (B, 1024, H_b, W_b)

            # Normalise → add noise → denormalise
            t_bn_norm = clip_and_normalise(t_bottleneck_clean, caps)
            B, C_t, H_b, W_b = t_bn_norm.shape
            noise = (
                torch.randn(B, C_t, H_b, W_b, device=device)
                * sigma.view(1, C_t, 1, 1)
            )
            t_bn_noisy_norm = t_bn_norm + noise
            t_bn_noisy = denormalise(t_bn_noisy_norm, caps)

        # Build teacher map M^T  (GKD Eq 2, 3)
        # Use precomputed teacher importance weights w_k^T
        # (one weight per teacher channel, averaged over the batch)
        M_T = build_gkd_map(
            t_bn_noisy.detach().mean(dim=0, keepdim=True),   # avg over batch
            teacher_importance,
        )   # (1, 1, H_b, W_b)

        # ==============================================================
        # STUDENT PATH
        # ==============================================================
        optimizer.zero_grad()

        # Student encoder → bottleneck
        s_enc = unet_encoder(student.backbone, inputs)
        s_bottleneck = s_enc[-1]                       # (B, 512, H_b, W_b)
        s_bottleneck.retain_grad()

        # Student decoder → logits
        s_feats  = unet_decoder(student.backbone, s_enc, s_bottleneck)
        s_logits = student.decode_head.forward(tuple(s_feats))

        # --- Task loss (CE + 3×Dice, student vs GT) ---
        t_loss = task_loss(s_logits, gt, args.ignore_index)

        # --- Student gradient weights w_k^S  (GKD Eq 1, on the fly) ---
        # Compute dL_task/dA_student_bottleneck using autograd.grad
        # create_graph=False: we do not need higher-order gradients
        # retain_graph=True:  we will call .backward() on total_loss later
        s_grads = torch.autograd.grad(
            outputs=t_loss,
            inputs=s_bottleneck,
            create_graph=False,
            retain_graph=True,
        )[0]   # (B, 512, H_b, W_b)

        # w_k^S = mean(|dL/dA_k^S|) over (batch, H, W)  — GKD Eq 1
        s_weights = s_grads.abs().mean(dim=(0, 2, 3)).detach()   # (512,)
        s_weights = s_weights.clamp(min=1e-12)

        # Build student map M^S  (GKD Eq 2, 3)
        M_S = build_gkd_map(
            s_bottleneck.mean(dim=0, keepdim=True),    # avg over batch
            s_weights,
        )   # (1, 1, H_b, W_b)

        # --- GKD loss  (GKD Eq 4, L1) ---
        g_loss = gkd_loss(M_T, M_S)

        # --- Total loss ---
        total = t_loss + args.lambda_gkd * g_loss

        # --- Backward + update student ---
        total.backward()
        optimizer.step()
        scheduler.step()

        iteration += 1

        # --- Logging ---
        if iteration % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Iter {iteration:5d}/{args.max_iters} | "
                  f"loss={total.item():.4f} "
                  f"task={t_loss.item():.4f} "
                  f"gkd={g_loss.item():.4f} | "
                  f"lr={lr_now:.5f}")

        # ==============================================================
        # VALIDATION
        # ==============================================================
        if iteration % args.val_interval == 0 or iteration == args.max_iters:
            student.eval()
            val_mdice_list  = []
            val_vdice_list  = []

            with torch.no_grad():
                for raw_val in val_loader:
                    data_v = move_to_device_val(student, raw_val, device)
                    data_v = pad_to_divisor(data_v, args.pad_divisor,
                                            args.ignore_index)
                    gt_v   = get_gt(data_v["data_samples"], device)
                    inputs_v = data_v["inputs"]

                    s_enc_v  = unet_encoder(student.backbone, inputs_v)
                    s_feats_v = unet_decoder(student.backbone, s_enc_v,
                                             s_enc_v[-1])
                    logits_v  = student.decode_head.forward(tuple(s_feats_v))

                    val_mdice_list.append(
                        mdice_metric(logits_v, gt_v,
                                     ignore_index=args.ignore_index))
                    val_vdice_list.append(
                        dice_metric(logits_v, gt_v,
                                    ignore_index=args.ignore_index))

            avg_mdice = float(np.mean(val_mdice_list))
            avg_vdice = float(np.mean(val_vdice_list))

            print(f"\n>>> Val iter {iteration}: "
                  f"mDice={avg_mdice*100:.2f}%  "
                  f"vessel_Dice={avg_vdice*100:.2f}%")

            history.append({
                "iter":        iteration,
                "mDice":       round(avg_mdice, 6),
                "vessel_Dice": round(avg_vdice, 6),
            })

            # Save best checkpoint
            if avg_mdice > best_mdice:
                best_mdice = avg_mdice
                best_iter  = iteration
                ckpt_path  = out_dir / "best_student.pth"
                torch.save(student.state_dict(), ckpt_path)
                print(f"    ✅ New best mDice={best_mdice*100:.2f}% "
                      f"saved to {ckpt_path}")

            # Save latest checkpoint
            torch.save(student.state_dict(), out_dir / "latest_student.pth")

            print()
            student.train()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    summary = {
        "noise_type":       args.noise_type,
        "epsilon":          args.epsilon,
        "K":                args.K,
        "lambda_gkd":       args.lambda_gkd,
        "max_iters":        args.max_iters,
        "best_mDice":       round(best_mdice, 6),
        "best_iter":        best_iter,
        "student_baseline_mDice": 0.8791,   # your measured baseline
        "history":          history,
        "teacher_config":   args.teacher_config,
        "student_config":   args.student_config,
        "importance_csv":   args.importance_csv,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Training complete.")
    print(f"Noise type:      {args.noise_type}")
    print(f"Epsilon:         {args.epsilon}")
    print(f"Best mDice:      {best_mdice*100:.2f}%  at iter {best_iter}")
    print(f"Student baseline:{0.8791*100:.2f}%")
    print(f"Lift over base:  {(best_mdice - 0.8791)*100:+.2f}%")
    print(f"Saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()