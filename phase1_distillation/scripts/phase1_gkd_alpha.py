#!/usr/bin/env python
"""
GKD Student Distillation v3 — Normalised Alpha Loss + Extended Metrics on DRIVE.

What changed from v2
----------------------
v2 used:
    total = task_loss + lambda_feat * feature_loss

Problem: lambda_feat does not give a true percentage split because
task_loss and feature_loss are on different scales. lambda=0.4 does
NOT mean 40% from teacher.

v3 uses a normalised alpha approach:
    scale         = task_loss.detach() / (feature_loss.detach() + 1e-8)
    f_loss_scaled = feature_loss * scale        ← brings f_loss to same scale as t_loss
    total         = (1 - alpha) * task_loss + alpha * f_loss_scaled

Now alpha is a TRUE percentage:
    alpha = 0.5  → exactly 50% from noisy teacher, 50% from ground truth
    alpha = 0.6  → exactly 60% from noisy teacher, 40% from ground truth
    alpha = 0.7  → exactly 70% from noisy teacher, 30% from ground truth
    alpha = 0.8  → exactly 80% from noisy teacher, 20% from ground truth
    alpha = 0.9  → exactly 90% from noisy teacher, 10% from ground truth

Also added extended metrics at validation:
    - mDice       (mean Dice across both classes)
    - vessel_Dice (Dice on vessel class only)
    - mIoU        (mean Intersection over Union across both classes)
    - pixel_acc   (overall pixel accuracy)
    - mean_acc    (per-class accuracy averaged)

Pipeline (every training iteration)
--------------------------------------
1. Teacher encoder  → clean bottleneck (B, 1024, 37, 36)  [frozen]
2. Add noise to teacher bottleneck (uniform or channel_WF)
3. Student encoder  → student bottleneck (B, 512, 37, 36)
4. Adapter (1x1 conv, trainable) → projected student (B, 1024, 37, 36)
5. feature_loss = MSE(projected_student, noisy_teacher_bn)
6. Student decoder → student logits
7. task_loss = CE + 3*Dice (student logits vs GT)
8. scale = task_loss.detach() / (feature_loss.detach() + 1e-8)
9. total = (1-alpha)*task_loss + alpha*(feature_loss*scale)
10. Backward + update student + adapter (teacher frozen)

Run one job per noise type per epsilon per alpha per seed:
  python phase1_gkd_alpha.py --noise-type uniform    --epsilon 2 --alpha 0.5 --seed 0
  python phase1_gkd_alpha.py --noise-type channel_WF --epsilon 2 --alpha 0.7 --seed 0
  ... etc

Total runs: 2 noise types x 4 epsilons x 5 alphas x 3 seeds = 120 jobs
"""

import argparse
import csv
import json
import math
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
# Privacy accounting
# ---------------------------------------------------------------------------

def eps_to_rho(eps: float, delta: float = 1e-5) -> float:
    """Bun-Steinke: (eps, delta)-DP → rho-zCDP."""
    log_inv_d = math.log(1.0 / delta)
    b = 2.0 * math.sqrt(log_inv_d)
    discriminant = b * b + 4.0 * eps
    sqrt_rho = (-b + math.sqrt(discriminant)) / 2.0
    return sqrt_rho * sqrt_rho


# ---------------------------------------------------------------------------
# Noise allocation
# ---------------------------------------------------------------------------

def waterfilling_sigma(
    deltas: torch.Tensor,
    importances: torch.Tensor,
    rdp_budget: float,
    eps_s: float = 1e-12,
) -> torch.Tensor:
    """Channel-WF: sigma_c = kappa * sqrt(delta_c) / s_c^{1/4}."""
    s = importances.clamp(min=eps_s)
    kappa = ((deltas * s.sqrt()).sum() / rdp_budget).sqrt()
    return kappa * deltas.sqrt() / s.pow(0.25)


def uniform_sigma(deltas: torch.Tensor, rdp_budget: float) -> torch.Tensor:
    """Uniform: same sigma for all channels."""
    sigma = (deltas.pow(2).sum() / rdp_budget).sqrt()
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


def move_to_device(model, data: Dict, device: str, training: bool) -> Dict:
    data = model.data_preprocessor(data, training=training)
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
# U-Net manual forward
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
# Normalisation
# ---------------------------------------------------------------------------

def clip_and_normalise(
    bottleneck: torch.Tensor,
    caps: torch.Tensor,
) -> torch.Tensor:
    """Per-channel L2 clipping then normalisation. After this every channel norm <= 1."""
    B, C, H, W = bottleneck.shape
    norms = bottleneck.flatten(2).norm(dim=2)
    scale = (caps.view(1, C) / norms.clamp(min=1e-12)).clamp(max=1.0)
    clipped = bottleneck * scale.view(B, C, 1, 1)
    normalised = clipped / caps.view(1, C, 1, 1).clamp(min=1e-12)
    return normalised


def denormalise(normalised: torch.Tensor, caps: torch.Tensor) -> torch.Tensor:
    _, C, H, W = normalised.shape
    return normalised * caps.view(1, C, 1, 1)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BottleneckAdapter(nn.Module):
    """1x1 conv: student_channels → teacher_channels."""
    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            student_channels,
            teacher_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss_fn(
    logits: torch.Tensor,
    gt: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-7,
) -> torch.Tensor:
    valid = (gt != ignore_index)
    prob  = torch.softmax(logits, dim=1)[:, 1]
    gt_f  = (gt == 1).float()
    prob  = prob * valid.float()
    gt_f  = gt_f * valid.float()
    inter = (prob * gt_f).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + gt_f.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def task_loss_fn(
    logits: torch.Tensor,
    gt: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """CE + 3xDice — matches teacher training config."""
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:],
                               mode="bilinear", align_corners=False)
    ce = F.cross_entropy(logits, gt, ignore_index=ignore_index)
    di = dice_loss_fn(logits, gt, ignore_index)
    return ce + 3.0 * di


def feature_matching_loss(
    projected_student: torch.Tensor,
    noisy_teacher_bn: torch.Tensor,
) -> torch.Tensor:
    """MSE between projected student and noisy teacher bottleneck."""
    return F.mse_loss(
        projected_student,
        noisy_teacher_bn.detach(),
        reduction="mean",
    )


def alpha_loss(
    t_loss: torch.Tensor,
    f_loss: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Normalised alpha loss — TRUE percentage split between task and teacher.

    Steps:
      1. Compute scale = t_loss / f_loss  (brings f_loss to same magnitude as t_loss)
      2. f_loss_scaled = f_loss * scale
      3. total = (1-alpha)*t_loss + alpha*f_loss_scaled

    Now alpha=0.7 means exactly 70% of the total loss comes from the
    noisy teacher signal and 30% from the ground truth task loss,
    regardless of the raw magnitudes of t_loss and f_loss.
    """
    scale = t_loss.detach() / (f_loss.detach() + 1e-8)
    f_loss_scaled = f_loss * scale
    return (1.0 - alpha) * t_loss + alpha * f_loss_scaled


# ---------------------------------------------------------------------------
# Extended metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    logits: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = 255,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics in one pass.

    Returns dict with:
        mDice      — mean Dice across all classes
        vessel_Dice — Dice on vessel class (class=1) only
        mIoU       — mean Intersection over Union across all classes
        pixel_acc  — overall pixel accuracy (correct pixels / total valid pixels)
        mean_acc   — per-class recall averaged across classes
    """
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:],
                               mode="bilinear", align_corners=False)

    pred  = logits.argmax(dim=1)   # (B, H, W)
    valid = gt != ignore_index     # (B, H, W) boolean mask

    dices      = []
    ious       = []
    class_accs = []

    for cls in range(num_classes):
        pred_c = (pred == cls) & valid   # predicted as cls and not ignored
        gt_c   = (gt   == cls) & valid   # ground truth is cls and not ignored

        dims  = tuple(range(1, gt.ndim))
        inter = (pred_c & gt_c).sum(dim=dims).float()   # true positives
        union = (pred_c | gt_c).sum(dim=dims).float()   # union

        # Dice
        denom_dice = pred_c.sum(dim=dims).float() + gt_c.sum(dim=dims).float()
        dice_c = ((2.0 * inter + eps) / (denom_dice + eps)).mean().item()
        dices.append(dice_c)

        # IoU
        iou_c = ((inter + eps) / (union + eps)).mean().item()
        ious.append(iou_c)

        # Per-class accuracy (recall): correctly predicted cls / all gt cls pixels
        gt_count = gt_c.sum(dim=dims).float()
        acc_c = ((inter + eps) / (gt_count + eps)).mean().item()
        class_accs.append(acc_c)

    # Pixel accuracy: total correct / total valid
    correct = ((pred == gt) & valid).sum().float()
    total   = valid.sum().float()
    pixel_acc = (correct / (total + eps)).item()

    return {
        "mDice":       float(np.mean(dices)),
        "vessel_Dice": dices[1],          # class 1 = vessel
        "mIoU":        float(np.mean(ious)),
        "pixel_acc":   pixel_acc,
        "mean_acc":    float(np.mean(class_accs)),
    }


# ---------------------------------------------------------------------------
# Importance CSV loader
# ---------------------------------------------------------------------------

def load_importances(csv_path: str) -> torch.Tensor:
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
        description=(
            "GKD v3: normalised alpha loss with extended metrics on DRIVE. "
            "alpha controls the TRUE percentage of learning from noisy teacher."
        )
    )
    p.add_argument("--teacher-config",     required=True)
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--student-config",     required=True)
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--data-root",          required=True)
    p.add_argument("--importance-csv",     required=True)
    p.add_argument("--out-dir",            required=True)
    p.add_argument("--noise-type",         required=True,
                   choices=["uniform", "channel_WF"])
    p.add_argument("--epsilon",            type=float, required=True)
    p.add_argument("--alpha",              type=float, required=True,
                   help=(
                       "Teacher learning percentage (0.0 to 1.0). "
                       "alpha=0.7 means 70%% from noisy teacher, 30%% from GT. "
                       "Use values: 0.5, 0.6, 0.7, 0.8, 0.9"
                   ))
    p.add_argument("--K",                  type=int,   default=1)
    p.add_argument("--delta-dp",           type=float, default=1e-5)
    p.add_argument("--max-iters",          type=int,   default=40000)
    p.add_argument("--val-interval",       type=int,   default=4000)
    p.add_argument("--device",             default="cuda:0")
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--pad-divisor",        type=int,   default=16)
    p.add_argument("--ignore-index",       type=int,   default=255)
    p.add_argument("--cap-quantile",       type=float, default=0.9)
    p.add_argument("--seed",               type=int,   default=0,
                   help="Random seed. Run with 0, 1, 2 for statistical averaging.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate alpha
    if not (0.0 < args.alpha < 1.0):
        raise ValueError(f"alpha must be between 0 and 1 exclusive, got {args.alpha}")

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
    # Build adapter (student channels → teacher channels)
    # ------------------------------------------------------------------
    student_bn_channels = s_cfg.model.backbone.base_channels * (2 ** 4)  # 512
    teacher_bn_channels = t_cfg.model.backbone.base_channels * (2 ** 4)  # 1024

    adapter = BottleneckAdapter(student_bn_channels, teacher_bn_channels).to(device)
    print(f"Adapter: {student_bn_channels} → {teacher_bn_channels} channels "
          f"(1x1 conv, {sum(p.numel() for p in adapter.parameters())} params)")

    # ------------------------------------------------------------------
    # Load teacher channel importance scores
    # ------------------------------------------------------------------
    teacher_importance = load_importances(args.importance_csv).to(device)
    C_T = teacher_importance.shape[0]
    print(f"Teacher importance scores: C={C_T} channels")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    train_loader = build_loader(s_cfg, "train", args.num_workers)
    val_loader   = build_loader(s_cfg, "val",   args.num_workers)

    # ------------------------------------------------------------------
    # Pass 0: estimate per-channel caps from training data
    # ------------------------------------------------------------------
    print("\nPass 0: estimating teacher bottleneck caps...")
    all_norms: List[torch.Tensor] = []
    n_cap = 0
    teacher.eval()
    with torch.no_grad():
        for raw_data in train_loader:
            data = move_to_device(teacher, raw_data, device, training=True)
            data = pad_to_divisor(data, args.pad_divisor, args.ignore_index)
            enc_outs   = unet_encoder(teacher.backbone, data["inputs"])
            bottleneck = enc_outs[-1]
            for b in range(bottleneck.shape[0]):
                norms = bottleneck[b].flatten(1).norm(dim=1)
                all_norms.append(norms.cpu())
                n_cap += 1
            if n_cap >= 100:
                break

    norms_tensor = torch.stack(all_norms, dim=0)
    caps   = torch.quantile(norms_tensor, args.cap_quantile, dim=0).to(device)

    # Sensitivity fix: normalised space sensitivity is constant 2/K
    deltas = torch.full_like(caps, 2.0 / args.K)

    print(f"Cap stats: min={caps.min():.3f}  max={caps.max():.3f}  mean={caps.mean():.3f}")
    print(f"K={args.K}  delta_c = 2/K = {2.0 / args.K:.4f}  (constant, normalised space)")

    # ------------------------------------------------------------------
    # Compute noise sigma
    # ------------------------------------------------------------------
    rho = eps_to_rho(args.epsilon, args.delta_dp)
    if args.noise_type == "channel_WF":
        sigma = waterfilling_sigma(deltas, teacher_importance, rho)
    else:
        sigma = uniform_sigma(deltas, rho)

    print(f"\nNoise type : {args.noise_type}")
    print(f"Epsilon    : {args.epsilon}  rho={rho:.5f}")
    print(f"Sigma      : min={sigma.min():.4f}  max={sigma.max():.4f}  mean={sigma.mean():.4f}")

    if args.noise_type == "channel_WF":
        top20 = torch.argsort(teacher_importance, descending=True)[:20]
        bot20 = torch.argsort(teacher_importance, descending=False)[:20]
        print(f"  WF top-20 important sigma  : {sigma[top20].mean():.4f}  ← SMALLER")
        print(f"  WF bot-20 unimportant sigma: {sigma[bot20].mean():.4f}  ← LARGER")
        if sigma[top20].mean() < sigma[bot20].mean():
            print("  ✅ Water-filling correct")
        else:
            print("  ⚠️  Water-filling inverted — check importance scores")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    all_params = list(student.parameters()) + list(adapter.parameters())
    optimizer  = torch.optim.SGD(
        all_params, lr=0.01, momentum=0.9, weight_decay=0.0005,
    )

    def poly_lr(iteration: int) -> float:
        return (1.0 - iteration / args.max_iters) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"v3: normalised alpha loss | "
          f"noise={args.noise_type} | eps={args.epsilon} | seed={args.seed}")
    print(f"alpha={args.alpha} → {args.alpha*100:.0f}% from noisy teacher, "
          f"{(1-args.alpha)*100:.0f}% from GT")
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
        inputs = data["inputs"]
        gt     = get_gt(data["data_samples"], device)

        # ==============================================================
        # TEACHER PATH — frozen, no grad
        # ==============================================================
        with torch.no_grad():
            t_enc              = unet_encoder(teacher.backbone, inputs)
            t_bottleneck_clean = t_enc[-1]

            # Clip + normalise → add noise → denormalise
            t_bn_norm = clip_and_normalise(t_bottleneck_clean, caps)
            B, C_t, H_b, W_b = t_bn_norm.shape
            noise = (
                torch.randn(B, C_t, H_b, W_b, device=device)
                * sigma.view(1, C_t, 1, 1)
            )
            t_bn_noisy = denormalise(t_bn_norm + noise, caps)

        # ==============================================================
        # STUDENT PATH
        # ==============================================================
        optimizer.zero_grad()

        s_enc        = unet_encoder(student.backbone, inputs)
        s_bottleneck = s_enc[-1]                         # (B, 512, H_b, W_b)
        s_projected  = adapter(s_bottleneck)             # (B, 1024, H_b, W_b)

        # Feature loss — student learns from noisy teacher
        f_loss = feature_matching_loss(s_projected, t_bn_noisy)

        # Task loss — student learns from ground truth
        s_feats  = unet_decoder(student.backbone, s_enc, s_bottleneck)
        s_logits = student.decode_head.forward(tuple(s_feats))
        t_loss   = task_loss_fn(s_logits, gt, args.ignore_index)

        # Normalised alpha loss — TRUE percentage split
        # scale brings f_loss to the same magnitude as t_loss
        # so alpha genuinely controls the percentage contribution
        total = alpha_loss(t_loss, f_loss, args.alpha)

        total.backward()
        optimizer.step()
        scheduler.step()

        iteration += 1

        if iteration % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            # Log the actual percentage contribution for transparency
            with torch.no_grad():
                scale_val = t_loss.detach() / (f_loss.detach() + 1e-8)
                f_scaled_val = f_loss.detach() * scale_val
                teacher_contrib = (args.alpha * f_scaled_val) / (total.detach() + 1e-8)
            print(f"Iter {iteration:5d}/{args.max_iters} | "
                  f"loss={total.item():.4f}  "
                  f"task={t_loss.item():.4f}  "
                  f"feat={f_loss.item():.4f}  "
                  f"teacher%={teacher_contrib.item()*100:.1f}% | "
                  f"lr={lr_now:.5f}")

        # ==============================================================
        # VALIDATION
        # ==============================================================
        if iteration % args.val_interval == 0 or iteration == args.max_iters:
            student.eval()
            adapter.eval()

            # Accumulators for all metrics
            all_metrics: Dict[str, List[float]] = {
                "mDice":       [],
                "vessel_Dice": [],
                "mIoU":        [],
                "pixel_acc":   [],
                "mean_acc":    [],
            }

            with torch.no_grad():
                for raw_val in val_loader:
                    dv    = move_to_device(student, raw_val, device, training=False)
                    dv    = pad_to_divisor(dv, args.pad_divisor, args.ignore_index)
                    gt_v  = get_gt(dv["data_samples"], device)
                    enc_v = unet_encoder(student.backbone, dv["inputs"])
                    fts_v = unet_decoder(student.backbone, enc_v, enc_v[-1])
                    log_v = student.decode_head.forward(tuple(fts_v))

                    m = compute_all_metrics(
                        log_v, gt_v,
                        num_classes=2,
                        ignore_index=args.ignore_index,
                    )
                    for k, v in m.items():
                        all_metrics[k].append(v)

            avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
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
    # Best metrics come from the history entry at best_iter
    best_entry = next(
        (h for h in history if h["iter"] == best_iter),
        history[-1] if history else {}
    )

    summary = {
        "version":                "v3_normalised_alpha",
        "noise_type":             args.noise_type,
        "epsilon":                args.epsilon,
        "alpha":                  args.alpha,
        "teacher_pct":            f"{args.alpha*100:.0f}%",
        "task_pct":               f"{(1-args.alpha)*100:.0f}%",
        "seed":                   args.seed,
        "K":                      args.K,
        "max_iters":              args.max_iters,
        "best_iter":              best_iter,
        "best_mDice":             round(best_mdice, 6),
        "best_vessel_Dice":       round(best_entry.get("vessel_Dice", 0), 6),
        "best_mIoU":              round(best_entry.get("mIoU",        0), 6),
        "best_pixel_acc":         round(best_entry.get("pixel_acc",   0), 6),
        "best_mean_acc":          round(best_entry.get("mean_acc",    0), 6),
        "student_baseline_mDice": 0.8794,
        "lift_over_baseline":     round(best_mdice - 0.8794, 6),
        "adapter_channels":       f"{student_bn_channels}→{teacher_bn_channels}",
        "history":                history,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("v3 Training complete.")
    print(f"Noise type  : {args.noise_type}")
    print(f"Epsilon     : {args.epsilon}")
    print(f"Alpha       : {args.alpha} ({args.alpha*100:.0f}% teacher / {(1-args.alpha)*100:.0f}% task)")
    print(f"Seed        : {args.seed}")
    print(f"Best mDice  : {best_mdice*100:.2f}%  at iter {best_iter}")
    print(f"Best mIoU   : {best_entry.get('mIoU', 0)*100:.2f}%")
    print(f"Best pix_acc: {best_entry.get('pixel_acc', 0)*100:.2f}%")
    print(f"Baseline    : 87.94%")
    print(f"Lift        : {(best_mdice - 0.8794)*100:+.2f}%")
    print(f"Saved       : {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()