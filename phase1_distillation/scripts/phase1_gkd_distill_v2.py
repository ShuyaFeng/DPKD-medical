#!/usr/bin/env python
"""
GKD Student Distillation v2 — Direct Feature Matching on DRIVE.

What changed from v1
----------------------
v1 used GKD attention maps (Eq 1-4 from the paper):
  - Collapsed 1024 teacher channels into one (37,36) spatial map M^T
  - Collapsed 512 student channels into one (37,36) spatial map M^S
  - L1 loss between M^T and M^S
  Problem: collapsing all channels into one map hides the per-channel
  noise difference between uniform and channel_WF. The student never
  sees which individual channels are noisy and which are clean.

v2 uses direct feature matching:
  - Teacher noisy bottleneck: (B, 1024, 37, 36)
  - Student bottleneck:       (B,  512, 37, 36)
  - Adapter (1x1 conv):       (B,  512, 37, 36) -> (B, 1024, 37, 36)
  - Feature loss: MSE(adapter(student_bn), noisy_teacher_bn)
  Benefit: the student sees per-channel noise directly. Under channel_WF
  the important teacher channels are less noisy, so the student can match
  them more accurately, leading to better learning from important channels.

Also added: --seed argument so each condition can be run 3 times
with different seeds and averaged for statistical reliability.

Pipeline (every training iteration)
--------------------------------------
1. Teacher encoder  → clean bottleneck (B, 1024, 37, 36)  [frozen]
2. Add noise to teacher bottleneck (uniform or channel_WF)
   → noisy teacher bottleneck
3. Student encoder  → student bottleneck (B, 512, 37, 36)
4. Adapter (1x1 conv, trainable) → projected student (B, 1024, 37, 36)
5. Feature matching loss = MSE(projected_student, noisy_teacher_bn)
6. Student decoder  → student logits
7. Task loss = CE + 3*Dice (student logits vs GT)
8. Total loss = task_loss + lambda_feat * feature_loss
9. Backward + update student + adapter (teacher frozen)

Run one job per noise type per epsilon per seed:
  python phase1_gkd_distill_v2.py --noise-type uniform    --epsilon 2 --seed 0
  python phase1_gkd_distill_v2.py --noise-type channel_WF --epsilon 2 --seed 0
  ... etc

Total recommended runs: 2 noise types x 4 epsilons x 3 seeds = 24 jobs
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
# Normalisation (same as all previous scripts)
# ---------------------------------------------------------------------------

def clip_and_normalise(
    bottleneck: torch.Tensor,
    caps: torch.Tensor,
) -> torch.Tensor:
    """
    Per-channel L2 clipping then normalisation.
    Works for any batch size B.
    After this call every channel has L2 norm in [0, 1].
    """
    B, C, H, W = bottleneck.shape
    norms = bottleneck.flatten(2).norm(dim=2)               # (B, C)
    scale = (caps.view(1, C) / norms.clamp(min=1e-12)).clamp(max=1.0)
    clipped = bottleneck * scale.view(B, C, 1, 1)
    normalised = clipped / caps.view(1, C, 1, 1).clamp(min=1e-12)
    return normalised


def denormalise(normalised: torch.Tensor, caps: torch.Tensor) -> torch.Tensor:
    _, C, H, W = normalised.shape
    return normalised * caps.view(1, C, 1, 1)


# ---------------------------------------------------------------------------
# Adapter  (the key new component in v2)
# ---------------------------------------------------------------------------

class BottleneckAdapter(nn.Module):
    """
    1x1 convolution adapter: student channels → teacher channels.

    Maps student bottleneck (512 channels) to the same channel dimension
    as the teacher bottleneck (1024 channels) so that direct feature
    matching (MSE) is possible without any spatial dimension change.

    This is a standard technique in feature-level knowledge distillation.
    The adapter is trained alongside the student — its weights are included
    in the same optimizer.

    Why 1x1 convolution?
    A 1x1 conv mixes channel information (learns which student channels
    correspond to which teacher channels) without touching spatial structure.
    It is the lightest possible adapter for this purpose.
    """
    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            student_channels,
            teacher_channels,
            kernel_size=1,
            bias=False,
        )
        # Initialise with small random weights
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out",
                                nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, student_channels, H, W) → (B, teacher_channels, H, W)"""
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
    """CE + 3×Dice — matches teacher training config."""
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
    """
    Direct feature matching loss (v2 replacement for GKD map L1).

    MSE between the adapter-projected student bottleneck and the
    noisy teacher bottleneck. Both tensors have shape (B, 1024, H, W).

    Why MSE?
    MSE penalises large channel-wise differences more strongly than L1.
    Under channel_WF, the important teacher channels are less noisy
    (cleaner signal). The student adapter learns to match those clean
    important channels better, gaining more useful information than
    when trained with uniform noise (where all channels are equally noisy).

    The teacher tensor is detached — we do not backprop into the teacher.
    """
    return F.mse_loss(
        projected_student,
        noisy_teacher_bn.detach(),
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mdice_metric(
    logits: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = 255,
    eps: float = 1e-7,
) -> float:
    """Mean Dice across all classes (matches mmengine IoUMetric mDice)."""
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:],
                               mode="bilinear", align_corners=False)
    pred  = logits.argmax(dim=1)
    valid = gt != ignore_index
    dices = []
    for cls in range(num_classes):
        pc = (pred == cls) & valid
        gc = (gt   == cls) & valid
        dims  = tuple(range(1, gt.ndim))
        inter = (pc & gc).sum(dim=dims).float()
        denom = pc.sum(dim=dims).float() + gc.sum(dim=dims).float()
        dices.append(((2.0 * inter + eps) / (denom + eps)).mean().item())
    return float(np.mean(dices))


def vessel_dice_metric(
    logits: torch.Tensor,
    gt: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-7,
) -> float:
    """Hard Dice on vessel class only (class=1)."""
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits, size=gt.shape[-2:],
                               mode="bilinear", align_corners=False)
    pred  = logits.argmax(dim=1)
    valid = gt != ignore_index
    pv = (pred == 1) & valid
    gv = (gt   == 1) & valid
    dims  = tuple(range(1, gt.ndim))
    inter = (pv & gv).sum(dim=dims).float()
    denom = pv.sum(dim=dims).float() + gv.sum(dim=dims).float()
    return float(((2.0 * inter + eps) / (denom + eps)).mean().item())


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
        description="GKD v2: direct feature matching with adapter on DRIVE."
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
    p.add_argument("--K",                  type=int,   default=1)
    p.add_argument("--delta-dp",           type=float, default=1e-5)
    p.add_argument("--lambda-feat",        type=float, default=0.4,
                   help="Weight of feature matching loss in total loss.")
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
    # Build adapter  (student channels → teacher channels)
    # ------------------------------------------------------------------
    # Get channel sizes from configs
    student_bn_channels = s_cfg.model.backbone.base_channels * (2 ** 4)  # 32*16=512
    teacher_bn_channels = t_cfg.model.backbone.base_channels * (2 ** 4)  # 64*16=1024

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
    # ------------------------------------------------------------------
    # SENSITIVITY FIX (was the results-invalidating bug)
    # ------------------------------------------------------------------
    # The Gaussian noise is added in the *normalised* space: see the
    # training loop, where clip_and_normalise() forces every channel's
    # L2 norm to <= 1 BEFORE the noise is added, and denormalise() (the
    # post-noise *caps) is pure post-processing that does not affect DP.
    #
    # In that normalised space the replace-one per-channel L2 sensitivity
    # is a data-independent constant 2/K (norm <= 1 each side -> diff <= 2),
    # NOT 2*caps/K. The old `deltas = 2*caps/K` double-counted `caps`
    # (once dividing the signal inside normalise, once inflating sigma
    # here), so sigma was ~caps x too large (caps ~ tens-hundreds for a
    # 1332-dim channel vector). That drowned the teacher target in noise
    # at *every* epsilon, collapsing every run to the student baseline and
    # making uniform indistinguishable from channel_WF.
    deltas = torch.full_like(caps, 2.0 / args.K)

    print(f"Cap stats: min={caps.min():.3f}  max={caps.max():.3f}  "
          f"mean={caps.mean():.3f}  (used only for clip/normalise, NOT for sensitivity)")
    print(f"K={args.K}  delta_c = 2/K (normalised-space L2 sensitivity) "
          f"= {2.0 / args.K:.4f}  (data-independent, constant across channels)")

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
    print(f"Sigma      : min={sigma.min():.4f}  max={sigma.max():.4f}  "
          f"mean={sigma.mean():.4f}")

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
    # Optimizer — includes both student AND adapter parameters
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
    print(f"v2: direct feature matching | "
          f"noise={args.noise_type} | eps={args.epsilon} | seed={args.seed}")
    print(f"lambda_feat={args.lambda_feat} | "
          f"max_iters={args.max_iters} | val_interval={args.val_interval}")
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

        # fetch batch
        try:
            raw_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            raw_data   = next(train_iter)

        data   = move_to_device(student, raw_data, device, training=True)
        inputs = data["inputs"]
        gt     = get_gt(data["data_samples"], device)

        # ==============================================================
        # TEACHER PATH — frozen, no grad on teacher parameters
        # ==============================================================
        with torch.no_grad():
            t_enc              = unet_encoder(teacher.backbone, inputs)
            t_bottleneck_clean = t_enc[-1]               # (B, 1024, H_b, W_b)

            # Clip + normalise → add noise → denormalise
            t_bn_norm = clip_and_normalise(t_bottleneck_clean, caps)
            B, C_t, H_b, W_b = t_bn_norm.shape
            noise = (
                torch.randn(B, C_t, H_b, W_b, device=device)
                * sigma.view(1, C_t, 1, 1)
            )
            t_bn_noisy = denormalise(t_bn_norm + noise, caps)
            # t_bn_noisy: (B, 1024, H_b, W_b) — this is the distillation target

        # ==============================================================
        # STUDENT PATH
        # ==============================================================
        optimizer.zero_grad()

        # Student encoder → bottleneck
        s_enc       = unet_encoder(student.backbone, inputs)
        s_bottleneck = s_enc[-1]                         # (B, 512, H_b, W_b)

        # Adapter: project student bottleneck to teacher channel space
        s_projected = adapter(s_bottleneck)              # (B, 1024, H_b, W_b)

        # Feature matching loss (v2 key change)
        # MSE between projected student and noisy teacher bottleneck
        # Under channel_WF: important teacher channels are LESS noisy
        # → student adapter learns better from those cleaner channels
        # Under uniform: all channels equally noisy → less signal to learn from
        f_loss = feature_matching_loss(s_projected, t_bn_noisy)

        # Student decoder → logits → task loss
        s_feats  = unet_decoder(student.backbone, s_enc, s_bottleneck)
        s_logits = student.decode_head.forward(tuple(s_feats))
        t_loss   = task_loss_fn(s_logits, gt, args.ignore_index)

        # Total loss
        total = t_loss + args.lambda_feat * f_loss

        total.backward()
        optimizer.step()
        scheduler.step()

        iteration += 1

        if iteration % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Iter {iteration:5d}/{args.max_iters} | "
                  f"loss={total.item():.4f}  "
                  f"task={t_loss.item():.4f}  "
                  f"feat={f_loss.item():.4f} | "
                  f"lr={lr_now:.5f}")

        # ==============================================================
        # VALIDATION
        # ==============================================================
        if iteration % args.val_interval == 0 or iteration == args.max_iters:
            student.eval()
            adapter.eval()
            mdice_list  = []
            vdice_list  = []

            with torch.no_grad():
                for raw_val in val_loader:
                    dv     = move_to_device(student, raw_val, device, training=False)
                    dv     = pad_to_divisor(dv, args.pad_divisor, args.ignore_index)
                    gt_v   = get_gt(dv["data_samples"], device)
                    inp_v  = dv["inputs"]
                    enc_v  = unet_encoder(student.backbone, inp_v)
                    fts_v  = unet_decoder(student.backbone, enc_v, enc_v[-1])
                    log_v  = student.decode_head.forward(tuple(fts_v))
                    mdice_list.append(mdice_metric(log_v, gt_v,
                                                   ignore_index=args.ignore_index))
                    vdice_list.append(vessel_dice_metric(log_v, gt_v,
                                                         ignore_index=args.ignore_index))

            avg_mdice = float(np.mean(mdice_list))
            avg_vdice = float(np.mean(vdice_list))

            print(f"\n>>> Val iter {iteration}: "
                  f"mDice={avg_mdice*100:.2f}%  "
                  f"vessel_Dice={avg_vdice*100:.2f}%")

            history.append({
                "iter":        iteration,
                "mDice":       round(avg_mdice, 6),
                "vessel_Dice": round(avg_vdice, 6),
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
    summary = {
        "version":                  "v2_direct_feature_matching",
        "noise_type":               args.noise_type,
        "epsilon":                  args.epsilon,
        "seed":                     args.seed,
        "K":                        args.K,
        "lambda_feat":              args.lambda_feat,
        "max_iters":                args.max_iters,
        "best_mDice":               round(best_mdice, 6),
        "best_iter":                best_iter,
        "student_baseline_mDice":   0.8791,
        "lift_over_baseline":       round(best_mdice - 0.8791, 6),
        "adapter_channels":         f"{student_bn_channels}→{teacher_bn_channels}",
        "history":                  history,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("v2 Training complete.")
    print(f"Noise type  : {args.noise_type}")
    print(f"Epsilon     : {args.epsilon}")
    print(f"Seed        : {args.seed}")
    print(f"Best mDice  : {best_mdice*100:.2f}%  at iter {best_iter}")
    print(f"Baseline    : 87.91%")
    print(f"Lift        : {(best_mdice - 0.8791)*100:+.2f}%")
    print(f"Saved       : {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()