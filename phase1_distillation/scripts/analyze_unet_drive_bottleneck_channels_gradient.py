#!/usr/bin/env python
"""
Gradient-based bottleneck channel importance for an MMSegmentation U-Net teacher
on DRIVE.

This is the GRADIENT-ONLY version. It computes ONLY the gradient-based channel
importance score. L2 activation and ablation-based Dice drop are intentionally
removed.

Formula (from Lan & Tian, "Gradient-Guided Knowledge Distillation for Object
Detectors," WACV 2024, Eq. 1):

        w_k^l = (1 / (W * H)) * sum_{i, j} d L_task / d A_{i, j, k}^l

That is, for each channel k in the chosen layer l, take the gradient of the
task loss w.r.t. every spatial position in that channel, then average over the
spatial dimensions H and W (and over the batch). This script computes:

    gradient_signed_mean   = mean over (B, H, W) of   d L / d A
    gradient_abs_mean      = mean over (B, H, W) of  |d L / d A|

The signed mean reproduces Eq. 1 exactly. The absolute mean is the channel
importance magnitude used for ranking and for the downstream noise-injection
step (more noise on less important channels, less noise on more important
channels).

Important details kept from the original script:

  1. Target layer = TRUE bottleneck. UNet-S5-D16 returns a feature tuple where
     feature[0] is the deepest encoder output (the bottleneck) and feature[4]
     is the final decoder output (what decode_head consumes). We manually run
     encoder -> bottleneck -> decoder -> decode_head and take the gradient at
     the bottleneck. We do NOT just hook backbone output, because the gradient
     into the bottleneck must travel through the decoder and skip connections.

  2. Padding. DRIVE validation images become 584 x 565 after Resize, but the
     UNet-S5-D16 backbone requires H and W divisible by 16. We pad input
     images to the next multiple of 16 (typically 592 x 576) with value 0, and
     pad ground-truth masks with the ignore index (255) so padded pixels do
     not contribute to the loss.

  3. SyncBN. The training config uses SyncBN. For single-GPU analysis we
     convert SyncBN to standard BN via revert_sync_batchnorm.

This script does NOT inject noise. That is the next phase.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
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


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gradient-only bottleneck channel importance analysis for an "
            "MMSeg U-Net teacher on DRIVE. Implements GKD Eq. 1."
        )
    )

    parser.add_argument("--config", required=True, help="Path to teacher config .py file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained teacher checkpoint .pth file")
    parser.add_argument("--data-root", required=True, help="Path to DRIVE dataset root")
    parser.add_argument("--out-dir", required=True, help="Directory where CSV/JSON results will be saved")

    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Which dataloader split from the config to analyze",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Limit number of validation/test images. Use -1 for all.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=255,
        help="Segmentation ignore index",
    )
    parser.add_argument(
        "--foreground-class",
        type=int,
        default=1,
        help="Foreground class for vessel Dice. DRIVE vessel class is usually 1.",
    )
    parser.add_argument(
        "--topk-print",
        type=int,
        default=20,
        help="How many top/bottom channels to print in summary.",
    )
    parser.add_argument(
        "--pad-divisor",
        type=int,
        default=16,
        help="Pad input images and GT masks so H and W are divisible by this value.",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Helpers: config, model, data
# ----------------------------------------------------------------------


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_data_root(cfg: Config, data_root: str) -> None:
    """Override val/test dataset data_root without changing the config file."""
    if hasattr(cfg, "data_root"):
        cfg.data_root = data_root

    for key in ["val_dataloader", "test_dataloader"]:
        if hasattr(cfg, key) and "dataset" in cfg[key]:
            cfg[key]["dataset"]["data_root"] = data_root

    if hasattr(cfg, "train_dataloader"):
        train_ds = cfg.train_dataloader.get("dataset", None)
        if train_ds is not None:
            if train_ds.get("type", None) == "RepeatDataset" and "dataset" in train_ds:
                train_ds["dataset"]["data_root"] = data_root
            elif "data_root" in train_ds:
                train_ds["data_root"] = data_root


def build_model(cfg: Config, checkpoint: str, device: str):
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = MODELS.build(cfg.model)

    # Training config uses SyncBN. For single-GPU analysis, convert it.
    if revert_sync_batchnorm is not None:
        model = revert_sync_batchnorm(model)

    load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def build_dataset_and_loader(cfg: Config, split: str, num_workers: int) -> DataLoader:
    dataloader_cfg = cfg.val_dataloader if split == "val" else cfg.test_dataloader
    dataset = DATASETS.build(dataloader_cfg["dataset"])

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=pseudo_collate,
    )
    return loader


def move_batch_to_device(model, data: Dict, device: str) -> Dict:
    """Use MMSeg data_preprocessor so inputs/data_samples match model expectations."""
    data = model.data_preprocessor(data, training=False)
    data["inputs"] = data["inputs"].to(device)
    return data


def get_gt_tensor(data_samples: Sequence, device: str) -> torch.Tensor:
    gts = []
    for sample in data_samples:
        gt = sample.gt_sem_seg.data.squeeze(0).long().to(device)
        gts.append(gt)
    return torch.stack(gts, dim=0)


def pad_inputs_and_gt_to_divisor(
    data: Dict,
    divisor: int = 16,
    ignore_index: int = 255,
) -> Dict:
    """
    Pad input image tensor and GT masks so H and W are divisible by `divisor`.

    DRIVE validation images are resized to 584 x 565.
    UNet-S5-D16 requires H and W divisible by 16.
    This pads 584 x 565 to 592 x 576.

    Image padding value: 0.
    GT padding value: ignore_index, usually 255.

    We replace the whole PixelData object instead of mutating .data in place,
    because PixelData rejects shape-changing in-place edits.
    """
    inputs = data["inputs"]
    _, _, h, w = inputs.shape

    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    if pad_h == 0 and pad_w == 0:
        return data

    new_h = h + pad_h
    new_w = w + pad_w

    data["inputs"] = F.pad(
        inputs,
        (0, pad_w, 0, pad_h),
        mode="constant",
        value=0,
    )

    for sample in data["data_samples"]:
        old_gt = sample.gt_sem_seg.data

        padded_gt = F.pad(
            old_gt,
            (0, pad_w, 0, pad_h),
            mode="constant",
            value=ignore_index,
        )

        sample.gt_sem_seg = PixelData(data=padded_gt)

        sample.set_metainfo(
            {
                "pad_shape": (new_h, new_w),
                "img_shape": (new_h, new_w),
            }
        )

    print(f"Padded input and GT from {(h, w)} to {(new_h, new_w)} for divisor {divisor}.")
    return data


# ----------------------------------------------------------------------
# Helpers: loss, metrics, decode head
# ----------------------------------------------------------------------


def resize_logits_to_gt(model, logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] == gt.shape[-2:]:
        return logits

    align_corners = getattr(model.decode_head, "align_corners", False)
    return F.interpolate(
        logits,
        size=gt.shape[-2:],
        mode="bilinear",
        align_corners=align_corners,
    )


def parse_total_loss(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    total = None

    for name, value in loss_dict.items():
        if "loss" not in name:
            continue

        if isinstance(value, (list, tuple)):
            value = sum(v.mean() for v in value)
        else:
            value = value.mean()

        total = value if total is None else total + value

    if total is None:
        raise RuntimeError(f"No loss terms found in loss_dict keys: {list(loss_dict.keys())}")

    return total


def dice_for_class_from_pred(
    pred: torch.Tensor,
    gt: torch.Tensor,
    cls: int = 1,
    ignore_index: int = 255,
    eps: float = 1e-7,
) -> torch.Tensor:
    valid = gt != ignore_index
    pred_cls = (pred == cls) & valid
    gt_cls = (gt == cls) & valid

    dims = tuple(range(1, gt.ndim))
    inter = (pred_cls & gt_cls).sum(dim=dims).float()
    denom = pred_cls.sum(dim=dims).float() + gt_cls.sum(dim=dims).float()
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


def dice_for_class_from_logits(
    model,
    logits: torch.Tensor,
    gt: torch.Tensor,
    cls: int,
    ignore_index: int,
) -> torch.Tensor:
    logits = resize_logits_to_gt(model, logits, gt)
    pred = logits.argmax(dim=1)
    return dice_for_class_from_pred(pred, gt, cls=cls, ignore_index=ignore_index)


def decode_from_feats(model, feats: Sequence[torch.Tensor]) -> torch.Tensor:
    return model.decode_head.forward(tuple(feats))


def get_decode_in_index(model) -> int:
    in_index = getattr(model.decode_head, "in_index", None)

    if isinstance(in_index, int):
        return in_index

    if isinstance(in_index, (list, tuple)) and len(in_index) == 1:
        return int(in_index[0])

    raise RuntimeError(f"Unsupported decode_head.in_index for this analysis: {in_index}")


# ----------------------------------------------------------------------
# Manual U-Net forward (encoder -> bottleneck -> decoder)
# ----------------------------------------------------------------------


def assert_mmseg_unet_backbone(model) -> None:
    backbone = model.backbone
    required = ["encoder", "decoder"]
    missing = [name for name in required if not hasattr(backbone, name)]

    if missing:
        raise RuntimeError(
            "This script expected an MMSeg UNet backbone with encoder/decoder attributes. "
            f"Missing: {missing}. Backbone type: {type(backbone)}"
        )


def unet_encoder_forward(backbone, x: torch.Tensor) -> List[torch.Tensor]:
    """
    Run MMSeg UNet encoder.

    For UNet-S5-D16:
        enc_outs[-1] is the deepest bottleneck feature.
    """
    enc_outs = []

    for enc in backbone.encoder:
        x = enc(x)
        enc_outs.append(x)

    return enc_outs


def unet_decoder_forward_from_bottleneck(
    backbone,
    enc_outs: Sequence[torch.Tensor],
    bottleneck: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Run MMSeg UNet decoder starting from a chosen bottleneck tensor.

    Returned list order follows MMSeg UNet conventions:
        dec_outs[0] = bottleneck
        dec_outs[1] = decoder stage 1
        dec_outs[2] = decoder stage 2
        dec_outs[3] = decoder stage 3
        dec_outs[4] = final decoder feature (decode_head consumes this)
    """
    x = bottleneck
    dec_outs = [x]

    for i in reversed(range(len(backbone.decoder))):
        x = backbone.decoder[i](enc_outs[i], x)
        dec_outs.append(x)

    return dec_outs


def clean_bottleneck_forward(model, inputs: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Forward path: input -> encoder -> bottleneck -> decoder -> decode_head feats.

    Returns:
        feats: list of features in decode_head order (feats[4] is what
               decode_head with in_index=4 consumes).
        bottleneck: the deepest encoder feature (the channel-importance target).
    """
    assert_mmseg_unet_backbone(model)

    enc_outs = unet_encoder_forward(model.backbone, inputs)
    bottleneck = enc_outs[-1]
    feats = unet_decoder_forward_from_bottleneck(model.backbone, enc_outs, bottleneck)

    return feats, bottleneck


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    cfg = Config.fromfile(args.config)
    set_data_root(cfg, args.data_root)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False. "
            "Use --device cpu or check GPU allocation."
        )

    model = build_model(cfg, args.checkpoint, device)
    loader = build_dataset_and_loader(cfg, args.split, args.num_workers)

    decode_in_index = get_decode_in_index(model)

    print("=" * 80)
    print("GRADIENT-ONLY bottleneck channel importance analysis  (GKD Eq. 1)")
    print(f"Config:       {args.config}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Data root:    {args.data_root}")
    print(f"Split:        {args.split}")
    print(f"Decode index: {decode_in_index}")
    print(f"Pad divisor:  {args.pad_divisor}")
    print("=" * 80)

    n_images = 0
    channel_count = None

    grad_signed_sum = None   # accumulates  mean_{B,H,W}( dL/dA )
    grad_abs_sum = None      # accumulates  mean_{B,H,W}( |dL/dA| )

    baseline_dice_sum = 0.0
    per_image_rows: List[Dict] = []

    for batch_idx, raw_data in enumerate(loader):
        if args.max_samples > 0 and batch_idx >= args.max_samples:
            break

        data = move_batch_to_device(model, raw_data, device)
        data = pad_inputs_and_gt_to_divisor(
            data,
            divisor=args.pad_divisor,
            ignore_index=args.ignore_index,
        )

        inputs = data["inputs"]
        data_samples = data["data_samples"]
        gt = get_gt_tensor(data_samples, device)

        model.zero_grad(set_to_none=True)

        # Forward + backward to get bottleneck gradient
        with torch.enable_grad():
            feats, bottleneck = clean_bottleneck_forward(model, inputs)
            bottleneck.retain_grad()

            if channel_count is None:
                channel_count = bottleneck.shape[1]
                grad_signed_sum = torch.zeros(channel_count, device=device)
                grad_abs_sum = torch.zeros(channel_count, device=device)

                print("Feature shapes from manual U-Net forward:")
                for i, f in enumerate(feats):
                    marker = "  <-- bottleneck target" if i == 0 else ""
                    print(f"  feature[{i}]: {tuple(f.shape)}{marker}")
                print(f"Analyzing bottleneck channels: C={channel_count}")

            # Clean prediction logits, for baseline Dice and for the loss
            logits = decode_from_feats(model, feats)

            baseline_dice = dice_for_class_from_logits(
                model,
                logits,
                gt,
                cls=args.foreground_class,
                ignore_index=args.ignore_index,
            )
            baseline_dice_sum += float(baseline_dice.detach().cpu())

            # Task loss = the same losses decode_head was trained with
            loss_dict = model.decode_head.loss_by_feat(logits, data_samples)
            total_loss = parse_total_loss(loss_dict)
            total_loss.backward()

            if bottleneck.grad is None:
                raise RuntimeError(
                    "bottleneck.grad is None. The bottleneck is not connected to the decode loss."
                )

            # GKD Eq. 1: mean over spatial dims of dL/dA.
            # Here we additionally average over the batch (B=1, so it is a no-op).
            grad = bottleneck.grad.detach()                # shape (B, C, H, W)
            grad_signed_per_ch = grad.mean(dim=(0, 2, 3))  # signed   per channel
            grad_abs_per_ch = grad.abs().mean(dim=(0, 2, 3))  # absolute per channel

            grad_signed_sum += grad_signed_per_ch
            grad_abs_sum += grad_abs_per_ch

        n_images += 1
        per_image_rows.append(
            {
                "image_index": batch_idx,
                "baseline_vessel_dice_direct_path": float(baseline_dice.detach().cpu()),
            }
        )

        if (batch_idx + 1) % 5 == 0:
            print(f"Processed {batch_idx + 1} images...")

    if n_images == 0:
        raise RuntimeError("No images were processed. Check dataset path/split.")

    # ------------------------------------------------------------------
    # Average over images
    # ------------------------------------------------------------------
    grad_signed_avg = (grad_signed_sum / n_images).detach().cpu()
    grad_abs_avg = (grad_abs_sum / n_images).detach().cpu()
    baseline_dice_avg = baseline_dice_sum / n_images

    # Rank channels by importance magnitude (absolute gradient).
    # Higher abs gradient => channel matters more => less noise in the next phase.
    importance = grad_abs_avg.clone()

    # Min-max normalize the importance for convenience downstream.
    imp_min = importance.min()
    imp_max = importance.max()
    if float(imp_max - imp_min) < 1e-12:
        importance_norm = torch.zeros_like(importance)
    else:
        importance_norm = (importance - imp_min) / (imp_max - imp_min)

    sorted_idx = torch.argsort(importance, descending=True)
    topk = min(args.topk_print, channel_count)

    # ------------------------------------------------------------------
    # Save channel CSV
    # ------------------------------------------------------------------
    channel_csv = out_dir / "channel_importance_scores_bottleneck_gradient.csv"

    with channel_csv.open("w", newline="") as f:
        fieldnames = [
            "channel",
            "gradient_signed_mean",   # GKD paper Eq. 1, signed
            "gradient_abs_mean",      # magnitude, used for ranking
            "importance_norm_0_1",    # gradient_abs_mean min-max normalized
            "rank_by_abs",            # 0 = most important, channel_count-1 = least important
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Build a rank lookup: rank_by_abs[ch] = position when sorted by importance desc
        rank_by_abs = [0] * channel_count
        for rank, ch in enumerate(sorted_idx.tolist()):
            rank_by_abs[ch] = rank

        for ch in range(channel_count):
            writer.writerow(
                {
                    "channel": ch,
                    "gradient_signed_mean": float(grad_signed_avg[ch]),
                    "gradient_abs_mean": float(grad_abs_avg[ch]),
                    "importance_norm_0_1": float(importance_norm[ch]),
                    "rank_by_abs": rank_by_abs[ch],
                }
            )

    # ------------------------------------------------------------------
    # Save per-image CSV
    # ------------------------------------------------------------------
    per_image_csv = out_dir / "per_image_baseline_dice_bottleneck_gradient.csv"

    with per_image_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_index",
                "baseline_vessel_dice_direct_path",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    # ------------------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------------------
    summary = {
        "method": "gradient_only_GKD_Eq1",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "split": args.split,
        "target_layer": "true_bottleneck",
        "decode_head_in_index": decode_in_index,
        "num_images": n_images,
        "num_channels": channel_count,
        "baseline_vessel_dice_direct_path": baseline_dice_avg,
        "ranking_metric": "gradient_abs_mean",
        "top_channels_by_abs_gradient": [int(x) for x in sorted_idx[:topk].tolist()],
        "bottom_channels_by_abs_gradient": [int(x) for x in sorted_idx[-topk:].tolist()],
        "formula_note": (
            "gradient_signed_mean implements GKD Eq. 1 from Lan & Tian (WACV 2024): "
            "w_k = (1 / (W*H)) * sum_{i,j} dL_task/dA_{i,j,k}. "
            "gradient_abs_mean is mean_{H,W}(|dL/dA|), used to rank channels by "
            "importance magnitude for the downstream noise-injection step."
        ),
        "outputs": {
            "channel_csv": str(channel_csv),
            "per_image_csv": str(per_image_csv),
        },
    }

    summary_json = out_dir / "summary_bottleneck_gradient.json"

    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Done.")
    print("Target layer: TRUE bottleneck")
    print(f"Images analyzed:   {n_images}")
    print(f"Channels analyzed: {channel_count}")
    print(f"Baseline vessel Dice (direct path): {baseline_dice_avg:.6f}")
    print(f"Saved: {channel_csv}")
    print(f"Saved: {per_image_csv}")
    print(f"Saved: {summary_json}")

    print("\nTop channels by |gradient| (most important):")
    print([int(x) for x in sorted_idx[:topk].tolist()])

    print("\nBottom channels by |gradient| (least important):")
    print([int(x) for x in sorted_idx[-topk:].tolist()])
    print("=" * 80)


if __name__ == "__main__":
    main()