#!/usr/bin/env python
"""
Phase 1 — Channel-wise noise experiment on the DRIVE U-Net teacher.
Corrected version: features are normalised before noise injection.

What changed from the first version
-------------------------------------
1. Per-channel L2 clipping (normalisation).
   Before adding noise, each channel is divided by its cap_c so its
   L2 norm becomes at most 1.0. After adding noise, the channel is
   multiplied back by cap_c. This matches what demo_noise.py does with
   clip_channels(). Without this step, the raw feature norms (~20-60)
   make the sensitivity delta_c enormous, which forces sigma into the
   thousands — destroying all useful signal.

2. K = 10 (simulated teachers).
   The sensitivity formula is delta_c = 2 * cap_c / K.
   demo_noise.py uses K=10 (10 hospitals / teachers).
   With K=1 (our old value), delta_c was 10x too large, making sigma
   ~3x too large (since sigma ∝ sqrt(delta_c)).
   We simulate K=10 even though we have one teacher — this represents
   the PATE setting where 10 hospitals each contribute a teacher and
   the sensitivity per teacher is 1/K of the full cap.

3. Wider epsilon range: {1, 2, 4, 8, 16, 32}.
   The original {1, 2, 4, 8} may all land in the "too noisy" regime.
   The wider range lets us see the full curve: where noise is severe,
   where channel-WF starts beating uniform, and where both recover.

Pipeline (per image)
----------------------
Pass 1 (clean):
  image → encoder → bottleneck (1024, 37, 36)
        → decoder → Dice score   [no noise, upper bound]
  Also collect per-channel L2 norms for cap estimation.

After pass 1:
  cap_c   = 90th-percentile per-channel norm across all images
  delta_c = 2 * cap_c / K

Pass 2 (noisy, for each epsilon):
  Compute sigma_c for each mechanism:
    - uniform:     all channels get the same sigma
    - channel_WF:  sigma_c ∝ sqrt(delta_c) / s_c^{1/4}  (water-filling)
  For each image:
    1. bottleneck → clip to cap_c → normalised (norm ≤ 1 per channel)
    2. add noise N(0, sigma_c^2) to normalised channel
    3. denormalise (multiply back by cap_c)
    4. run decoder → Dice score

Outputs
--------
  phase1_channel_noise_results_v2.csv    per-image, per-epsilon, per-mechanism Dice
  phase1_channel_noise_summary_v2.json   mean Dice and lift over uniform
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
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


# ---------------------------------------------------------------------------
# Privacy accounting  (exact same formula as phase0_validation.py)
# ---------------------------------------------------------------------------

def eps_to_rho(eps: float, delta: float = 1e-5) -> float:
    """
    Convert (epsilon, delta)-DP to rho-zCDP via Bun-Steinke.
    eps = rho + 2 * sqrt(rho * log(1/delta))
    Solved as a quadratic in sqrt(rho).
    """
    log_inv_d = math.log(1.0 / delta)
    b = 2.0 * math.sqrt(log_inv_d)
    discriminant = b * b + 4.0 * eps
    sqrt_rho = (-b + math.sqrt(discriminant)) / 2.0
    return sqrt_rho * sqrt_rho


# ---------------------------------------------------------------------------
# Noise allocation  (same formula as demo_noise.py waterfilling_sigma)
# ---------------------------------------------------------------------------

def waterfilling_sigma(
    deltas: torch.Tensor,
    importances: torch.Tensor,
    rdp_budget: float,
    eps_s: float = 1e-12,
) -> torch.Tensor:
    """
    Closed-form channel-WF allocation.

    sigma_c = kappa * sqrt(delta_c) / s_c^{1/4}
    kappa^2 = sum_c(delta_c * sqrt(s_c)) / rdp_budget

    Important channels (high s_c) → small sigma_c → less noise.
    Unimportant channels (low s_c) → large sigma_c → more noise.
    """
    s = importances.clamp(min=eps_s)
    kappa = ((deltas * s.sqrt()).sum() / rdp_budget).sqrt()
    return kappa * deltas.sqrt() / s.pow(0.25)


def uniform_sigma(deltas: torch.Tensor, rdp_budget: float) -> torch.Tensor:
    """
    Uniform allocation — same sigma for every channel.
    sum_c (delta_c^2 / sigma^2) = rdp_budget
    => sigma = sqrt(sum_c delta_c^2 / rdp_budget)
    """
    sigma = (deltas.pow(2).sum() / rdp_budget).sqrt()
    return sigma.expand(deltas.shape[0]).clone()


# ---------------------------------------------------------------------------
# MMSeg utilities
# ---------------------------------------------------------------------------

def set_data_root(cfg: Config, data_root: str) -> None:
    if hasattr(cfg, "data_root"):
        cfg.data_root = data_root
    for key in ["val_dataloader", "test_dataloader"]:
        if hasattr(cfg, key) and "dataset" in cfg[key]:
            cfg[key]["dataset"]["data_root"] = data_root


def build_model(cfg: Config, checkpoint: str, device: str):
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = MODELS.build(cfg.model)
    if revert_sync_batchnorm is not None:
        model = revert_sync_batchnorm(model)
    load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def build_loader(cfg: Config, num_workers: int) -> DataLoader:
    dataset = DATASETS.build(cfg.val_dataloader["dataset"])
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=pseudo_collate,
    )


def move_to_device(model, data: Dict, device: str) -> Dict:
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


def get_bottleneck(model, inputs: torch.Tensor) -> Tuple[List, torch.Tensor]:
    """Run encoder only. Returns enc_outs and bottleneck."""
    enc_outs = unet_encoder(model.backbone, inputs)
    return enc_outs, enc_outs[-1]


def decode_and_predict(
    model,
    enc_outs: List[torch.Tensor],
    bottleneck: torch.Tensor,
) -> torch.Tensor:
    """Run decoder + decode_head from a given bottleneck. Returns logits."""
    # Replace last enc_out with the (possibly noisy) bottleneck
    enc_outs_for_dec = enc_outs[:-1] + [bottleneck]
    feats = unet_decoder(model.backbone, enc_outs_for_dec[:-1], bottleneck)
    return model.decode_head.forward(tuple(feats))


# ---------------------------------------------------------------------------
# Dice metric
# ---------------------------------------------------------------------------

def dice_score(
    logits: torch.Tensor,
    gt: torch.Tensor,
    model,
    cls: int = 1,
    ignore_index: int = 255,
    eps: float = 1e-7,
) -> float:
    if logits.shape[-2:] != gt.shape[-2:]:
        align = getattr(model.decode_head, "align_corners", False)
        logits = F.interpolate(
            logits, size=gt.shape[-2:],
            mode="bilinear", align_corners=align,
        )
    pred = logits.argmax(dim=1)
    valid = gt != ignore_index
    pred_cls = (pred == cls) & valid
    gt_cls   = (gt   == cls) & valid
    dims = tuple(range(1, gt.ndim))
    inter = (pred_cls & gt_cls).sum(dim=dims).float()
    denom = pred_cls.sum(dim=dims).float() + gt_cls.sum(dim=dims).float()
    return float(((2.0 * inter + eps) / (denom + eps)).mean().item())


# ---------------------------------------------------------------------------
# Normalisation helpers  (core fix)
# ---------------------------------------------------------------------------

def clip_and_normalise(
    bottleneck: torch.Tensor,
    caps: torch.Tensor,
) -> torch.Tensor:
    """
    Per-channel L2 clipping then normalisation.

    For each channel c:
      1. If the L2 norm of channel c > cap_c, scale it down so norm = cap_c.
         (This is L2 clipping — same as demo_noise.py clip_channels.)
      2. Divide by cap_c so the normalised channel has norm ≤ 1.

    After this function, every channel has L2 norm in [0, 1].
    This means the sensitivity of each channel (how much one patient
    can change it) is bounded by 2 / K regardless of the original scale.

    Args:
        bottleneck: (1, C, H, W)
        caps:       (C,) per-channel cap

    Returns:
        normalised: (1, C, H, W) with per-channel norm ≤ 1
    """
    _, C, H, W = bottleneck.shape
    # L2 norm of each channel: shape (1, C)
    norms = bottleneck.flatten(2).norm(dim=2)          # (1, C)
    # Scale factor: 1.0 if norm ≤ cap, else cap/norm
    scale = (caps.view(1, C) / norms.clamp(min=1e-12)).clamp(max=1.0)
    clipped = bottleneck * scale.view(1, C, 1, 1)
    # Normalise by cap so all channels are in [-1, 1] range
    normalised = clipped / caps.view(1, C, 1, 1).clamp(min=1e-12)
    return normalised


def denormalise(
    normalised_noisy: torch.Tensor,
    caps: torch.Tensor,
) -> torch.Tensor:
    """
    Reverse the normalisation: multiply each channel back by cap_c.

    Args:
        normalised_noisy: (1, C, H, W)
        caps:             (C,)

    Returns:
        (1, C, H, W) in original feature scale
    """
    _, C, H, W = normalised_noisy.shape
    return normalised_noisy * caps.view(1, C, 1, 1)


# ---------------------------------------------------------------------------
# Importance CSV loader
# ---------------------------------------------------------------------------

def load_importances(csv_path: str, column: str = "gradient_abs_mean") -> torch.Tensor:
    """Read per-channel gradient importance from the Phase 1 Step B CSV."""
    vals = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            vals.append(float(row[column]))
    return torch.tensor(vals, dtype=torch.float32).clamp(min=1e-12)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 1 (v2): channel-wise noise on DRIVE U-Net bottleneck. "
            "Features are normalised before noise injection."
        )
    )
    p.add_argument("--config",           required=True)
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--data-root",        required=True)
    p.add_argument("--importance-csv",   required=True,
                   help="channel_importance_scores_bottleneck_gradient.csv")
    p.add_argument("--out-dir",          required=True)
    p.add_argument("--device",           default="cuda:0")
    p.add_argument("--num-workers",      type=int, default=4)
    p.add_argument("--pad-divisor",      type=int, default=16)
    p.add_argument("--ignore-index",     type=int, default=255)
    p.add_argument("--foreground-class", type=int, default=1)
    p.add_argument("--delta-dp",         type=float, default=1e-5)
    p.add_argument("--K",                type=int, default=10,
                   help="Number of simulated teachers (K=10 matches demo_noise.py).")
    p.add_argument("--epsilons",         type=float, nargs="+",
                   default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    p.add_argument("--cap-quantile",     type=float, default=0.9,
                   help="Quantile for per-channel cap estimation.")
    p.add_argument("--seed",             type=int, default=42)
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

    # ------------------------------------------------------------------
    # Build model and data loader
    # ------------------------------------------------------------------
    cfg = Config.fromfile(args.config)
    set_data_root(cfg, args.data_root)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Use --device cpu.")

    model = build_model(cfg, args.checkpoint, args.device)
    loader = build_loader(cfg, args.num_workers)

    # ------------------------------------------------------------------
    # Load channel importance scores  (s_c in the water-filling formula)
    # ------------------------------------------------------------------
    importances = load_importances(args.importance_csv).to(args.device)
    C = importances.shape[0]
    print(f"Loaded importance scores for C={C} channels.")
    print(f"  min={importances.min():.3e}  "
          f"max={importances.max():.3e}  "
          f"mean={importances.mean():.3e}")

    # ------------------------------------------------------------------
    # PASS 1: clean forward — collect Dice and per-channel norms
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Pass 1: clean forward — collecting Dice + per-channel L2 norms")
    print("=" * 70)

    all_norms    : List[torch.Tensor] = []   # each (C,)
    clean_dice_list: List[float]       = []
    n_images = 0

    with torch.no_grad():
        for batch_idx, raw_data in enumerate(loader):
            data   = move_to_device(model, raw_data, args.device)
            data   = pad_to_divisor(data, args.pad_divisor, args.ignore_index)
            inputs = data["inputs"]
            gt     = get_gt(data["data_samples"], args.device)

            enc_outs, bottleneck = get_bottleneck(model, inputs)
            logits = decode_and_predict(model, enc_outs, bottleneck)

            # Per-channel L2 norm: shape (C,)
            norms = bottleneck.squeeze(0).flatten(1).norm(dim=1)
            all_norms.append(norms.cpu())

            clean_dice_list.append(
                dice_score(logits, gt, model,
                           cls=args.foreground_class,
                           ignore_index=args.ignore_index)
            )
            n_images += 1
            if (batch_idx + 1) % 5 == 0:
                print(f"  pass 1: {batch_idx + 1} / {n_images} done")

    clean_dice_avg = float(np.mean(clean_dice_list))
    print(f"\nClean Dice (no noise): {clean_dice_avg:.6f}")

    # ------------------------------------------------------------------
    # Estimate per-channel cap and sensitivity
    # ------------------------------------------------------------------
    # all_norms: (N_images, C)
    norms_tensor = torch.stack(all_norms, dim=0)

    # cap_c = 90th-percentile of per-channel norms across all images
    caps   = torch.quantile(norms_tensor, args.cap_quantile, dim=0).to(args.device)

    # Sensitivity: delta_c = 2 * cap_c / K
    # After normalisation each channel has norm ≤ 1, so effective cap is 1.
    # We still keep cap_c in deltas for the noise formula, but normalise
    # before adding noise and denormalise after — the net effect is that
    # delta_c used in the noise formula should be 2/K (unit-normalised scale).
    # We use delta_c = 2 / K to match the normalised feature space.
    deltas = torch.full((C,), 2.0 / args.K, device=args.device)

    print(f"\nPer-channel cap C_c  (90th pct norm): "
          f"min={caps.min():.3f}  max={caps.max():.3f}  mean={caps.mean():.3f}")
    print(f"K (simulated teachers): {args.K}")
    print(f"delta_c (sensitivity after normalisation): {deltas[0]:.4f} "
          f"(same for all channels)")

    # ------------------------------------------------------------------
    # Pre-compute sigma for each epsilon × mechanism
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Noise allocation per epsilon")
    print("=" * 70)

    sigma_table: Dict[float, Dict[str, torch.Tensor]] = {}

    for eps in args.epsilons:
        rho          = eps_to_rho(eps, args.delta_dp)
        sigma_wf     = waterfilling_sigma(deltas, importances, rho)
        sigma_uni    = uniform_sigma(deltas, rho)

        sigma_table[eps] = {
            "channel_WF": sigma_wf,
            "uniform":    sigma_uni,
            "rho":        rho,
        }

        top20 = torch.argsort(importances, descending=True)[:20]
        bot20 = torch.argsort(importances, descending=False)[:20]

        print(f"\n  eps={eps}  rho={rho:.5f}")
        print(f"    uniform  sigma: {sigma_uni[0]:.4f} (all channels)")
        print(f"    WF sigma range: [{sigma_wf.min():.4f}, {sigma_wf.max():.4f}]  "
              f"mean={sigma_wf.mean():.4f}")
        print(f"    WF top-20 important channels sigma mean: "
              f"{sigma_wf[top20].mean():.4f}  "
              f"← should be SMALLER than unimportant")
        print(f"    WF bot-20 unimportant channels sigma mean: "
              f"{sigma_wf[bot20].mean():.4f}")

        # Sanity check: important channels must get less noise
        if sigma_wf[top20].mean() < sigma_wf[bot20].mean():
            print(f"    ✅ Water-filling correct: important < unimportant")
        else:
            print(f"    ⚠️  Water-filling inverted — check importance scores")

    # ------------------------------------------------------------------
    # PASS 2: noisy forward passes
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Pass 2: noisy forward passes")
    print("=" * 70)

    mechanisms = ["no_noise", "uniform", "channel_WF"]
    results: Dict[str, Dict[float, List[float]]] = {
        m: {eps: [] for eps in args.epsilons} for m in mechanisms
    }

    # No-noise result is already known from pass 1
    for eps in args.epsilons:
        results["no_noise"][eps] = clean_dice_list.copy()

    with torch.no_grad():
        for batch_idx, raw_data in enumerate(loader):
            data   = move_to_device(model, raw_data, args.device)
            data   = pad_to_divisor(data, args.pad_divisor, args.ignore_index)
            inputs = data["inputs"]
            gt     = get_gt(data["data_samples"], args.device)

            # Get clean encoder outputs and bottleneck
            enc_outs, bottleneck = get_bottleneck(model, inputs)

            # Normalise bottleneck: clip to cap_c, then divide by cap_c
            # Result has per-channel L2 norm ≤ 1
            bn_normalised = clip_and_normalise(bottleneck, caps)
            # bn_normalised shape: (1, C, H_b, W_b)

            _, C_b, H_b, W_b = bottleneck.shape

            for eps in args.epsilons:
                for mech in ["uniform", "channel_WF"]:
                    sigma = sigma_table[eps][mech]   # (C,)

                    # Build noise in NORMALISED space
                    # sigma_c is for the normalised channel (L2 norm ≤ 1)
                    noise = (
                        torch.randn(1, C_b, H_b, W_b, device=args.device)
                        * sigma.view(1, C_b, 1, 1)
                    )

                    # Add noise in normalised space
                    bn_noisy_normalised = bn_normalised + noise

                    # Denormalise: multiply back by cap_c
                    # Now the noisy bottleneck is back in the original scale
                    bn_noisy = denormalise(bn_noisy_normalised, caps)

                    # Decode from noisy bottleneck
                    logits = decode_and_predict(model, enc_outs, bn_noisy)

                    results[mech][eps].append(
                        dice_score(logits, gt, model,
                                   cls=args.foreground_class,
                                   ignore_index=args.ignore_index)
                    )

            if (batch_idx + 1) % 5 == 0:
                print(f"  pass 2: {batch_idx + 1} images done")

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS — Mean Dice across 20 images")
    print("=" * 70)

    header = f"{'mechanism':<14s}" + \
             "".join(f"  eps={e:>5.1f}" for e in args.epsilons)
    print(header)
    print("-" * len(header))

    mean_dice: Dict[str, Dict[float, float]] = {}
    for mech in mechanisms:
        means = {eps: float(np.mean(results[mech][eps]))
                 for eps in args.epsilons}
        mean_dice[mech] = means
        row = f"{mech:<14s}" + \
              "".join(f"  {means[e]:>8.4f}" for e in args.epsilons)
        print(row)

    print()
    print("Lift of channel-WF over uniform (Dice points):")
    print(header)
    print("-" * len(header))

    lifts: Dict[float, float] = {}
    for eps in args.epsilons:
        lifts[eps] = round(
            mean_dice["channel_WF"][eps] - mean_dice["uniform"][eps], 6
        )
    lift_row = f"{'WF - uniform':<14s}" + \
               "".join(f"  {lifts[e]:>+8.4f}" for e in args.epsilons)
    print(lift_row)

    # ------------------------------------------------------------------
    # Save per-image CSV
    # ------------------------------------------------------------------
    per_image_csv = out_dir / "phase1_channel_noise_results_v2.csv"
    rows = []
    for mech in mechanisms:
        for eps in args.epsilons:
            for img_idx, d in enumerate(results[mech][eps]):
                rows.append({
                    "image_index": img_idx,
                    "mechanism":   mech,
                    "epsilon":     eps,
                    "dice":        round(d, 6),
                })

    with per_image_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_index", "mechanism", "epsilon", "dice"])
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------------------
    summary = {
        "version":          "v2_normalised",
        "config":           args.config,
        "checkpoint":       args.checkpoint,
        "importance_csv":   args.importance_csv,
        "n_images":         n_images,
        "n_channels":       int(C),
        "K_teachers":       args.K,
        "delta_dp":         args.delta_dp,
        "epsilons":         args.epsilons,
        "clean_dice":       round(clean_dice_avg, 6),
        "cap_stats": {
            "min":    round(float(caps.min()),    4),
            "max":    round(float(caps.max()),    4),
            "mean":   round(float(caps.mean()),   4),
            "median": round(float(caps.median()), 4),
        },
        "sensitivity_delta_c_after_normalisation": round(float(deltas[0]), 6),
        "mean_dice": {
            mech: {str(eps): round(mean_dice[mech][eps], 6)
                   for eps in args.epsilons}
            for mech in mechanisms
        },
        "lift_channel_WF_over_uniform_dice": {
            str(eps): lifts[eps] for eps in args.epsilons
        },
        "interpretation": (
            "lift > 0 at any epsilon means channel-WF preserved Dice better "
            "than uniform noise at the same privacy budget. "
            "This is the Phase 1 test from RESEARCH_PLAN.md Section 8."
        ),
        "outputs": {
            "per_image_csv": str(per_image_csv),
        },
    }

    summary_json = out_dir / "phase1_channel_noise_summary_v2.json"
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Done.")
    print(f"Clean Dice (no noise):  {clean_dice_avg:.6f}")
    print(f"Saved: {per_image_csv}")
    print(f"Saved: {summary_json}")

    key_eps = 2.0
    if key_eps in lifts:
        print(f"\nKey result at epsilon=2: "
              f"channel-WF lift over uniform = {lifts[key_eps]:+.4f} Dice points")
    print("=" * 70)


if __name__ == "__main__":
    main()