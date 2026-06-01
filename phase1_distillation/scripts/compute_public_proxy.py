#!/usr/bin/env python
"""
Compute per-channel caps and importance scores on PUBLIC retinal data.

These quantities are needed by phase1_gkd_distill_v2.py to allocate the
Gaussian noise on the (private) DRIVE bottleneck. Estimating them on
PUBLIC data (HRF / STARE / CHASE-DB1 — vessel-segmentation datasets
disjoint from DRIVE) means they consume ZERO privacy budget on DRIVE.

Outputs (CSV, one row per teacher bottleneck channel)
-----------------------------------------------------
    public_caps.csv         columns: cap_norm
    public_importance.csv   columns: gradient_abs_mean

Pipeline per public image
-------------------------
  1. Load image, resize / centre-crop to the teacher's expected input.
  2. Forward through teacher encoder, get bottleneck z (C, H, W).
  3. Per-channel L2 norm  -> collect for cap-quantile estimation.
  4. Pseudo-label = teacher.decode_head(z) argmax  (no public mask needed).
  5. Loss = CE(student-side prediction logits, pseudo-label)
     Gradient |dL / dz|_c  -> per-channel importance.

Why pseudo-labels
-----------------
HRF and CHASE-DB1 ship with expert vessel masks, STARE has hand-segmented
masks. We could use those. But using the TEACHER's own argmax as the
target keeps this script working on any unlabelled retinal image
collection, which makes it much easier to swap in additional proxy
datasets later. The cost is a slight loss of signal — argmax is a
biased target — which we accept because both caps and importance are
robust to label noise (caps are a 95-th percentile, importance is
averaged across all spatial locations and many images).

Run
---
    python3 compute_public_proxy.py \
        --teacher-config     ... \
        --teacher-checkpoint ... \
        --public-data-root   /path/to/HRF/images \
        --out-dir            /path/to/public_proxy_out \
        --image-size         584 565            # match teacher's training resolution
        --cap-quantile       0.95               # conservative (vs 0.9 on train)
        --device             cuda:0
"""

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

try:
    from mmengine.model import revert_sync_batchnorm
except Exception:
    revert_sync_batchnorm = None

from mmseg.registry import MODELS


# ---------------------------------------------------------------------------
# Teacher loading (mirrors phase1_gkd_distill_v2.py)
# ---------------------------------------------------------------------------

def build_teacher(cfg_path: str, ckpt_path: str, device: str):
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = MODELS.build(cfg.model)
    if revert_sync_batchnorm is not None:
        model = revert_sync_batchnorm(model)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.to(device).eval()
    return model


def unet_encoder(backbone, x: torch.Tensor) -> List[torch.Tensor]:
    outs = []
    for enc in backbone.encoder:
        x = enc(x)
        outs.append(x)
    return outs


def unet_decoder(backbone, enc_outs, bottleneck):
    x = bottleneck
    feats = [x]
    for i in reversed(range(len(backbone.decoder))):
        x = backbone.decoder[i](enc_outs[i], x)
        feats.append(x)
    return feats


# ---------------------------------------------------------------------------
# Public-image loading
# ---------------------------------------------------------------------------

PUBLIC_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".ppm", ".gif")


def list_public_images(root: str) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"public-data-root does not exist: {root}")
    paths = []
    for ext in PUBLIC_EXTS:
        paths.extend(root.rglob(f"*{ext}"))
        paths.extend(root.rglob(f"*{ext.upper()}"))
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError(f"no public images found under {root}")
    return paths


def load_image_tensor(
    path: Path,
    target_size: Tuple[int, int],
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
) -> torch.Tensor:
    """Load RGB image, resize, normalise with mmseg's default ImageNet stats."""
    img = Image.open(path).convert("RGB").resize(target_size[::-1], Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)              # (H, W, 3) 0-255
    arr = (arr - np.array(mean)) / np.array(std)
    t = torch.from_numpy(arr).permute(2, 0, 1).float()    # (3, H, W)
    return t


def pad_to_divisor(t: torch.Tensor, divisor: int = 16) -> torch.Tensor:
    _, h, w = t.shape
    ph = (divisor - h % divisor) % divisor
    pw = (divisor - w % divisor) % divisor
    if ph == 0 and pw == 0:
        return t
    return F.pad(t, (0, pw, 0, ph), value=0)


# ---------------------------------------------------------------------------
# Per-channel statistics on the public set
# ---------------------------------------------------------------------------

def compute_caps_and_importance(
    teacher,
    image_paths: List[Path],
    device: str,
    image_size: Tuple[int, int],
    cap_quantile: float,
    pad_divisor: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (caps[C], importance[C]) computed across all public images.

    caps[c]       = cap_quantile-th percentile of channel-c L2 norm.
    importance[c] = mean over images of mean_{i,j} |dL/dz_{c,i,j}|, with
                    L = CE(teacher decode-head logits, teacher argmax).
    """
    all_norms: List[torch.Tensor] = []
    grad_accum = None
    n_grad_imgs = 0

    for idx, path in enumerate(image_paths):
        t = load_image_tensor(path, image_size).unsqueeze(0).to(device)
        t = pad_to_divisor(t, pad_divisor)

        # ----- caps: forward + per-channel norm (no grad needed) -----
        with torch.no_grad():
            enc_outs = unet_encoder(teacher.backbone, t)
            z = enc_outs[-1]                                # (1, C, H, W)
            norms = z.squeeze(0).flatten(1).norm(dim=1)      # (C,)
            all_norms.append(norms.detach().cpu())

        # ----- importance: gradient of CE(logits, pseudo-label) wrt z -----
        # Re-forward with grad enabled on the bottleneck only.
        enc_outs = unet_encoder(teacher.backbone, t)
        z = enc_outs[-1].detach().clone().requires_grad_(True)
        feats = unet_decoder(teacher.backbone, enc_outs, z)
        logits = teacher.decode_head.forward(tuple(feats))
        if logits.shape[-2:] != t.shape[-2:]:
            logits = F.interpolate(logits, size=t.shape[-2:],
                                   mode="bilinear", align_corners=False)
        with torch.no_grad():
            pseudo_label = logits.argmax(dim=1)              # (1, H, W)
        loss = F.cross_entropy(logits, pseudo_label)
        grad = torch.autograd.grad(loss, z, retain_graph=False)[0]   # (1, C, H, W)
        per_ch = grad.abs().mean(dim=(0, 2, 3)).detach().cpu()        # (C,)
        if grad_accum is None:
            grad_accum = per_ch.clone()
        else:
            grad_accum += per_ch
        n_grad_imgs += 1

        if (idx + 1) % 5 == 0:
            print(f"  processed {idx + 1} / {len(image_paths)} images")

    norms_tensor = torch.stack(all_norms, dim=0)               # (N, C)
    caps = torch.quantile(norms_tensor, cap_quantile, dim=0)   # (C,)
    importance = grad_accum / max(n_grad_imgs, 1)
    importance = importance.clamp(min=1e-12)
    return caps, importance


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def write_caps_csv(path: Path, caps: torch.Tensor) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cap_norm"])
        for v in caps.tolist():
            w.writerow([f"{v:.6f}"])


def write_importance_csv(path: Path, importance: torch.Tensor) -> None:
    """Same column name expected by phase1_gkd_distill_v2.load_importances."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gradient_abs_mean"])
        for v in importance.tolist():
            w.writerow([f"{v:.6e}"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-config",     required=True)
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--public-data-root",   required=True,
                   help="Directory of public retinal images (HRF, STARE, CHASE-DB1).")
    p.add_argument("--out-dir",            required=True)
    p.add_argument("--image-size",         type=int, nargs=2, default=(584, 565),
                   help="Resize public images to (H, W) before forwarding.")
    p.add_argument("--cap-quantile",       type=float, default=0.95,
                   help="Conservative percentile (default 0.95) to absorb "
                        "distribution shift between public and private data.")
    p.add_argument("--pad-divisor",        type=int, default=16)
    p.add_argument("--device",             default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    print(f"Building teacher from {args.teacher_checkpoint}...")
    teacher = build_teacher(args.teacher_config,
                            args.teacher_checkpoint,
                            args.device)

    paths = list_public_images(args.public_data_root)
    print(f"Found {len(paths)} public images in {args.public_data_root}.")

    print("Computing per-channel caps + importance on public data...")
    caps, importance = compute_caps_and_importance(
        teacher=teacher,
        image_paths=paths,
        device=args.device,
        image_size=tuple(args.image_size),
        cap_quantile=args.cap_quantile,
        pad_divisor=args.pad_divisor,
    )

    caps_path = out_dir / "public_caps.csv"
    imp_path  = out_dir / "public_importance.csv"
    write_caps_csv(caps_path, caps)
    write_importance_csv(imp_path, importance)

    # Sanity print
    print(f"\nCaps        : min={caps.min():.3f}  max={caps.max():.3f}  "
          f"mean={caps.mean():.3f}  median={caps.median():.3f}")
    print(f"Importance  : min={importance.min():.3e}  max={importance.max():.3e}  "
          f"mean={importance.mean():.3e}")
    skew = importance.max() / importance.min()
    print(f"Importance max/min ratio = {skew:.1f}  "
          f"(higher = more skew = bigger expected WF advantage)")

    print(f"\nWrote:\n  {caps_path}\n  {imp_path}")
    print("\nNext step: in your sbatch script, pass:")
    print(f"  --threat-model public-proxy")
    print(f"  --public-caps-csv {caps_path}")
    print(f"  --importance-csv  {imp_path}")


if __name__ == "__main__":
    main()
