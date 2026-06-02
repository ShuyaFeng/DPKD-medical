"""
Synthetic demo: a minimal self-contained reproduction of phase1_distillation
that runs on CPU/MPS without DRIVE, mmseg, or a HPC cluster.

What it shows (the same ideas as the real scripts):
  1. Train a small "teacher" U-Net on a synthetic 2-class segmentation task
     (random circles/squares on noisy backgrounds).
  2. Compute gradient-based channel importance on the teacher bottleneck
     (GKD Eq. 1, == analyze_unet_drive_bottleneck_channels_gradient.py).
  3. Convert (eps, delta)-DP to rho-zCDP and compute noise sigma for two
     mechanisms: uniform and channel-WF
     (== phase1_drive_channel_noise_v2.py).
  4. "No-training" probe: inject noise into the clean teacher bottleneck,
     observe IoU drop under both mechanisms across a sweep of epsilons
     (== phase1_drive_channel_noise_v2.py Pass 2).
  5. Distil a smaller student U-Net using direct feature matching with a
     1x1 adapter, just like phase1_gkd_distill_v2.py.

The teacher / student / data are TINY so it runs in minutes on CPU/MPS.
Do NOT expect numbers to match the real DRIVE experiment — the point is
to demonstrate that the math + pipeline behave the way the paper claims.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 1. Synthetic segmentation dataset
# =============================================================================

class SyntheticShapes(Dataset):
    """
    Each image is a (1, 64, 64) noisy grayscale image with 1-3 random shapes
    (circles or squares) drawn at random positions. The label is a (64, 64)
    binary mask: 1 inside any shape, 0 elsewhere.

    This is the analog of DRIVE: a binary foreground/background segmentation
    task, just much smaller and procedurally generated so it needs no files.
    """

    def __init__(self, n_samples: int = 256, img_size: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.images = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
        self.masks  = np.zeros((n_samples,    img_size, img_size), dtype=np.int64)

        for i in range(n_samples):
            # background = mild noise
            img = rng.normal(0.0, 0.15, size=(img_size, img_size)).astype(np.float32)
            msk = np.zeros((img_size, img_size), dtype=np.int64)

            n_shapes = rng.integers(1, 4)
            for _ in range(n_shapes):
                cx = rng.integers(8, img_size - 8)
                cy = rng.integers(8, img_size - 8)
                r  = rng.integers(4, 10)

                if rng.random() < 0.5:
                    # circle
                    yy, xx = np.ogrid[:img_size, :img_size]
                    mask_shape = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
                else:
                    # square
                    mask_shape = np.zeros_like(msk, dtype=bool)
                    mask_shape[max(0, cy-r):cy+r, max(0, cx-r):cx+r] = True

                img[mask_shape] += rng.uniform(0.5, 1.0)
                msk[mask_shape]  = 1

            self.images[i, 0] = img
            self.masks[i]     = msk

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]), torch.from_numpy(self.masks[idx])


# =============================================================================
# 2. Tiny U-Net (mimics the structure used by the real teacher/student)
# =============================================================================

def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """
    A 3-stage U-Net. `base` controls the channel multiplier — base=16 gives
    a "small" student (bottleneck channels = 16 * 4 = 64) and base=32 gives
    a "big" teacher (bottleneck = 32 * 4 = 128). This is the same shape
    pattern as the real DRIVE experiment (teacher 1024 ch / student 512 ch).
    """

    def __init__(self, in_ch: int = 1, num_classes: int = 2, base: int = 16):
        super().__init__()
        self.base = base
        self.enc1 = conv_block(in_ch,   base)        # 64x64 ->  base
        self.enc2 = conv_block(base,    base * 2)    # 32x32 -> 2*base
        self.enc3 = conv_block(base * 2, base * 4)   # 16x16 -> 4*base  (BOTTLENECK)
        self.pool = nn.MaxPool2d(2)
        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1  = nn.ConvTranspose2d(base * 2, base,     2, 2)
        self.dec1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))   # the bottleneck
        return e1, e2, e3

    def decode(self, e1, e2, bottleneck):
        d2 = self.up2(bottleneck)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)

    def forward(self, x):
        e1, e2, e3 = self.encode(x)
        return self.decode(e1, e2, e3)


# =============================================================================
# 3. Losses + metrics  (same shape as the real scripts)
# =============================================================================

def dice_loss(logits: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, 1]
    gtf  = (gt == 1).float()
    inter = (prob * gtf).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + gtf.sum(dim=(1, 2))
    return (1.0 - (2 * inter + eps) / (denom + eps)).mean()


def task_loss(logits, gt):
    return F.cross_entropy(logits, gt) + 3.0 * dice_loss(logits, gt)


def iou_score(logits, gt, eps: float = 1e-7) -> float:
    pred = logits.argmax(dim=1)
    pv = (pred == 1)
    gv = (gt   == 1)
    inter = (pv & gv).sum(dim=(1, 2)).float()
    union = (pv | gv).sum(dim=(1, 2)).float()
    return float(((inter + eps) / (union + eps)).mean().item())


# =============================================================================
# 4. DP machinery  (verbatim from phase1_drive_channel_noise_v2.py)
# =============================================================================

def eps_to_rho(eps: float, delta: float = 1e-5) -> float:
    log_inv_d = math.log(1.0 / delta)
    b = 2.0 * math.sqrt(log_inv_d)
    discriminant = b * b + 4.0 * eps
    return ((-b + math.sqrt(discriminant)) / 2.0) ** 2


def waterfilling_sigma(deltas, importances, rdp_budget, eps_s: float = 1e-12):
    s = importances.clamp(min=eps_s)
    kappa = ((deltas * s.sqrt()).sum() / rdp_budget).sqrt()
    return kappa * deltas.sqrt() / s.pow(0.25)


def uniform_sigma(deltas, rdp_budget):
    sigma = (deltas.pow(2).sum() / rdp_budget).sqrt()
    return sigma.expand(deltas.shape[0]).clone()


def clip_and_normalise(bn, caps):
    B, C, H, W = bn.shape
    norms = bn.flatten(2).norm(dim=2)
    scale = (caps.view(1, C) / norms.clamp(min=1e-12)).clamp(max=1.0)
    clipped = bn * scale.view(B, C, 1, 1)
    return clipped / caps.view(1, C, 1, 1).clamp(min=1e-12)


def denormalise(bn, caps):
    _, C, _, _ = bn.shape
    return bn * caps.view(1, C, 1, 1)


# =============================================================================
# 5. Training utilities
# =============================================================================

def train_model(model, loader, n_epochs: int, lr: float, device: str,
                tag: str = "model"):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = task_loss(logits, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  [{tag}] epoch {epoch+1}/{n_epochs}  loss={total/len(loader):.4f}")


@torch.no_grad()
def evaluate_iou(model, loader, device):
    model.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        scores.append(iou_score(model(x), y))
    return float(np.mean(scores))


# =============================================================================
# 6. Channel importance via gradient  (analyze_*.py reproduced)
# =============================================================================

def compute_channel_importance(model, loader, device):
    """
    For each batch, do a forward pass that goes encoder -> bottleneck ->
    decoder -> logits -> task_loss, then take dL/dA at the bottleneck and
    average |dL/dA| over (batch, H, W) to get a (C,) importance vector.
    """
    model.eval()
    bn_channels = model.base * 4
    grad_abs_sum = torch.zeros(bn_channels, device=device)
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)

        e1, e2, e3 = model.encode(x)
        e3.retain_grad()
        logits = model.decode(e1, e2, e3)
        loss = task_loss(logits, y)
        loss.backward()

        grad = e3.grad.detach()                                  # (B, C, H, W)
        grad_abs_sum += grad.abs().mean(dim=(0, 2, 3))           # (C,)
        n += 1

    return (grad_abs_sum / n).cpu()


# =============================================================================
# 7. Noise-only probe (no training)  (phase1_drive_channel_noise_v2.py Pass 2)
# =============================================================================

def collect_caps(model, loader, device, q: float = 0.9):
    """90-percentile of per-channel L2 norms across all images."""
    model.eval()
    all_norms = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, _, e3 = model.encode(x)
            for b in range(e3.shape[0]):
                all_norms.append(e3[b].flatten(1).norm(dim=1).cpu())
    return torch.quantile(torch.stack(all_norms), q, dim=0)


def probe_noise_only(model, loader, importances, caps, device,
                    epsilons, K: int = 10, delta_dp: float = 1e-5):
    """
    For each epsilon, run uniform and channel_WF noise injection on the
    clean teacher bottleneck (no student involved). Returns IoU per (mech, eps).
    """
    bn_channels = importances.shape[0]
    deltas = torch.full((bn_channels,), 2.0 / K, device=device)

    sigma_table: Dict[float, Dict[str, torch.Tensor]] = {}
    for eps in epsilons:
        rho = eps_to_rho(eps, delta_dp)
        sigma_table[eps] = {
            "uniform":     uniform_sigma(deltas, rho),
            "channel_WF":  waterfilling_sigma(deltas, importances.to(device), rho),
            "rho":         rho,
        }

    results = {"no_noise": [], "uniform": {e: [] for e in epsilons},
               "channel_WF": {e: [] for e in epsilons}}

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            e1, e2, e3 = model.encode(x)
            # baseline (no noise)
            results["no_noise"].append(iou_score(model.decode(e1, e2, e3), y))

            bn_norm = clip_and_normalise(e3, caps.to(device))
            B, C, H, W = bn_norm.shape
            for eps in epsilons:
                for mech in ["uniform", "channel_WF"]:
                    sigma = sigma_table[eps][mech]
                    noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                    bn_noisy = denormalise(bn_norm + noise, caps.to(device))
                    logits = model.decode(e1, e2, bn_noisy)
                    results[mech][eps].append(iou_score(logits, y))

    mean_results = {
        "no_noise": float(np.mean(results["no_noise"])),
        "uniform":     {e: float(np.mean(results["uniform"][e]))     for e in epsilons},
        "channel_WF":  {e: float(np.mean(results["channel_WF"][e]))  for e in epsilons},
    }
    return mean_results, sigma_table


# =============================================================================
# 8. Student distillation  (phase1_gkd_distill_v2.py reproduced)
# =============================================================================

class Adapter(nn.Module):
    """1x1 conv: student_ch -> teacher_ch."""
    def __init__(self, s_ch: int, t_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(s_ch, t_ch, 1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.conv(x)


def distill_student(teacher, student_base: int, train_loader, val_loader,
                    importances, caps, device,
                    epsilon: float, noise_type: str, *,
                    K: int = 10, delta_dp: float = 1e-5,
                    lambda_feat: float = 0.4, n_epochs: int = 5,
                    seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    student = TinyUNet(base=student_base).to(device)
    s_ch = student.base * 4
    t_ch = teacher.base * 4
    adapter = Adapter(s_ch, t_ch).to(device)

    deltas = torch.full((t_ch,), 2.0 / K, device=device)
    rho = eps_to_rho(epsilon, delta_dp)
    if noise_type == "channel_WF":
        sigma = waterfilling_sigma(deltas, importances.to(device), rho)
    else:
        sigma = uniform_sigma(deltas, rho)

    opt = torch.optim.Adam(list(student.parameters()) + list(adapter.parameters()),
                           lr=1e-3)
    teacher.eval()

    for epoch in range(n_epochs):
        student.train(); adapter.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                _, _, t_e3 = teacher.encode(x)
                t_bn_norm = clip_and_normalise(t_e3, caps.to(device))
                B, C, H, W = t_bn_norm.shape
                noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                t_bn_noisy = denormalise(t_bn_norm + noise, caps.to(device))

            opt.zero_grad()
            s_e1, s_e2, s_e3 = student.encode(x)
            s_proj = adapter(s_e3)
            f_loss = F.mse_loss(s_proj, t_bn_noisy.detach())
            s_logits = student.decode(s_e1, s_e2, s_e3)
            t_loss = task_loss(s_logits, y)
            loss = t_loss + lambda_feat * f_loss
            loss.backward()
            opt.step()
            total += loss.item()
        val_iou = evaluate_iou(student, val_loader, device)
        print(f"    epoch {epoch+1}/{n_epochs} loss={total/len(train_loader):.4f}  "
              f"val_IoU={val_iou:.4f}")

    return evaluate_iou(student, val_loader, device)


# =============================================================================
# 9. Main entry point
# =============================================================================

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(0)
    np.random.seed(0)

    # --- data ---
    print("\n[1/5] Building synthetic dataset...")
    train_ds = SyntheticShapes(n_samples=256, seed=0)
    val_ds   = SyntheticShapes(n_samples=64,  seed=1)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  image_shape=(1,64,64)")

    # --- teacher (the "big" model) ---
    print("\n[2/5] Training teacher (base=32, bottleneck=128 ch)...")
    teacher = TinyUNet(base=32).to(device)
    t0 = time.time()
    train_model(teacher, train_loader, n_epochs=6, lr=1e-3, device=device, tag="teacher")
    print(f"  teacher time: {time.time()-t0:.1f}s")
    teacher_iou = evaluate_iou(teacher, val_loader, device)
    print(f"  teacher val IoU: {teacher_iou:.4f}")

    # --- student baseline (the "small" model, no distillation) ---
    print("\n[3/5] Training student baseline (base=16, bottleneck=64 ch)...")
    student_baseline = TinyUNet(base=16).to(device)
    train_model(student_baseline, train_loader, n_epochs=6, lr=1e-3,
                device=device, tag="student_baseline")
    baseline_iou = evaluate_iou(student_baseline, val_loader, device)
    print(f"  student baseline val IoU: {baseline_iou:.4f}")

    # --- channel importance + cap collection ---
    print("\n[4/5] Computing channel importance + per-channel caps on teacher...")
    importances = compute_channel_importance(teacher, train_loader, device)
    caps        = collect_caps(teacher, train_loader, device)
    print(f"  importance C={importances.shape[0]}  "
          f"min={importances.min():.3e} max={importances.max():.3e}")
    print(f"  caps min={caps.min():.3f} max={caps.max():.3f} mean={caps.mean():.3f}")

    top5    = torch.argsort(importances, descending=True)[:5].tolist()
    bottom5 = torch.argsort(importances, descending=False)[:5].tolist()
    print(f"  Top-5 important channels:   {top5}")
    print(f"  Bottom-5 unimportant ch.:   {bottom5}")

    # --- noise-only probe (no student) ---
    print("\n[5/5] Phase A — noise-only probe (no training, no student)...")
    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    probe_results, sigma_table = probe_noise_only(
        teacher, val_loader, importances, caps, device, epsilons,
    )
    print(f"  clean (no noise) IoU = {probe_results['no_noise']:.4f}")
    print(f"  {'eps':>5}  {'uniform':>10}  {'WF':>10}  {'WF-uni':>10}")
    print("  " + "-" * 42)
    probe_lift: Dict[float, float] = {}
    for eps in epsilons:
        u = probe_results["uniform"][eps]
        w = probe_results["channel_WF"][eps]
        probe_lift[eps] = w - u
        print(f"  {eps:>5.1f}  {u:>10.4f}  {w:>10.4f}  {w-u:>+10.4f}")

    # --- student distillation across mechanisms and epsilons ---
    print("\n[Distillation] Phase B — distil student with feature matching...")
    distill_results = {}
    for noise_type in ["uniform", "channel_WF"]:
        for eps in [1.0, 2.0, 8.0]:        # subset for runtime
            key = f"{noise_type}_eps{eps}"
            print(f"\n  >>> {key}")
            iou = distill_student(
                teacher, student_base=16,
                train_loader=train_loader, val_loader=val_loader,
                importances=importances, caps=caps, device=device,
                epsilon=eps, noise_type=noise_type,
                n_epochs=4, seed=0,
            )
            distill_results[key] = iou
            print(f"    final val IoU = {iou:.4f}  "
                  f"(baseline {baseline_iou:.4f}, lift {iou-baseline_iou:+.4f})")

    # --- save all results to JSON ---
    out_path = Path(__file__).parent / "synthetic_demo_results.json"
    out_path.write_text(json.dumps({
        "device": device,
        "teacher_iou": teacher_iou,
        "student_baseline_iou": baseline_iou,
        "importance_top5_channels": top5,
        "importance_bottom5_channels": bottom5,
        "caps_stats": {
            "min": float(caps.min()),
            "max": float(caps.max()),
            "mean": float(caps.mean()),
        },
        "noise_probe": {
            "no_noise":   probe_results["no_noise"],
            "uniform":    {str(e): probe_results["uniform"][e]    for e in epsilons},
            "channel_WF": {str(e): probe_results["channel_WF"][e] for e in epsilons},
            "lift_WF_over_uniform": {str(e): probe_lift[e] for e in epsilons},
        },
        "distillation": distill_results,
    }, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
