"""
EXP-3: Compute spatial importance saliency for spatial-WF.

PURPOSE
-------
Spatial-WF needs a per-pixel importance map s_{i,j} on the bottleneck
spatial grid (H_b × W_b). It must come from PUBLIC data so it costs
zero privacy on DRIVE.

We compute it on HRF retinal images via Frangi vesselness filter
(scikit-image), then downsample to the bottleneck spatial size that the
TinyUNet uses for 96×96 inputs.

OUTPUT
------
  spatial_saliency.pt    a torch tensor (H_b, W_b) with vessel-likeness
                         saliency averaged across HRF images
  drive_spatial_saliency_results.json
                         metadata + R_spatial = max(s) / min(s) for
                         the decision rule

DECISION RULE
-------------
  R_spatial > 20   -> proceed to EXP-4 (spatial-WF on student). Strong signal.
  R_spatial 5-20   -> spatial-WF may help marginally. Proceed but expect ±0.01.
  R_spatial < 5    -> spatial-WF unlikely to help. Skip EXP-4, focus on PATE.

Falls back to a synthetic radial-bias saliency if HRF is not available
(so the script always runs; the falls-back map should be replaced with
real HRF before paper submission).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet


def infer_bottleneck_spatial(input_size=96, base=32, device="cpu"):
    """Run a dummy forward to find out what (H_b, W_b) the bottleneck has."""
    t = TinyUNet(in_ch=3, num_classes=2, base=base).to(device).eval()
    x = torch.zeros(1, 3, input_size, input_size, device=device)
    with torch.no_grad():
        _, _, e3 = t.encode(x)
    return e3.shape[-2], e3.shape[-1]


def load_hrf_or_fallback(hrf_dir: Path, H_b: int, W_b: int):
    """Load HRF images. If absent, generate a synthetic radial-bias saliency."""
    images = []
    if hrf_dir.exists():
        from PIL import Image
        from skimage import filters, transform
        for p in sorted(hrf_dir.glob("*.jpg")) + sorted(hrf_dir.glob("*.png")):
            img = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            # downsample to a manageable size, then Frangi
            img_small = transform.resize(img, (256, 256), anti_aliasing=True)
            vesselness = filters.frangi(img_small, sigmas=range(1, 5))
            # Pool to bottleneck spatial size
            v_pool = transform.resize(vesselness, (H_b, W_b),
                                      anti_aliasing=True, mode="constant")
            images.append(v_pool)
        if images:
            return np.mean(images, axis=0), f"HRF ({len(images)} images, Frangi)"
    # Fallback: radial-Gaussian centered saliency.
    yy, xx = np.meshgrid(np.linspace(-1, 1, H_b), np.linspace(-1, 1, W_b),
                         indexing="ij")
    r2 = yy**2 + xx**2
    fallback = np.exp(-r2 * 4.0)
    return fallback, "fallback radial-bias (HRF directory missing)"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\n[1/3] Probing TinyUNet bottleneck spatial size for 96×96 input...")
    H_b, W_b = infer_bottleneck_spatial(input_size=96, base=32, device=device)
    print(f"  Bottleneck: ({H_b}, {W_b})  → {H_b * W_b} spatial positions")

    print("\n[2/3] Computing spatial saliency from public proxy (HRF)...")
    hrf_dir = Path.home() / "public_retinal" / "HRF"
    saliency, source = load_hrf_or_fallback(hrf_dir, H_b, W_b)
    saliency = np.clip(saliency, 1e-12, None)              # non-negative
    saliency_t = torch.tensor(saliency, dtype=torch.float32)

    R_spatial = float(saliency.max() / saliency.min())
    print(f"  Source: {source}")
    print(f"  Saliency stats: min={saliency.min():.4e}  max={saliency.max():.4e}")
    print(f"  R_spatial = {R_spatial:.2f}  (max/min ratio)")

    print("\n[3/3] Saving spatial saliency...")
    out_pt = Path(__file__).parent / "spatial_saliency.pt"
    torch.save({"saliency": saliency_t, "H_b": H_b, "W_b": W_b,
                "source": source, "R_spatial": R_spatial}, out_pt)

    out_json = Path(__file__).parent / "drive_spatial_saliency_results.json"
    decision = ("GO-strong" if R_spatial > 20 else
                "GO-marginal" if R_spatial > 5 else "NO-GO")
    with out_json.open("w") as f:
        json.dump({
            "H_b": H_b, "W_b": W_b,
            "source": source,
            "R_spatial": R_spatial,
            "saliency_min": float(saliency.min()),
            "saliency_max": float(saliency.max()),
            "saliency_mean": float(saliency.mean()),
            "decision_on_EXP4": decision,
        }, f, indent=2)

    print(f"  Wrote: {out_pt}")
    print(f"  Wrote: {out_json}")
    print(f"\nDECISION on EXP-4: {decision}  (R_spatial = {R_spatial:.1f})")


if __name__ == "__main__":
    main()
