"""
Export ONE validation image's uniform prediction and CANAL prediction as
separate PNG files (plus the input and ground truth), for direct use in a
paper figure. Uses the single-best-seed checkpoints (per Dr. Tian).

Usage:
  python export_single_comparison.py --dataset busi --epsilon 2.0 \
      --uniform-seed 500 --canal-seed 200 --filename benign_286
"""
import argparse, sys, glob
from pathlib import Path
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet, vessel_dice

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["busi", "kvasir", "isic"])
ap.add_argument("--epsilon", required=True, type=str)
ap.add_argument("--uniform-seed", required=True, type=int)
ap.add_argument("--canal-seed", required=True, type=int)
ap.add_argument("--filename", required=True, help="exact filename (no extension), e.g. benign_286")
ap.add_argument("--suffix", default="", help="checkpoint filename suffix, e.g. rerun3 for ISIC")
ap.add_argument("--size", type=int, default=96)
ap.add_argument("--upscale", type=int, default=6, help="upscale factor for a crisper exported image")
args = ap.parse_args()

HERE = Path(__file__).parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_FOLDER = {"busi": "BUSI_HF", "kvasir": "KVASIR_HF", "isic": "ISIC_HF"}[args.dataset]

if args.dataset == "busi":
    from busi_dataset import BUSIDataset
    val_ds = BUSIDataset("val", args.size)
elif args.dataset == "kvasir":
    from kvasir_dataset import KvasirDataset
    val_ds = KvasirDataset("val", args.size)
else:
    from isic_dataset import ISICDataset
    val_ds = ISICDataset("val", args.size)

in_dir = HERE.parent / "data" / DATASET_FOLDER / "val" / "input"
all_filenames = [Path(p).stem for p in sorted(glob.glob(str(in_dir / "*.png")))]
assert len(all_filenames) == len(val_ds), "filename/dataset length mismatch"
if args.filename not in all_filenames:
    raise ValueError(f"'{args.filename}' not found in {args.dataset} val set")
idx = all_filenames.index(args.filename)

def load_model(seed, canal_bool):
    sfx = f"_{args.suffix}" if args.suffix else ""
    ckpt_path = HERE / "checkpoints" / f"{args.dataset}_K3_canal{canal_bool}_eps{args.epsilon}_seed{seed}{sfx}.pt"
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = TinyUNet(in_ch=ckpt["in_ch"], num_classes=2, base=ckpt["student_base"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded {ckpt_path.name}  (best_dice={ckpt['best_dice']:.4f}, best_epoch={ckpt['best_epoch']})")
    return model

uniform_model = load_model(args.uniform_seed, "False")
canal_model = load_model(args.canal_seed, "True")

def to_display_rgb(x_tensor):
    arr = x_tensor.cpu().numpy()
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = (arr.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return arr

def mask_to_rgb(mask_tensor):
    arr = mask_tensor.cpu().numpy()
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[arr == 1] = (255, 255, 255)
    return rgb

@torch.no_grad()
def predict_and_dice(model, x, y):
    logits = model(x.unsqueeze(0).to(dev))
    pred = logits.argmax(dim=1).squeeze(0)
    dice = vessel_dice(logits, y.unsqueeze(0).to(dev))
    return pred, dice

x, y_gt = val_ds[idx]
uni_pred, uni_dice = predict_and_dice(uniform_model, x, y_gt)
can_pred, can_dice = predict_and_dice(canal_model, x, y_gt)

sz, up = args.size, args.upscale
disp_sz = sz * up

out_dir = HERE / "qualitative_exports"
out_dir.mkdir(exist_ok=True)
prefix = f"{args.dataset}_{args.filename}_eps{args.epsilon}"

Image.fromarray(to_display_rgb(x)).resize((disp_sz, disp_sz), Image.Resampling.NEAREST).save(out_dir / f"{prefix}_input.png")
Image.fromarray(mask_to_rgb(y_gt)).resize((disp_sz, disp_sz), Image.Resampling.NEAREST).save(out_dir / f"{prefix}_groundtruth.png")
Image.fromarray(mask_to_rgb(uni_pred)).resize((disp_sz, disp_sz), Image.Resampling.NEAREST).save(out_dir / f"{prefix}_uniform_dice{uni_dice:.3f}.png")
Image.fromarray(mask_to_rgb(can_pred)).resize((disp_sz, disp_sz), Image.Resampling.NEAREST).save(out_dir / f"{prefix}_canal_dice{can_dice:.3f}.png")

print(f"\nSaved 4 images to {out_dir}/:")
print(f"  {prefix}_input.png")
print(f"  {prefix}_groundtruth.png")
print(f"  {prefix}_uniform_dice{uni_dice:.3f}.png   (Dice={uni_dice:.4f})")
print(f"  {prefix}_canal_dice{can_dice:.3f}.png     (Dice={can_dice:.4f})")
