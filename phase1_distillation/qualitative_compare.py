"""
Qualitative comparison figure: Input | Ground Truth | Uniform pred | CANAL
pred, for a SPECIFIC, chosen list of filenames (strong win / typical win /
honest failure), using the single-best-seed checkpoints (per Dr. Tian:
"single-best is fine").

Usage:
  python qualitative_compare.py --dataset busi --epsilon 2.0 \
      --uniform-seed 500 --canal-seed 200 \
      --filenames benign_286 malignant_114 benign_138
"""
import argparse, sys, glob
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet, vessel_dice

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["busi", "kvasir", "isic"])
ap.add_argument("--epsilon", required=True, type=str)
ap.add_argument("--uniform-seed", required=True, type=int)
ap.add_argument("--canal-seed", required=True, type=int)
ap.add_argument("--filenames", required=True, nargs="+",
                help="exact filenames (no extension) to display, e.g. benign_286 malignant_114 benign_138")
ap.add_argument("--suffix", default="", help="checkpoint filename suffix, e.g. rerun3 for ISIC")
ap.add_argument("--size", type=int, default=96)
ap.add_argument("--upscale", type=int, default=4, help="display upscale factor for clarity")
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
assert len(all_filenames) == len(val_ds), f"{len(all_filenames)} filenames vs {len(val_ds)} dataset items -- mismatch!"
fname_to_idx = {f: i for i, f in enumerate(all_filenames)}

for fname in args.filenames:
    if fname not in fname_to_idx:
        raise ValueError(f"filename '{fname}' not found in {args.dataset} val set")

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

sz = args.size
up = args.upscale
disp_sz = sz * up
header_h = 24
row_label_w = 110
dice_label_h = 16
n_cols = 4
headers = ["Input", "Ground Truth", "Uniform", "CANAL"]

n_rows = len(args.filenames)
canvas_w = row_label_w + disp_sz * n_cols
canvas_h = header_h + (disp_sz + dice_label_h) * n_rows
canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)

for col, h in enumerate(headers):
    x0 = row_label_w + col * disp_sz
    draw.text((x0 + disp_sz // 2 - 25, 4), h, fill="black")

y = header_h
for fname in args.filenames:
    idx = fname_to_idx[fname]
    x, y_gt = val_ds[idx]
    uni_pred, uni_dice = predict_and_dice(uniform_model, x, y_gt)
    can_pred, can_dice = predict_and_dice(canal_model, x, y_gt)

    input_img = Image.fromarray(to_display_rgb(x)).resize((disp_sz, disp_sz), Image.Resampling.NEAREST)
    gt_img = Image.fromarray(mask_to_rgb(y_gt)).resize((disp_sz, disp_sz), Image.Resampling.NEAREST)
    uni_img = Image.fromarray(mask_to_rgb(uni_pred)).resize((disp_sz, disp_sz), Image.Resampling.NEAREST)
    can_img = Image.fromarray(mask_to_rgb(can_pred)).resize((disp_sz, disp_sz), Image.Resampling.NEAREST)

    draw.text((4, y + disp_sz // 2), fname, fill="black")
    canvas.paste(input_img, (row_label_w, y))
    canvas.paste(gt_img, (row_label_w + disp_sz, y))
    canvas.paste(uni_img, (row_label_w + disp_sz * 2, y))
    canvas.paste(can_img, (row_label_w + disp_sz * 3, y))

    draw.text((row_label_w + disp_sz * 2 + disp_sz // 2 - 30, y + disp_sz + 1),
               f"Dice={uni_dice:.3f}", fill="black")
    draw.text((row_label_w + disp_sz * 3 + disp_sz // 2 - 30, y + disp_sz + 1),
               f"Dice={can_dice:.3f}", fill="black")

    y += disp_sz + dice_label_h

out_path = HERE / f"qualitative_{args.dataset}_eps{args.epsilon}.png"
canvas.save(out_path)
print(f"\nSaved: {out_path}")
