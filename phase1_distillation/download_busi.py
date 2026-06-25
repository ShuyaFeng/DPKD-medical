"""
Build a BUSI (Breast Ultrasound Images) dataset in the same layout the
pipeline expects:
    data/BUSI_HF/
      train/input/<id>.png   train/label/<id>.png
      val/input/<id>.png     val/label/<id>.png
BUSI is grayscale (in_ch=1) -- unlike DRIVE/ISIC/Kvasir which are RGB.
We only use the benign+malignant images (647 total) since the 133 "normal"
images have no lesion to segment (common convention in the literature).
Some images have multiple lesions -> multiple mask files (e.g.
"benign (100)_mask.png" and "benign (100)_mask_1.png") -- these are merged
with a logical OR into a single binary mask.

Download the zip first via the Kaggle CLI:
  cd data/_busi_raw
  kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset

Usage:  python download_busi.py --n 0 --size 96
"""
import argparse, zipfile, random, re
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data"
RAW, OUT = ROOT / "_busi_raw", ROOT / "BUSI_HF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="subset size (0 = all 647)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    zpath = RAW / "breast-ultrasound-images-dataset.zip"
    if not zpath.exists():
        raise FileNotFoundError(
            f"{zpath} not found. Download it first via:\n"
            f"  cd {RAW}\n"
            f"  kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset"
        )

    zf = zipfile.ZipFile(zpath)
    names = zf.namelist()

    # Folder layout: Dataset_BUSI_with_GT/benign/benign (N).png + benign (N)_mask.png [+ _mask_1.png ...]
    #                Dataset_BUSI_with_GT/malignant/...   (skip "normal" -- no lesion to segment)
    img_re = re.compile(r"/(benign|malignant)/\1 \((\d+)\)\.png$", re.IGNORECASE)
    mask_re = re.compile(r"/(benign|malignant)/\1 \((\d+)\)_mask(_\d+)?\.png$", re.IGNORECASE)

    img_by_id = {}
    masks_by_id = {}
    for n in names:
        m = img_re.search(n)
        if m:
            cls, num = m.group(1).lower(), m.group(2)
            img_by_id[f"{cls}_{num}"] = n
            continue
        m = mask_re.search(n)
        if m:
            cls, num = m.group(1).lower(), m.group(2)
            masks_by_id.setdefault(f"{cls}_{num}", []).append(n)

    ids = sorted(set(img_by_id) & set(masks_by_id))
    rng = random.Random(args.seed); rng.shuffle(ids)
    if args.n > 0:
        ids = ids[:args.n]

    nval = int(len(ids) * args.val_frac)
    val_ids = set(ids[:nval])
    print(f"building BUSI_HF: {len(ids)} images (benign+malignant only)  "
          f"(val={len(val_ids)}, train={len(ids)-len(val_ids)})  size={args.size}")

    for sp in ("train", "val"):
        for s in ("input", "label"):
            (OUT / sp / s).mkdir(parents=True, exist_ok=True)

    sz = args.size
    done = 0
    for iid in ids:
        sp = "val" if iid in val_ids else "train"

        img = Image.open(zf.open(img_by_id[iid])).convert("L").resize((sz, sz), Image.BILINEAR)
        img.save(OUT / sp / "input" / f"{iid}.png")

        merged = None
        for mname in masks_by_id[iid]:
            m = np.array(Image.open(zf.open(mname)).convert("L").resize((sz, sz), Image.NEAREST))
            m = (m > 127)
            merged = m if merged is None else (merged | m)
        Image.fromarray((merged.astype(np.uint8) * 255)).save(OUT / sp / "label" / f"{iid}.png")

        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(ids)}")

    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
