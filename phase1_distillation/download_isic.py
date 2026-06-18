"""
Build an ISIC 2018 Task-1 (skin-lesion segmentation) dataset in the same
layout the pipeline expects:

    data/ISIC_HF/
      train/input/<id>.png   train/label/<id>.png
      val/input/<id>.png     val/label/<id>.png

Masks come from the 26 MB official GT zip. Images are pulled from the 10.6 GB
official input zip WITHOUT downloading it whole: remotezip uses HTTP range
requests to fetch only the N needed entries. Default builds a SUBSET (local
validation); --n 0 grabs all 2594 (for a full/cluster build).

Usage:  python download_isic.py --n 250 --size 96
"""
import argparse, io, zipfile, random
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data"
RAW, OUT = ROOT / "_isic_raw", ROOT / "ISIC_HF"
MASK_URL = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip"
IMG_URL  = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="subset size (0 = all 2594)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    mzip = RAW / "masks.zip"
    if not mzip.exists():
        import urllib.request
        print("downloading GT masks (26 MB)...")
        urllib.request.urlretrieve(MASK_URL, mzip)
    mz = zipfile.ZipFile(mzip)
    mask_by_id = {Path(n).name.replace("_segmentation.png", ""): n
                  for n in mz.namelist() if n.lower().endswith("_segmentation.png")}
    ids = sorted(mask_by_id)
    rng = random.Random(args.seed); rng.shuffle(ids)
    if args.n > 0:
        ids = ids[:args.n]
    nval = int(len(ids) * args.val_frac)
    val_ids = set(ids[:nval])
    print(f"building ISIC_HF: {len(ids)} images  (val={len(val_ids)}, train={len(ids)-len(val_ids)})  size={args.size}")

    for sp in ("train", "val"):
        for s in ("input", "label"):
            (OUT / sp / s).mkdir(parents=True, exist_ok=True)

    sz = args.size
    from remotezip import RemoteZip
    with RemoteZip(IMG_URL) as rz:
        img_by_id = {Path(n).name.replace(".jpg", ""): n
                     for n in rz.namelist() if n.lower().endswith(".jpg")}
        done = 0
        for iid in ids:
            if iid not in img_by_id:
                continue
            sp = "val" if iid in val_ids else "train"
            img = Image.open(io.BytesIO(rz.read(img_by_id[iid]))).convert("RGB").resize((sz, sz), Image.BILINEAR)
            img.save(OUT / sp / "input" / f"{iid}.png")
            m = Image.open(io.BytesIO(mz.read(mask_by_id[iid]))).convert("L").resize((sz, sz), Image.NEAREST)
            Image.fromarray(((np.array(m) > 127).astype(np.uint8) * 255)).save(OUT / sp / "label" / f"{iid}.png")
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(ids)}")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
