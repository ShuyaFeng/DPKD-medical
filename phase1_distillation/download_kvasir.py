"""
Build a Kvasir-SEG (polyp segmentation) dataset in the same layout the
pipeline expects:
    data/KVASIR_HF/
      train/input/<id>.png   train/label/<id>.png
      val/input/<id>.png     val/label/<id>.png
Kvasir-SEG is only ~46MB (1000 images total), so we download the whole zip
directly -- no need for remotezip range requests like ISIC's 10.6GB archive.
Usage:  python download_kvasir.py --n 0 --size 96
"""
import argparse, io, zipfile, random
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data"
RAW, OUT = ROOT / "_kvasir_raw", ROOT / "KVASIR_HF"
ZIP_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="subset size (0 = all 1000)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "kvasir-seg.zip"
    if not zpath.exists():
        import urllib.request
        print("downloading Kvasir-SEG (46 MB)...")
        urllib.request.urlretrieve(ZIP_URL, zpath)

    zf = zipfile.ZipFile(zpath)
    names = zf.namelist()

    # images/<id>.jpg  and  masks/<id>.jpg  (matched by filename stem)
    img_by_id = {Path(n).stem: n for n in names
                 if "/images/" in n.lower() and n.lower().endswith((".jpg", ".jpeg", ".png"))}
    mask_by_id = {Path(n).stem: n for n in names
                  if "/masks/" in n.lower() and n.lower().endswith((".jpg", ".jpeg", ".png"))}

    ids = sorted(set(img_by_id) & set(mask_by_id))
    rng = random.Random(args.seed); rng.shuffle(ids)
    if args.n > 0:
        ids = ids[:args.n]

    nval = int(len(ids) * args.val_frac)
    val_ids = set(ids[:nval])
    print(f"building KVASIR_HF: {len(ids)} images  (val={len(val_ids)}, train={len(ids)-len(val_ids)})  size={args.size}")

    for sp in ("train", "val"):
        for s in ("input", "label"):
            (OUT / sp / s).mkdir(parents=True, exist_ok=True)

    sz = args.size
    done = 0
    for iid in ids:
        sp = "val" if iid in val_ids else "train"
        img = Image.open(io.BytesIO(zf.read(img_by_id[iid]))).convert("RGB").resize((sz, sz), Image.BILINEAR)
        img.save(OUT / sp / "input" / f"{iid}.png")

        m = Image.open(io.BytesIO(zf.read(mask_by_id[iid]))).convert("L").resize((sz, sz), Image.NEAREST)
        Image.fromarray(((np.array(m) > 127).astype(np.uint8) * 255)).save(OUT / sp / "label" / f"{iid}.png")

        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(ids)}")

    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()