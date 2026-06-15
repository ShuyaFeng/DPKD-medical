"""
Download + reorganize the DRIVE dataset into the layout the drive_*.py
scripts expect:

    <repo>/data/DRIVE_HF/
      train/input/<n>.tif      train/label/<n>.png
      val/input/<n>.tif        val/label/<n>.png

Source: Hugging Face mirror
  Zomba/DRIVE-digital-retinal-images-for-vessel-extraction
(inspect its license before redistributing).

The script is robust to the exact internal layout of the HF repo:
it walks everything, classifies each file as image vs mask by extension
+ path keywords, assigns train/val by 'training'/'test' in the path
(falling back to DRIVE's numbering: 21-40 train, 1-20 test), matches
image↔mask by the leading number, and writes a clean DRIVE_HF tree.

Usage
-----
# 1) download from HF then reorganize:
python download_drive.py

# 2) if you already have DRIVE unpacked somewhere, just reorganize:
python download_drive.py --src-dir /path/to/unpacked/DRIVE

# 3) custom HF repo:
python download_drive.py --repo OTHER/repo-id
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "DRIVE_HF"
DEFAULT_REPO = "Zomba/DRIVE-digital-retinal-images-for-vessel-extraction"

IMG_EXT = {".tif", ".tiff", ".ppm", ".jpg", ".jpeg", ".png"}
MASK_EXT = {".gif", ".png"}
MASK_KEY = ("manual", "mask", "label", "1st", "2nd", "_gt", "vessel", "seg")


def leading_num(p):
    m = re.search(r"(\d+)", Path(p).stem)
    return m.group(1) if m else None


def is_mask(p):
    s = str(p).lower()
    return Path(p).suffix.lower() in MASK_EXT and any(k in s for k in MASK_KEY)


def which_split(p):
    s = str(p).lower()
    if "train" in s:
        return "train"
    if "test" in s or "/val" in s or "\\val" in s:
        return "val"
    n = leading_num(p)
    if n is not None:
        return "train" if 21 <= int(n) <= 40 else "val"
    return None


def collect(root):
    """Walk root, return {(split,num): path} for images and masks."""
    imgs, masks = {}, {}
    for f in Path(root).rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in (IMG_EXT | MASK_EXT):
            continue
        sp = which_split(f)
        n = leading_num(f)
        if sp is None or n is None:
            continue
        if is_mask(f):
            masks.setdefault((sp, n), f)
        elif ext in {".tif", ".tiff", ".ppm", ".jpg", ".jpeg"}:
            imgs.setdefault((sp, n), f)
        elif ext == ".png":          # a .png that isn't flagged a mask → image
            imgs.setdefault((sp, n), f)
    return imgs, masks


def reorganize(imgs, masks, out):
    out = Path(out)
    written = {"train": 0, "val": 0}
    for (sp, n), imgp in sorted(imgs.items()):
        mp = masks.get((sp, n))
        if mp is None:
            print(f"  [skip] image {sp}/{n} ({imgp.name}) has no matching mask")
            continue
        (out / sp / "input").mkdir(parents=True, exist_ok=True)
        (out / sp / "label").mkdir(parents=True, exist_ok=True)
        Image.open(imgp).convert("RGB").save(out / sp / "input" / f"{n}.tif")
        Image.open(mp).convert("L").save(out / sp / "label" / f"{n}.png")
        written[sp] += 1
    return written


def try_datasets_fallback(repo, out):
    """If snapshot has no image files (e.g. parquet-packed), use the
    datasets library to decode examples into the DRIVE_HF tree."""
    print("  [fallback] trying datasets.load_dataset (parquet-packed repo)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not installed; `pip install datasets` to enable.")
        return {"train": 0, "val": 0}
    ds = load_dataset(repo)
    written = {"train": 0, "val": 0}
    img_keys = ("image", "img", "pixel_values", "fundus")
    msk_keys = ("label", "mask", "annotation", "segmentation", "manual")
    for split in ds:
        out_split = "train" if "train" in split.lower() else "val"
        for i, ex in enumerate(ds[split]):
            ik = next((k for k in img_keys if k in ex), None)
            mk = next((k for k in msk_keys if k in ex), None)
            if ik is None or mk is None:
                if i == 0:
                    print(f"  split '{split}' fields = {list(ex.keys())}; "
                          f"could not find image/mask keys")
                break
            (out / out_split / "input").mkdir(parents=True, exist_ok=True)
            (out / out_split / "label").mkdir(parents=True, exist_ok=True)
            ex[ik].convert("RGB").save(out / out_split / "input" / f"{i}.tif")
            ex[mk].convert("L").save(out / out_split / "label" / f"{i}.png")
            written[out_split] += 1
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="HF dataset repo id to download.")
    ap.add_argument("--src-dir", default=None,
                    help="Skip download; reorganize an already-unpacked dir.")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Output DRIVE_HF root.")
    args = ap.parse_args()
    out = Path(args.out)

    if args.src_dir:
        src = Path(args.src_dir)
        print(f"Reorganizing from local dir: {src}")
    else:
        print(f"Downloading HF repo: {args.repo}")
        from huggingface_hub import snapshot_download
        src = None
        for rtype in ("dataset", "model"):
            try:
                src = Path(snapshot_download(args.repo, repo_type=rtype))
                print(f"  downloaded ({rtype}) to {src}")
                break
            except Exception as e:
                print(f"  repo_type={rtype} failed: {e}")
        if src is None:
            print("Download failed for both repo types. Provide --src-dir "
                  "after manually downloading, or pass a valid --repo.")
            sys.exit(1)

    imgs, masks = collect(src)
    print(f"\nFound {len(imgs)} images, {len(masks)} masks in source.")
    written = reorganize(imgs, masks, out)

    if written["train"] + written["val"] == 0:
        written = try_datasets_fallback(args.repo, out)

    print(f"\n=== DRIVE_HF written to {out} ===")
    print(f"  train pairs: {written['train']}")
    print(f"  val   pairs: {written['val']}")
    for sp in ("train", "val"):
        n_tif = len(list((out / sp / "input").glob("*.tif"))) if (out/sp/"input").exists() else 0
        n_png = len(list((out / sp / "label").glob("*.png"))) if (out/sp/"label").exists() else 0
        flag = "OK" if n_tif == n_png and n_tif >= 1 else "!! mismatch/empty"
        print(f"  {sp}: {n_tif} input .tif / {n_png} label .png  {flag}")

    if written["train"] >= 1 and written["val"] >= 1:
        print("\nSUCCESS. drive_*.py scripts can now find data/DRIVE_HF/.")
    else:
        print("\nINCOMPLETE. Inspect the source layout; the repo structure may "
              "differ from DRIVE standard. Re-run with --src-dir after manual "
              "unpack, or check the field names printed above.")


if __name__ == "__main__":
    main()
