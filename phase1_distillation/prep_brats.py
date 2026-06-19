"""
Prepare BraTS (multi-modal MRI tumor segmentation) for the 2D pipeline.

Each case = 4 modality volumes (T1, T1ce, T2, FLAIR) + a seg volume, all
(240,240,155) NIfTI. We take the axial slice(s) with the most tumor, stack the
4 modalities -> a 4-channel 2D input, and binarize seg to WHOLE TUMOR ({1,2,4}->1).
Output mirrors DRIVE/ISIC:

    data/BraTS_HF/{train,val}/input/<case>.npy   # (4, size, size) float32 in [0,1]
                              /label/<case>.npy   # (size, size) uint8 {0,1}

Per-patient unit: split is by CASE. --slices-per-case 1 keeps k=1 (one image per
patient, like DRIVE/ISIC); >1 yields k>1 (tests per-patient composition).

Modes:
  --src <BraTS_root>   standard per-case NIfTI folders (the cluster path)
  --hf-mini            LOGIC-ONLY local check: pulls anhaltai/brats2021_mini
                       (1 modality, 10 cases) and replicates it to 4 channels
                       just to validate slicing/loader shapes (NOT real 4-modal).

Usage (cluster):  python prep_brats.py --src /path/to/BraTS2021 --n 0 --size 96
Usage (local):    python prep_brats.py --hf-mini --size 96
"""
import argparse, glob, io, gzip, tempfile, os, random
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage.transform import resize as sk_resize

ROOT = Path(__file__).resolve().parent.parent / "data"
OUT = ROOT / "BraTS_HF"
MODS = ["t1", "t1ce", "t2", "flair"]      # channel order
WHOLE_TUMOR = (1, 2, 4)


def norm01(ch):
    """Per-modality robust [0,1] using the 1/99 percentiles of nonzero voxels."""
    nz = ch[ch > 0]
    if nz.size == 0:
        return np.zeros_like(ch, dtype=np.float32)
    lo, hi = np.percentile(nz, 1), np.percentile(nz, 99)
    return np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def best_slices(seg, k):
    """Axial slice indices (last axis) with the most tumor; fallback to middle."""
    area = (seg > 0).reshape(-1, seg.shape[2]).sum(0)
    if area.max() == 0:
        return [seg.shape[2] // 2]
    return list(np.argsort(area)[::-1][:k])


def save_case(cid, mods4, seg, slices, size, split):
    """mods4: list of 4 (H,W,D) arrays; seg: (H,W,D). Write one .npy per slice."""
    for j, z in enumerate(slices):
        chans = np.stack([sk_resize(norm01(m[:, :, z]), (size, size), preserve_range=True,
                                    anti_aliasing=True) for m in mods4], 0).astype(np.float32)
        y = np.isin(seg[:, :, z], WHOLE_TUMOR).astype(np.float32)
        y = (sk_resize(y, (size, size), preserve_range=True, anti_aliasing=False, order=0) > 0.5).astype(np.uint8)
        stem = f"{cid}_z{z}" if len(slices) > 1 else cid
        np.save(OUT / split / "input" / f"{stem}.npy", chans)
        np.save(OUT / split / "label" / f"{stem}.npy", y)


def load_nii(path):
    return nib.load(str(path)).get_fdata()


def find_cases(src):
    """Return {case_id: {mod: path, 'seg': path}} from a standard BraTS tree."""
    cases = {}
    for seg in glob.glob(os.path.join(src, "**", "*_seg.nii.gz"), recursive=True):
        pref = seg[:-len("_seg.nii.gz")]
        cid = os.path.basename(pref)
        rec = {"seg": seg}
        for m in MODS:
            for cand in (f"{pref}_{m}.nii.gz", f"{pref}_{m.replace('t1ce','t1c')}.nii.gz"):
                if os.path.exists(cand):
                    rec[m] = cand; break
        if all(m in rec for m in MODS):
            cases[cid] = rec
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=None, help="BraTS root (per-case NIfTI folders)")
    ap.add_argument("--hf-mini", action="store_true", help="local logic check (1-modality mini -> 4ch)")
    ap.add_argument("--n", type=int, default=0, help="subset #cases (0=all)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--slices-per-case", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for sp in ("train", "val"):
        for s in ("input", "label"):
            (OUT / sp / s).mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    if args.hf_mini:
        print("[LOGIC-ONLY] anhaltai/brats2021_mini: 1 modality replicated to 4ch (shape check, NOT real multi-modal)")
        from huggingface_hub import hf_hub_download
        import pandas as pd
        df = pd.read_parquet(hf_hub_download("anhaltai/brats2021_mini",
                             "data/train-00000-of-00001.parquet", repo_type="dataset"))
        ids = list(range(len(df))); rng.shuffle(ids)
        nval = max(1, int(len(ids) * args.val_frac))
        for rank, i in enumerate(ids):
            def to_vol(col):
                with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
                    f.write(gzip.decompress(df.iloc[i][col]["bytes"])); tmp = f.name
                v = nib.load(tmp).get_fdata(); os.unlink(tmp); return v
            vol, seg = to_vol("image"), to_vol("annotations")
            mods4 = [vol] * 4                      # replicate (logic only)
            save_case(f"mini{i:03d}", mods4, seg, best_slices(seg, args.slices_per_case),
                      args.size, "val" if rank < nval else "train")
        print(f"done -> {OUT}  (logic validation)")
        return

    assert args.src, "give --src <BraTS root> (cluster) or --hf-mini (local)"
    cases = find_cases(args.src)
    cids = sorted(cases); rng.shuffle(cids)
    if args.n > 0:
        cids = cids[:args.n]
    nval = int(len(cids) * args.val_frac)
    print(f"BraTS: {len(cids)} cases (val={nval})  size={args.size}  slices/case={args.slices_per_case}")
    for rank, cid in enumerate(cids):
        rec = cases[cid]
        mods4 = [load_nii(rec[m]) for m in MODS]
        seg = load_nii(rec["seg"])
        save_case(cid, mods4, seg, best_slices(seg, args.slices_per_case),
                  args.size, "val" if rank < nval else "train")
        if (rank + 1) % 25 == 0:
            print(f"  {rank+1}/{len(cids)}")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
