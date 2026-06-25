"""
BUSI breast ultrasound lesion segmentation dataset.
Grayscale (in_ch=1) -- unlike DRIVE/ISIC/Kvasir (in_ch=3). Same
__getitem__ contract otherwise: returns (x: 1xsizexsize float32 in [0,1],
y: sizexsize int64 in {0,1}). Class 1 is the lesion (benign or malignant
combined).
Build the data first with: python download_busi.py --n 0 --size 96
"""
import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import transform

BUSI_ROOT = Path(__file__).resolve().parent.parent / "data" / "BUSI_HF"


class BUSIDataset(Dataset):
    def __init__(self, split: str = "train", size: int = 96):
        in_dir, lab_dir = BUSI_ROOT / split / "input", BUSI_ROOT / split / "label"
        self.items = []
        for p in sorted(glob.glob(str(in_dir / "*.png"))):
            stem = Path(p).stem
            lab = lab_dir / f"{stem}.png"
            if not lab.exists():
                continue
            img = np.array(Image.open(p).convert("L"), dtype=np.uint8)
            if img.shape[0] != size or img.shape[1] != size:
                img = transform.resize(img, (size, size), preserve_range=True,
                                       anti_aliasing=True).astype(np.uint8)
            x = (img.astype(np.float32) / 255.0)[None, :, :]

            m = np.array(Image.open(lab).convert("L"))
            if m.shape[0] != size or m.shape[1] != size:
                m = transform.resize(m.astype(float), (size, size), preserve_range=True,
                                     anti_aliasing=False, order=0)
            y = (m > 127).astype(np.int64)

            self.items.append((torch.from_numpy(x), torch.from_numpy(y)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
