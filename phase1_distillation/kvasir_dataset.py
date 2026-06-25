"""
Kvasir-SEG polyp segmentation dataset.
Same __getitem__ contract as ISICDataset / DriveDataset -- returns
(x: 3xsizexsize float32 in [0,1], y: sizexsize int64 in {0,1}) -- drops
straight into the dataset-agnostic pipeline helpers. Here class 1 is the
polyp, so "vessel_dice" measures polyp Dice.
Build the data first with: python download_kvasir.py --n 0 --size 96
"""
import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import transform

KVASIR_ROOT = Path(__file__).resolve().parent.parent / "data" / "KVASIR_HF"


class KvasirDataset(Dataset):
    def __init__(self, split: str = "train", size: int = 96):
        in_dir, lab_dir = KVASIR_ROOT / split / "input", KVASIR_ROOT / split / "label"
        self.items = []
        for p in sorted(glob.glob(str(in_dir / "*.png"))):
            stem = Path(p).stem
            lab = lab_dir / f"{stem}.png"
            if not lab.exists():
                continue
            img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
            if img.shape[0] != size or img.shape[1] != size:
                img = transform.resize(img, (size, size, 3), preserve_range=True,
                                       anti_aliasing=True).astype(np.uint8)
            x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)

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