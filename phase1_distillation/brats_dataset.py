"""
BraTS multi-modal (T1/T1ce/T2/FLAIR) 2D-slice dataset.

Same __getitem__ contract as DriveDataset/ISICDataset but with 4 INPUT CHANNELS:
returns (x: 4×size×size float32 in [0,1], y: size×size int64 {0,1}) where class 1
is the whole tumor. Pass in_ch=4 to the pipeline helpers (train_K_teachers,
train_student_distill) when using this dataset.

Build the data first:
  cluster:  python prep_brats.py --src /path/to/BraTS2021 --size 96
  local logic check: python prep_brats.py --hf-mini --size 96
"""
import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

BRATS_ROOT = Path(__file__).resolve().parent.parent / "data" / "BraTS_HF"


class BRATSDataset(Dataset):
    def __init__(self, split: str = "train", size: int = 96):
        in_dir, lab_dir = BRATS_ROOT / split / "input", BRATS_ROOT / split / "label"
        self.items = []
        for p in sorted(glob.glob(str(in_dir / "*.npy"))):
            stem = Path(p).stem
            lab = lab_dir / f"{stem}.npy"
            if not lab.exists():
                continue
            x = np.load(p).astype(np.float32)            # (4, size, size)
            y = np.load(lab).astype(np.int64)            # (size, size)
            self.items.append((torch.from_numpy(x), torch.from_numpy(y)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
