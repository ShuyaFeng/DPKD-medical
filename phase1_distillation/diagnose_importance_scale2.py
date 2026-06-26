import sys, math
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from busi_dataset import BUSIDataset
from drive_pate_poc import train_K_teachers
from drive_pate_pruning_joint import shared_importance
from torch.utils.data import DataLoader
from synthetic_demo import eps_to_rho

dev = "cuda" if torch.cuda.is_available() else "cpu"
train_ds = BUSIDataset("train", 96)
tk = train_K_teachers(train_ds, 3, dev, n_epochs=60, in_ch=1)  # match the REAL run's epoch count
imp = shared_importance(tk[0], DataLoader(train_ds, batch_size=8), dev)

print(f"importance: min={imp.min().item():.10f} max={imp.max().item():.10f} mean={imp.mean().item():.10f}")
print(f"importance (raw tensor, first 10 values): {imp[:10]}")
print(f"nonzero count: {(imp != 0).sum().item()} / {imp.numel()}")
