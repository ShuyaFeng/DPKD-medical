import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from busi_dataset import BUSIDataset
from drive_pate_poc import train_K_teachers
from drive_pate_pruning_joint import shared_importance
from torch.utils.data import DataLoader

dev = "cuda" if torch.cuda.is_available() else "cpu"
train_ds = BUSIDataset("train", 96)
tk = train_K_teachers(train_ds, 3, dev, n_epochs=10, in_ch=1)  # quick, just for scale check
imp = shared_importance(tk[0], DataLoader(train_ds, batch_size=8), dev)

print(f"importance: min={imp.min().item():.6f} max={imp.max().item():.6f} mean={imp.mean().item():.6f}")

CLIP_IMP = 100.0
N = len(train_ds)
sens_imp = 2.0 * CLIP_IMP / N
print(f"sens_imp = {sens_imp:.6f}")

for rho_imp_frac, eps in [(0.1, 2.0), (0.1, 8.0)]:
    from synthetic_demo import eps_to_rho
    rho = eps_to_rho(eps)
    rho_imp = rho_imp_frac * rho
    import math
    sigma_imp = sens_imp / math.sqrt(2.0 * rho_imp)
    print(f"eps={eps}: rho_imp={rho_imp:.6f}  sigma_imp={sigma_imp:.6f}  "
          f"(importance mean={imp.mean().item():.6f} -> noise is {sigma_imp/imp.mean().item():.1f}x the mean importance)")
