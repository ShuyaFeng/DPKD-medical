"""Smoke test: validate the loader + full DP pipeline on the ISIC subset.
(1) clean teacher lesion Dice  (2) K=3 PATE + keep-2% subsampling @ ε=2 student."""
import sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from isic_dataset import ISICDataset
from drive_local_demo import TinyUNet, train_teacher, evaluate_vessel_dice
from drive_pate_poc import train_K_teachers
from drive_pate_pruning_joint import shared_importance, thresholded_uniform_sigma, precompute_joint_cache
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
train_ds, val_ds = ISICDataset("train", 96), ISICDataset("val", 96)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
print(f"ISIC subset: train={len(train_ds)}  val={len(val_ds)}")

# (1) single clean teacher — sanity that lesion segmentation learns
t0 = time.time()
teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
train_teacher(teacher, DataLoader(train_ds, batch_size=8, shuffle=True), n_epochs=40, lr=1e-3, device=device)
clean = evaluate_vessel_dice(teacher, val_loader, device)
print(f"\n[1] clean teacher lesion Dice = {clean:.4f}   ({time.time()-t0:.0f}s)")

# (2) full DP joint pipeline: K=3 PATE + keep-2% subsampling @ ε=2
K = 3
teachers, caps_list = train_K_teachers(train_ds, K, device, n_epochs=40)
Cb = teachers[0].base * 4
imp = shared_importance(teachers, DataLoader(train_ds, batch_size=8), device).to(device)
rank = torch.argsort(imp, descending=True)
n_active = max(1, int(round(0.02 * Cb)))
active = torch.zeros(Cb, dtype=torch.bool, device=device); active[rank[:n_active]] = True
deltas = torch.full((Cb,), 2.0 / K, device=device)
eps = 2.0
sigma = thresholded_uniform_sigma(deltas, eps_to_rho(eps), active)
cache = precompute_joint_cache(teachers, caps_list, train_ds, sigma, active, device, seed=42)
best, _ = train_student_distill(train_ds, val_loader, cache, device,
                                student_base=16, teacher_base=32,
                                n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=100)
print(f"\n[2] DP joint K={K} keep2% ε={eps}: student lesion Dice = {best:.4f}")
print(f"    (n_active={n_active}/{Cb}, σ_active={sigma[active].mean():.3f})")
print("\nSMOKE OK — loader + PATE + subsampling + student distill all run on ISIC.")
