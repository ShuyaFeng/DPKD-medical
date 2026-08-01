"""
Layer ablation: distill at e1, e2, or e3 (bottleneck).

All current experiments use e3 (the deepest bottleneck layer, base*4 channels).
This ablation tests whether earlier layers have:
  - More skewed importance distributions (more room for CANAL to help)
  - Better distillation signal for the student

Architecture (teacher base=32, student base=16, input 96×96):
  e1: 96×96 × teacher_base=32ch   /   student_base=16ch   (adapter: 16→32)
  e2: 48×48 × teacher_base*2=64ch /   student_base*2=32ch (adapter: 32→64)
  e3: 24×24 × teacher_base*4=128ch/   student_base*4=64ch (adapter: 64→128, current)

For each layer:
  1. Importance computed at that layer (gradient energy wrt. task loss)
  2. Caps computed at that layer (90th percentile L2 norm per channel)
  3. Cache: noisy teacher features at that layer
  4. Student MSE loss between adapter(student_feat) and noisy teacher_feat

Conditions: uniform vs CANAL at each layer, top 10% channel selection.
Budget split: rho_imp=10%, rho_rel=90%.

Usage:
  python canal_layer_ablation.py --dataset isic --seeds 3 --te 60 --se 40
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_pate_poc import train_K_teachers, correct_uniform_sigma
from drive_pate_pruning_joint import shared_importance, precompute_joint_cache
from drive_pate_canal_combined import add_dp_noise_to_importance, correct_waterfilling_sigma
from synthetic_demo import eps_to_rho, clip_and_normalise, denormalise, task_loss, TinyUNet

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

ALPHA_IMP = 0.1
CLIP_IMP = 1e-8
K = 3
KEEP_FRAC = 0.1


def get_dataset(name, split, size):
    if name == "isic":
        from isic_dataset import ISICDataset
        return ISICDataset(split, size), 3
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset
        return KvasirDataset(split, size), 3
    if name == "busi":
        from busi_dataset import BUSIDataset
        return BUSIDataset(split, size), 1
    raise ValueError(name)


def compute_importance_layer(model, loader, device, layer="e3"):
    """Gradient energy importance at e1, e2, or e3."""
    model.eval()
    ch = {"e1": model.base, "e2": model.base * 2, "e3": model.base * 4}[layer]
    grad_sum = torch.zeros(ch, device=device)
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        e1, e2, e3 = model.encode(x)
        feat = {"e1": e1, "e2": e2, "e3": e3}[layer]
        feat.retain_grad()
        loss = task_loss(model.decode(e1, e2, e3), y)
        loss.backward()
        if feat.grad is not None:
            grad_sum += feat.grad.detach().pow(2).mean(dim=(0, 2, 3))
        n += 1
    return (grad_sum / n).cpu()


@torch.no_grad()
def collect_caps_layer(model, loader, device, layer="e3", q=0.9):
    """Per-channel L2 cap at e1, e2, or e3."""
    model.eval()
    norms = []
    for x, _ in loader:
        x = x.to(device)
        e1, e2, e3 = model.encode(x)
        feat = {"e1": e1, "e2": e2, "e3": e3}[layer]
        for b in range(feat.shape[0]):
            norms.append(feat[b].flatten(1).norm(dim=1).cpu())
    return torch.quantile(torch.stack(norms), q, dim=0)


@torch.no_grad()
def precompute_layer_cache(teachers, caps_list_per_layer, train_ds, sigma,
                           active_mask, device, seed, layer="e3"):
    """Build noisy teacher cache at the given layer."""
    for t in teachers:
        t.eval()
    K = len(teachers)
    g = torch.Generator().manual_seed(seed)
    caps_mean = torch.stack([c.to(device) for c in caps_list_per_layer]).mean(dim=0)
    C = active_mask.shape[0]
    cache = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    for x, _ in loader:
        x = x.to(device)
        agg = None
        for t, caps in zip(teachers, caps_list_per_layer):
            e1, e2, e3 = t.encode(x)
            feat = {"e1": e1, "e2": e2, "e3": e3}[layer]
            feat_norm = clip_and_normalise(feat, caps.to(device))
            agg = feat_norm if agg is None else agg + feat_norm
        agg = agg / K
        B, Cc, H, W = agg.shape
        agg = agg * active_mask.view(1, Cc, 1, 1).float()
        noise = torch.randn(B, Cc, H, W, generator=g).to(device) * sigma.view(1, Cc, 1, 1)
        z_noisy = denormalise(agg + noise, caps_mean)
        for b in range(B):
            cache.append(z_noisy[b].detach().cpu())
    return cache


class Adapter(nn.Module):
    def __init__(self, s_ch, t_ch):
        super().__init__()
        self.conv = nn.Conv2d(s_ch, t_ch, 1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.conv(x)


def train_student_at_layer(train_ds, val_loader, cache, device, layer,
                            student_base=16, teacher_base=32,
                            n_epochs=40, lr=1e-3, lambda_feat=0.4,
                            seed=100, in_ch=3):
    """Train student with feature loss at the specified layer."""
    from drive_local_demo import evaluate_metrics
    torch.manual_seed(seed)
    student = TinyUNet(in_ch=in_ch, num_classes=2, base=student_base).to(device)
    s_ch = {"e1": student_base, "e2": student_base * 2, "e3": student_base * 4}[layer]
    t_ch = {"e1": teacher_base, "e2": teacher_base * 2, "e3": teacher_base * 4}[layer]
    adapter = Adapter(s_ch, t_ch).to(device)
    opt = torch.optim.Adam(list(student.parameters()) + list(adapter.parameters()), lr=lr)

    N = len(train_ds)
    batch_size = 4
    best_dice = 0.0
    best_state = None

    for ep in range(n_epochs):
        student.train(); adapter.train()
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idxs = perm[i:i + batch_size].tolist()
            xs = torch.stack([train_ds[idx][0] for idx in idxs]).to(device)
            ys = torch.stack([train_ds[idx][1] for idx in idxs]).to(device)
            t_targets = torch.stack([cache[idx] for idx in idxs]).to(device)

            opt.zero_grad()
            e1, e2, e3 = student.encode(xs)
            feat = {"e1": e1, "e2": e2, "e3": e3}[layer]
            s_proj = adapter(feat)
            f_loss = F.mse_loss(s_proj, t_targets)
            logits = student.decode(e1, e2, e3)
            t_loss = task_loss(logits, ys)
            (t_loss + lambda_feat * f_loss).backward()
            opt.step()

        metrics = evaluate_metrics(student, val_loader, device)
        val_dice = metrics["vessel_dice"]
        if val_dice > best_dice:
            best_dice = val_dice
            best_state = copy.deepcopy(student.state_dict())

    return best_dice


def cell_stats(dices):
    m = float(np.mean(dices))
    s = float(np.std(dices))
    sem = s / math.sqrt(len(dices))
    return {"dices": dices, "mean": m, "std": s, "sem": sem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="isic", choices=["isic", "kvasir", "busi"])
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--te", type=int, default=60)
    ap.add_argument("--se", type=int, default=40)
    ap.add_argument("--epsilons", default="0.5,1,2,4,6")
    ap.add_argument("--layers", default="e1,e2,e3", help="layers to ablate")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    LAYERS = args.layers.split(",")
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    print(f"[{args.dataset}] device={dev}  K={K}  eps={EPS}  layers={LAYERS}  seeds={SEEDS}")
    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds, _ = get_dataset(args.dataset, "val", args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    N = len(train_ds)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    print(f"  train={N}  val={len(val_ds)}  in_ch={in_ch}")

    teachers, _ = train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch)
    sens_imp = 2.0 * CLIP_IMP / float(N)

    results = {
        "dataset": args.dataset, "K": K, "ALPHA_IMP": ALPHA_IMP,
        "keep_frac": KEEP_FRAC, "epsilons": EPS, "seeds": SEEDS, "layers": LAYERS,
        "sweep": {},  # [layer][eps] → {uniform, canal}
    }

    for layer in LAYERS:
        print(f"\n{'='*72}")
        t_ch = {"e1": 32, "e2": 64, "e3": 128}[layer]  # teacher base=32
        print(f"Layer {layer}: teacher_ch={t_ch}  student_ch={t_ch//2}")

        # Compute caps per teacher at this layer
        caps_list_layer = []
        for t in teachers:
            caps = collect_caps_layer(t, train_loader, dev, layer=layer)
            caps_list_layer.append(caps.to(dev))

        # Importance at this layer (average across teachers)
        imp_list = [compute_importance_layer(t, train_loader, dev, layer=layer) for t in teachers]
        importance = torch.stack(imp_list, dim=0).mean(dim=0).to(dev)
        imp_cpu = importance.cpu()
        Cb = t_ch  # number of channels at this layer
        n_keep = max(1, int(round(KEEP_FRAC * Cb)))
        deltas = torch.full((Cb,), 2.0 / K, device=dev)

        print(f"  C={Cb}  n_keep={n_keep}  importance max/mean={float(imp_cpu.max()/imp_cpu.mean()):.2f}x"
              f"  max/min={float(imp_cpu.max()/imp_cpu.min()):.1f}x")

        results["sweep"][layer] = {
            "C": Cb, "n_keep": n_keep,
            "importance_max_over_mean": float(imp_cpu.max() / imp_cpu.mean()),
            "importance_max_over_min": float(imp_cpu.max() / imp_cpu.min()),
            "eps_results": {}
        }

        for eps in EPS:
            rho = eps_to_rho(eps)
            rho_imp = ALPHA_IMP * rho
            rho_rel = (1.0 - ALPHA_IMP) * rho
            noise_seed = int(eps * 100) + 7
            imp_noisy = add_dp_noise_to_importance(importance, sens_imp, rho_imp, seed=noise_seed)
            rank_desc = torch.argsort(imp_noisy, descending=True)
            top_indices = rank_desc[:n_keep]

            top_mask = torch.zeros(Cb, dtype=torch.bool, device=dev)
            top_mask[top_indices] = True

            sigma_uni = torch.zeros(Cb, device=dev)
            sigma_uni[top_mask] = correct_uniform_sigma(deltas[top_mask], rho_rel)

            sigma_canal = torch.zeros(Cb, device=dev)
            sigma_canal[top_mask] = correct_waterfilling_sigma(
                deltas[top_mask], imp_noisy[top_mask], rho_rel
            )

            ep_res = {}
            for cond, sigma in [("uniform", sigma_uni), ("canal", sigma_canal)]:
                print(f"  {layer}  eps={eps}  {cond} ...", end="", flush=True)
                t0 = time.time()
                cache = precompute_layer_cache(
                    teachers, caps_list_layer, train_ds, sigma, top_mask, dev,
                    seed=int(eps * 1000) + (1 if cond == "uniform" else 2) + hash(layer) % 100,
                    layer=layer
                )
                dices = []
                for s in SEEDS:
                    best = train_student_at_layer(
                        train_ds, val_loader, cache, dev, layer=layer,
                        student_base=16, teacher_base=32,
                        n_epochs=args.se, lr=1e-3, lambda_feat=0.4,
                        seed=s, in_ch=in_ch,
                    )
                    dices.append(best)
                ep_res[cond] = cell_stats(dices)
                print(f"  Dice={ep_res[cond]['mean']:.4f}±{ep_res[cond]['sem']:.4f}  ({time.time()-t0:.1f}s)")

            ep_res["canal_minus_uniform"] = ep_res["canal"]["mean"] - ep_res["uniform"]["mean"]
            results["sweep"][layer]["eps_results"][str(eps)] = ep_res

    # Summary
    for eps in EPS:
        print(f"\n{'='*72}")
        print(f"LAYER ABLATION  ({args.dataset.upper()}, eps={eps})")
        print(f"{'layer':>6} | {'C':>5} | {'uniform':>9} | {'canal':>9} | {'diff':>7} | {'imp_ratio':>10}")
        print("-"*60)
        for layer in LAYERS:
            r = results["sweep"][layer]
            er = r["eps_results"][str(eps)]
            print(f"{layer:>6} | {r['C']:>5} | {er['uniform']['mean']:>9.4f} | "
                  f"{er['canal']['mean']:>9.4f} | {er['canal_minus_uniform']:>+7.4f} | "
                  f"{r['importance_max_over_mean']:>9.2f}x")

    out = HERE / "results" / f"{args.dataset}_layer_ablation_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
