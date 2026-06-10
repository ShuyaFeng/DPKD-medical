"""
EXP-1: Active-set importance ratio R analysis.

PURPOSE
-------
Before running any expensive distillation, answer the single question:

  "Within the top-K most-important bottleneck channels, what is the
   importance ratio R = s_top1 / s_topK?"

If full-bottleneck R = 1.96 is killing channel-WF, maybe the active
subset (after pruning bottom 90%) has a bigger R and WF can be revived.

This is PURE ANALYSIS — no training. Reads/recomputes importance and
reports R at several keep-fractions. Takes ~30 seconds.

DECISION RULE
-------------
  R(top 10%) >  5  -> spend half a day running EXP-2 ablation. CANAL has a chance.
  R(top 10%) <  3  -> abandon CANAL rescue. Skip to EXP-3 (spatial-WF).

OUTPUT
------
  drive_active_set_R_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher, compute_importance,
)


def report_R_at_keep_fractions(importance: torch.Tensor):
    """For each keep fraction k, report R = top1 / top(k*C)."""
    C = importance.shape[0]
    sorted_imp, _ = torch.sort(importance, descending=True)
    rows = []
    for k_frac in [1.00, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01]:
        kept = max(1, int(round(k_frac * C)))
        R = float(sorted_imp[0] / sorted_imp[kept - 1])
        top_mean = float(sorted_imp[:kept].mean())
        bot_mean = float(sorted_imp[kept:].mean()) if kept < C else 0.0
        rows.append({
            "keep_fraction": k_frac,
            "n_active": kept,
            "R_top1_over_topK": R,
            "active_mean": top_mean,
            "pruned_mean": bot_mean,
        })
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    print("\n[1/2] Loading DRIVE + training teacher (60 epochs)...")
    train_ds = DriveDataset("train", size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3, device=device)

    print("\n[2/2] Computing importance + R at multiple keep fractions...")
    importance = compute_importance(teacher, train_loader, device).cpu()
    C = importance.shape[0]
    print(f"  Bottleneck channels C = {C}")

    rows = report_R_at_keep_fractions(importance)

    print()
    print(f"  {'keep%':>6s}  {'n_active':>8s}  {'R top1/topK':>13s}  "
          f"{'active_mean':>12s}  {'pruned_mean':>12s}")
    print("  " + "-" * 64)
    for r in rows:
        print(f"  {r['keep_fraction']*100:>5.1f}%  {r['n_active']:>8d}  "
              f"{r['R_top1_over_topK']:>13.2f}  "
              f"{r['active_mean']:>12.4e}  {r['pruned_mean']:>12.4e}")
    print()

    out = {
        "C": C,
        "importance_global_R": float(importance.max() / importance.min()),
        "keep_fractions": rows,
        "decision_rule": {
            "R_top_10pct_threshold_for_CANAL_rescue": 5.0,
            "go_or_no_go": "GO" if rows[3]["R_top1_over_topK"] >= 5.0 else "NO-GO",
        },
    }
    out_path = Path(__file__).parent / "drive_active_set_R_results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"  Wrote: {out_path}")
    print(f"  DECISION: {out['decision_rule']['go_or_no_go']} on CANAL rescue (EXP-2)")


if __name__ == "__main__":
    main()
