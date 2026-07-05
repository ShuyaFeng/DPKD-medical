"""
Find the single best seed (per Dr. Tian's confirmation: "single-best is
fine") for uniform and CANAL, at a chosen epsilon, from a results JSON.

Usage: python find_best_seeds.py --dataset busi --epsilon 2.0
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["busi", "kvasir", "isic"])
ap.add_argument("--epsilon", required=True, type=str)
args = ap.parse_args()

HERE = Path(__file__).parent
results_path = HERE / "results" / f"{args.dataset}_canal_k3_eps2_results.json"
R = json.loads(results_path.read_text())

seeds = R["seeds"]
e = args.epsilon

for method_key, canal_bool in [("PATE K=3 uniform", "False"), ("PATE K=3 + CANAL", "True")]:
    dices = R["series"][method_key][e]["dices"]
    best_idx = max(range(len(dices)), key=lambda i: dices[i])
    best_seed = seeds[best_idx]
    best_dice = dices[best_idx]
    ckpt_name = f"{args.dataset}_K3_canal{canal_bool}_eps{e}_seed{best_seed}.pt"
    print(f"{method_key}: best seed={best_seed}  dice={best_dice:.4f}  -> checkpoints/{ckpt_name}")
