"""Architecture bake-off — screen configs at MATCHED COMPUTE before committing
to an expensive run. The principle: a config that's better per-unit-compute at
small scale almost always stays better at scale, so we rank cheaply and only
scale the winner (see the strategy notes in STATUS.md / EXPERIMENTS.md).

Fairness: configs cost different amounts per step (recurrence and active params
both raise it), so comparing at equal *steps* is unfair. We equalize FLOPs
instead: cost_i = active_params_i * recurrence_i, and steps_i = ref_steps *
max_cost / cost_i, so every config consumes ~the same compute. We rank by the
val loss each reaches under that equal budget.

Usage (run where a packed dataset + tokenizer live, e.g. the GPU pod):
  python pipeline/bakeoff.py --data py_pretrain --configs proof recur3 deep8 coherent \
      --ref-steps 1500 --batch 8 --seq 512

Each config trains in its own subprocess (clean GPU memory), then we read the
best val loss from its metrics.jsonl and print a ranked table.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs import build_config, PRESETS   # noqa: E402
from model import Model                     # noqa: E402

PACKED = os.path.join(ROOT, "data", "packed")


def cost_of(cfg_name, vocab):
    """Relative per-token compute: active expert params * recurrence."""
    cfg = build_config(cfg_name, vocab)
    m = Model(cfg)
    total = sum(p.numel() for p in m.parameters())
    one_expert = sum(p.numel() for p in m.blocks[0].moe.experts[0].parameters())
    active = total - (cfg.n_routed_experts - cfg.n_active_experts) * one_expert * cfg.n_layers
    rec = getattr(cfg, "recurrence", 1)
    return {"total": total, "active": active, "recurrence": rec,
            "layers": cfg.n_layers, "eff_layers": cfg.n_layers * rec,
            "cost": active * rec}


def best_val(run_dir):
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    vals = [json.loads(l)["val_loss"] for l in open(path) if '"val_loss"' in l]
    return min(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="packed basename under data/packed/")
    ap.add_argument("--configs", nargs="+", required=True, help="preset names to compare")
    ap.add_argument("--ref-steps", type=int, default=1500, help="steps for the most expensive config")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--out", default="runs/bakeoff")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(PACKED, f"{args.data}.meta.json")))
    vocab = meta["vocab_size"]

    # cost-equalize: most expensive config does --ref-steps, cheaper ones do more
    costs = {c: cost_of(c, vocab) for c in args.configs}
    max_cost = max(v["cost"] for v in costs.values())
    steps = {c: max(50, round(args.ref_steps * max_cost / v["cost"])) for c, v in costs.items()}

    print("matched-compute plan (equal FLOPs):")
    for c in args.configs:
        v = costs[c]
        print(f"  {c:12s} total {v['total']/1e6:5.0f}M active {v['active']/1e6:5.0f}M "
              f"x rec{v['recurrence']} = {v['eff_layers']} eff layers -> {steps[c]} steps")

    results = []
    for c in args.configs:
        run = os.path.join(args.out, c)
        print(f"\n=== training {c} ({steps[c]} steps) ===", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "pipeline", "train.py"),
                        "--data", args.data, "--config", c, "--out", run,
                        "--steps", str(steps[c]), "--batch", str(args.batch),
                        "--seq", str(args.seq), "--lr", str(args.lr),
                        "--eval-every", str(max(50, steps[c] // 6)),
                        "--save-every", str(steps[c])], cwd=ROOT)
        results.append((c, best_val(run)))

    results.sort(key=lambda r: (r[1] is None, r[1]))
    print("\n" + "=" * 58)
    print("  BAKE-OFF RESULTS (matched compute, lower val loss = better)")
    print("=" * 58)
    print(f"  {'config':12s} {'val_loss':>9s} {'total':>7s} {'active':>7s} {'eff_L':>6s} {'steps':>6s}")
    for c, v in results:
        ci = costs[c]
        print(f"  {c:12s} {('%.4f'%v) if v else '   n/a':>9s} "
              f"{ci['total']/1e6:6.0f}M {ci['active']/1e6:6.0f}M {ci['eff_layers']:>6d} {steps[c]:>6d}")
    if results and results[0][1]:
        print(f"\n  winner: {results[0][0]} (val {results[0][1]:.4f})")
    print("  note: single-size screen. For the scaling SLOPE, rerun across 2-3 sizes per config.")


if __name__ == "__main__":
    main()
