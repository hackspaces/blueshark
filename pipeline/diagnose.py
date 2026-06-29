"""Read the gauges of a training run and turn them into a verdict.

Maps the metrics in a run's metrics.jsonl onto the three laws (see LEARNING.md):
  - descent stability   <- grad-norm behaviour
  - capacity vs data    <- generalization gap (val - train)
  - optimization health <- loss reduction, MoE load balance

  python pipeline/diagnose.py runs/sft_proof
"""
import json
import os
import sys
import statistics as st


def load(run_dir):
    p = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(p):
        sys.exit(f"no metrics.jsonl in {run_dir}")
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = load(run)
    train = [(r["step"], r["ema"]) for r in rows if "ema" in r]
    val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    gnorm = [r["grad_norm"] for r in rows if "grad_norm" in r]
    gaps = [(r["step"], r["gen_gap"]) for r in rows if "gen_gap" in r]
    imb = [r["imbalance"] for r in rows if "imbalance" in r]

    print(f"\n=== diagnostics: {run} ===")
    if train:
        print(f"train loss: {train[0][1]:.3f} -> {train[-1][1]:.3f}  (down {train[0][1]-train[-1][1]:.2f})")
    if val:
        best = min(v for _, v in val)
        print(f"val loss:   best {best:.3f}  final {val[-1][1]:.3f}")

    print("\n[law 1] descent stability — grad norm")
    if gnorm:
        med = st.median(gnorm); mx = max(gnorm)
        spikes = sum(1 for g in gnorm if g > 3 * med)
        print(f"  median {med:.2f}  max {mx:.2f}  spikes(>3x median) {spikes}/{len(gnorm)}")
        v1 = "STABLE" if spikes <= len(gnorm) * 0.05 and mx < 10 else "UNSTABLE — lower LR / add QK-norm"
        print(f"  verdict: {v1}")

    print("\n[law 3] capacity vs data — generalization gap (val - train)")
    if gaps:
        trend = gaps[-1][1] - gaps[0][1]
        print(f"  gap: {gaps[0][1]:+.3f} -> {gaps[-1][1]:+.3f}  (trend {trend:+.3f})")
        if trend > 0.15:
            print("  verdict: OVERFITTING — gap widening; need more/cleaner data or a smaller model")
        elif gaps[-1][1] > 0.5:
            print("  verdict: high gap — data-bound; more data would help")
        else:
            print("  verdict: HEALTHY — capacity matched to data")

    if imb:
        print("\n[MoE] expert load balance (max/mean, 1.0 = perfect)")
        print(f"  mean {st.mean(imb):.2f}  final {imb[-1]:.2f}  "
              f"-> {'balanced' if imb[-1] < 1.5 else 'imbalanced — check balancer'}")
    print()


if __name__ == "__main__":
    main()
