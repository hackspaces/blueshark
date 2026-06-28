"""Build a verified SFT dataset of Indian-context agentic trajectories.

Teacher backends:
  --teacher claude-code   # uses your Claude Max login via `claude -p`
  --teacher api           # uses ANTHROPIC_API_KEY
  --teacher mock          # no calls; demonstrates the full loop offline

Examples:
  python build_dataset.py --teacher mock --n 4
  python build_dataset.py --teacher claude-code --n 50 --out train.jsonl
  python build_dataset.py --teacher api --model claude-opus-4-8 --n 50
"""
import argparse
from datagen.pipeline import build_dataset


def make_teacher(kind, model):
    if kind == "mock":
        from datagen.teacher import MockTeacher
        from datagen._mock_solutions import SOLUTIONS
        # mix: mostly 'good' (kept) plus some 'naive' (filtered out) to show the filter bite
        return MockTeacher(SOLUTIONS, quality="good")
    if kind == "claude-code":
        from datagen.teacher import ClaudeCodeTeacher
        return ClaudeCodeTeacher(model=model)
    if kind == "api":
        from datagen.teacher import AnthropicAPITeacher
        return AnthropicAPITeacher(model=model or "claude-opus-4-8")
    raise SystemExit(f"unknown teacher: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="mock", choices=["mock", "claude-code", "api"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=5, help="samples per domain")
    ap.add_argument("--out", default="train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    teacher = make_teacher(args.teacher, args.model)
    stats = build_dataset(teacher, n_per_domain=args.n, seed=args.seed, out_path=args.out)

    print(f"\n  Data engine — teacher={args.teacher}")
    print("  " + "-" * 46)
    for d, s in stats["by_domain"].items():
        print(f"  {d:10s}  kept {s['kept']:3d} / {s['attempted']:<3d} attempted")
    print("  " + "-" * 46)
    print(f"  TOTAL kept {stats['kept']}/{stats['attempted']} "
          f"(keep-rate {stats['keep_rate']*100:.0f}%)  ->  {stats['out_path']}")
    print("  Every kept row is VERIFIED correct against the real Indian rule.\n")


if __name__ == "__main__":
    main()
