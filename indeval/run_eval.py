"""Run the eval against a REAL model-under-test.

Works with any OpenAI-compatible endpoint:

  # Gemma / Qwen / Ornith served locally on the laptop or an IndiaAI node:
  vllm serve google/gemma-3-12b-it --port 8000
  python run_eval.py --base-url http://localhost:8000/v1 --model google/gemma-3-12b-it

  # any hosted OpenAI-compatible API:
  python run_eval.py --base-url https://api.example.com/v1 --model some-model --api-key $KEY

The model gets one self-debug round (the agentic loop). Results print as a table and
are written to results.json.
"""
import argparse
from indeval.tasks import TASKS
from indeval.grader import grade
from indeval.agent import solve
from indeval import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--rounds", type=int, default=2, help="agentic self-debug rounds (1 = single-shot)")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    results = []
    for task in TASKS:
        print(f"  running {task.task_id} ...", flush=True)
        code = solve(task, args.base_url, args.model, api_key=args.api_key,
                     max_rounds=args.rounds, grader=grade)
        results.append(grade(task, code))

    print(report.render(f"model = {args.model}", results))
    with open(args.out, "w") as f:
        f.write(report.to_json(args.model, results))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
