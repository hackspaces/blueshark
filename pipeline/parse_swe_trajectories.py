"""Parse Nebius SWE-agent-trajectories into blueshark's agentic token format.

Each raw trajectory is an alternating ai/user message list:
  - user[0]            = the GitHub issue (the task)
  - ai[k]              = reasoning text + a ```command``` block (reason + act)
  - user[k+1]          = the command's stdout (the observation / tool result)
  - final ai           = reasoning + `submit`

We render each turn into the reserved control tokens:
  <think>{reasoning}</think><tool_call>{command}<tool_result>{observation}
(Only </think> has a closing token by design; <tool_call> is implicitly ended
by the following <tool_result>, which is ended by the next <think> or EOS.)

For proper SFT we emit per-segment loss masks: the model is trained to PRODUCE
its own turns (<think>/<tool_call> spans, trainable=1) but NOT the issue text or
the <tool_result> observations the environment hands back (trainable=0).

Output (to data/sft/):
  swe_agentic_sft.jsonl  - one record/trajectory: {instance_id, resolved, segments:[[text,trainable]...]}
  swe_agentic.txt        - the rendered texts concatenated, for plain LM/pretraining use

Usage:
  .venv/bin/python pipeline/parse_swe_trajectories.py                 # resolved-only (gold)
  .venv/bin/python pipeline/parse_swe_trajectories.py --all           # every trajectory
  .venv/bin/python pipeline/parse_swe_trajectories.py --max 2000      # cap count
"""
import argparse
import glob
import json
import os
import re

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "swe_trajectories")
OUT = os.path.join(ROOT, "data", "sft")

CODE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.DOTALL)
PLAN, THINK, THINK_END, CALL, RESULT = "<plan>", "<think>", "</think>", "<tool_call>", "<tool_result>"


def parse_ai(text):
    """Split an ai message into (reasoning, command)."""
    commands = [m.strip() for m in CODE.findall(text)]
    reasoning = CODE.sub("", text).strip()
    return reasoning, "\n".join(c for c in commands if c).strip()


def parse_trajectory(traj):
    """Return (issue, turns) where each turn = {think, tool_call, tool_result}."""
    msgs = [m for m in traj if m.get("role") in ("ai", "user")]
    issue, i = "", 0
    if msgs and msgs[0]["role"] == "user":
        issue, i = msgs[0]["text"].strip(), 1

    turns = []
    while i < len(msgs):
        if msgs[i]["role"] == "ai":
            think, call = parse_ai(msgs[i]["text"])
            obs = None
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user":
                obs = msgs[i + 1]["text"].strip()
                i += 2
            else:
                i += 1
            turns.append({"think": think, "tool_call": call, "tool_result": obs})
        else:
            i += 1
    return issue, turns


def build_segments(issue, turns):
    """Return [[text, trainable], ...]. trainable=1 on the model's own turns
    (think + tool_call), 0 on context the model should NOT learn to produce
    (the issue and the environment's tool_result observations).

    The issue is followed by a FIXED entry marker `<plan>` (prompt-side, mask 0):
    it gives the model a constant boundary to learn 'after <plan>, emit <think>'
    (the varied issue text alone was too rare a cue), and it GATES the agentic
    loop — plain prompts without <plan> stay in code-completion mode instead of
    bleeding agent artifacts. At inference, prompt with `{issue}<plan>` for an
    agentic turn; omit it for plain code completion."""
    segs = [[issue.strip() + "\n", 0], [PLAN, 0]]
    for t in turns:
        if t["think"]:
            segs.append([f"{THINK}{t['think']}{THINK_END}", 1])
        if t["tool_call"]:
            segs.append([f"{CALL}{t['tool_call']}\n", 1])
        if t["tool_result"] is not None:
            segs.append([f"{RESULT}{t['tool_result']}\n", 0])
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include unresolved trajectories")
    ap.add_argument("--max", type=int, default=0, help="cap number of trajectories (0 = no cap)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet shards in {RAW}")

    jsonl_path = os.path.join(OUT, "swe_agentic_sft.jsonl")
    txt_path = os.path.join(OUT, "swe_agentic.txt")
    n = kept = total_turns = total_chars = 0

    with open(jsonl_path, "w", encoding="utf-8") as jf, open(txt_path, "w", encoding="utf-8") as tf:
        for f in files:
            t = pq.read_table(f, columns=["instance_id", "model_name", "target",
                                          "trajectory", "generated_patch"])
            cols = {c: t.column(c).to_pylist() for c in t.column_names}
            for k in range(t.num_rows):
                n += 1
                resolved = bool(cols["target"][k])
                if not resolved and not args.all:
                    continue
                issue, turns = parse_trajectory(cols["trajectory"][k])
                if not turns:
                    continue
                segments = build_segments(issue, turns)
                text = "".join(s[0] for s in segments)
                rec = {
                    "instance_id": cols["instance_id"][k],
                    "model_name": cols["model_name"][k],
                    "resolved": resolved,
                    "n_turns": len(turns),
                    "segments": segments,
                    "generated_patch": cols["generated_patch"][k],
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tf.write(text + "\n<|endoftext|>\n")
                kept += 1
                total_turns += len(turns)
                total_chars += len(text)
                if args.max and kept >= args.max:
                    break
            if args.max and kept >= args.max:
                break

    print(f"scanned {n:,} trajectories, kept {kept:,} "
          f"({'all' if args.all else 'resolved-only'})")
    print(f"avg turns/traj: {total_turns/max(kept,1):.1f}")
    print(f"total chars: {total_chars/1e6:.1f}M  (~{total_chars//4/1e6:.1f}M tokens rough)")
    print(f"wrote {jsonl_path}")
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
