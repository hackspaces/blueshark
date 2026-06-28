# datagen — the frontier-teacher data engine

Turns a teacher model (your Claude Max login, the API, or any open model) into a stream
of **verified** training trajectories for Indian-context agentic tasks. This is the
"use my Claude subscription to make the data" step — built so the data is auto-filtered
for correctness before it's ever used.

## The loop (this is the whole pitch)

```
generate a task  ->  teacher solves it (read->reason->act->verify CoT)  ->
  run the answer through the indeval rule-check  ->  KEEP only if it passes  ->  write JSONL
```

Because our domain has hard ground truth (a GSTIN checksum is right or wrong), the
filtering step that costs others days of hand-review is, for us, automatic. **Every row
in the output dataset is verified correct against the real Indian rule.**

## Prove it works (no API, no auth, no GPU)

```bash
python build_dataset.py --teacher mock --n 4
```

- A **rule-knowing** teacher  -> keep-rate **100%**
- A **plausible-but-ignorant** teacher -> keep-rate **0%** (every bad trajectory dropped)

That gap is the point: the engine only emits training data that actually encodes the
Indian rule.

## Run it for real

```bash
# uses your Claude Max subscription via Claude Code, headless:
python build_dataset.py --teacher claude-code --n 50 --out train.jsonl

# or the API:
ANTHROPIC_API_KEY=... python build_dataset.py --teacher api --model claude-opus-4-8 --n 50
```

Output is standard chat-format JSONL (system / user / assistant) ready for SFT with
Unsloth or TRL. Loss is taken on the assistant turn (the read->reason->act->verify
trajectory + the verified solution).

## Design notes a panel will respect

- **Contamination control:** training instances here are deliberately DISTINCT from the
  indeval eval tasks (different prompts, signatures, freshly randomised vectors). They
  share the rule, not the items. The eval set stays held out.
- **Teacher-agnostic:** swap Claude for Qwen/Gemma if licensing or cost demands it — the
  backend is one class.
- **The teacher caps the student:** distillation only transfers what the teacher knows;
  for long-tail Indian edge cases, mix in real documentation / human examples.

## Honest caveat (say it out loud)

Automating a Claude subscription and using model outputs to train another model has
licensing implications under Anthropic's usage policy. Confirm the path before relying on
it at scale; the pluggable backend means you are never locked to one teacher.
