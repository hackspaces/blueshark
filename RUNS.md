# Training runs & evals log

Every training run and what it taught us. Primary eval throughout is **held-out
masked validation loss** (lower = better); generation quality is qualitative.
All meaningful checkpoints are in GitHub Releases (see [MODELS.md](MODELS.md)).

## Runs (2026-06-27)

| # | Where | Config | Data | Steps | Tokens | Vocab | Val loss | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | laptop (MPS) | proof | Python stdlib | 800 | 1.9M | 4096 | ~2.6 (train) | first model — clean "Python texture" |
| 2 | laptop | proof | Python stdlib | 5000 | 1.9M | 4096 | ~1.8 (train) | **overfit** — train loss fell, output got worse. The data-bound lesson. |
| 3 | laptop | finegrained | Python stdlib | 800 | 1.9M | 4096 | ~3.2 | 16-expert variant, sanity |
| 4 | Colab T4 | proof | Nebius SWE traces, masked SFT **from scratch** | 1500 | 31M | 8192 | **3.166** @1000 | agentic behaviour emerged (reasons → `find_file "..."`) |
| 5 | Colab T4 | proof | github-code-clean (pretrain) | 5000 | 95M | 8192 | **2.413** | code-pretrained base |
| 6 | Colab T4 | proof | warm-start(#5) + Nebius SFT (masked) | 2000 | 31M | 8192 | **2.312** @1750 | **pretrain→SFT**; SFT start-loss 5.0 vs 9.0 from scratch |
| 7 | Colab T4 | proof | code_pretrain | 1500 | 95M | 8192 | **3.58** @1000 | recurrence-experiment baseline |
| 8 | Colab T4 | **recur3** (4L×3 = 12 eff.) | code_pretrain | 1500 | 95M | 8192 | **2.84** @1000 | **recurrence win — 0.74 nats better than #7 at identical params** |
| 9 | Colab T4 | deep8 (8L, 32M) | code_pretrain | ~500 | 95M | 8192 | 3.87 @500 | control; run lost to a Colab session reset |
| 10 | Colab T4 | recur3 | github-code (multi-lang) | 1200 | 67M | 8192 (code) | 3.852 @1000 | dedicated recur3, exfiltrated to the local viewer |

## Headline results

1. **pretrain → SFT** (runs 4 vs 6): warm-starting SFT from a code-pretrained base
   cut val loss **3.166 → 2.312** and the SFT start-loss **9.0 → 5.0**. The model
   walks into SFT already knowing code. SFT/RL need a pretrained base — you can't
   fine-tune what can't code yet.

2. **Recurrent depth** (runs 7 vs 8): weight-tied recursion (rerun the block stack
   K times) gave **val 2.84 vs 3.58 at IDENTICAL 17.1M params** — a 0.74-nat win,
   just from K× effective depth. Cost is ~3-5× wall-clock; it trades compute for
   quality at fixed memory, which is the right trade for constrained hardware.
   (Confound to clean up: recur3 used seq 512 vs proof's 256.) See [EXPERIMENTS.md](EXPERIMENTS.md).

3. **Data-bound, not arch-bound** (runs 1 vs 2): more steps on 1.9M tokens
   *overfit* and got worse. Quality is gated by data + training, not step count.

4. **Narrow domain = coherence**: broad (600-language) code data made a 17M model
   produce multilingual mush; narrow (Python) data is the path to a coherent small
   model (TinyStories logic). This is why the next run is Python-only.

5. **Decoding matters**: a repetition penalty (CTRL-style) killed the degenerate
   `'''` loops small/undertrained models fall into. Val loss is the honest signal;
   greedy/low-temp sampling lies.

## Evals

- **Held-out masked val loss** — the in-loop metric in every run above
  (`metrics.jsonl` per run dir).
- **`indeval`** (16 execution-graded India-context tasks) — reference solution
  100% (94/94), format-only naive ~54%, all 16 tasks discriminate. **Not yet run
  on our models** — they are too small/undertrained to score meaningfully. indeval
  is the *target* for the scaled coherence model and the verifiable-reward env for
  later RL.
- **Generation (qualitative trajectory)**: "Python texture" (run 1) → emergent
  agentic `find_file` behaviour (run 4) → structured code constructs across the
  language (run 10, with rep-penalty). Not yet coherent — that needs the scale +
  narrow-domain run in [PLAN_COHERENCE.md](PLAN_COHERENCE.md).

## What's next
The coherence run: Python-only corpus + the `coherent` config (132M total / 57M
active, recurrence 2) + pretrain→SFT + every validated lever, on a rented GPU.
Full recipe in [PLAN_COHERENCE.md](PLAN_COHERENCE.md).
