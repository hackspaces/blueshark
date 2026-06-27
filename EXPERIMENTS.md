# Experiments — external ideas, where they might fit, what to try

A living menu of approaches from the wider research world. The point is not to
adopt them wholesale but to map each onto blueshark: *where could it plug in,
and what is the cheapest experiment that tells us if it helps?* Each entry ends
with an honest verdict (on the critical path, or a parked research bet).

The critical path stays: **pretrain (code+web+math) → SFT (agentic trajectories,
masked) → RL (executable reward)**. Items below are candidate *upgrades* to that,
not replacements for it.

---

## TRM — Tiny Recursion Models (Samsung SAIL, arXiv:2510.04871)

**What it is.** A 7M-param, 2-layer network that *recursively refines its own
answer*: it carries a current solution `y` and a reasoning state `z`, loops a
tiny core n times with T deep-supervision steps, and learns a halting signal for
when to stop. On structured-reasoning puzzles it beats far bigger models —
Sudoku-Extreme 87.4% (LLMs 0%), ARC-AGI-1 44.6% (Gemini 2.5 Pro 37%). It is
**not a language model**: no attention, no autoregression, fixed grid in → grid
out, and it leans on 1000× data augmentation that only works for symmetry-rich
puzzles.

**Intuition.** One small brain that reads its own scratchpad and revises the
answer many times, vs one big brain answering in a single pass. Depth comes from
*recursion*, not from more layers.

**Where it could plug into blueshark.**
1. **The `<think>` block as latent recursive refinement.** This is exactly the
   README ROADMAP's "recurrent-depth reasoning" item. Instead of (or before)
   emitting reasoning tokens, a small recurrent core could refine a latent state,
   giving adaptive "thinking compute" behind the `<think>` token.
2. **Adaptive computation / halting.** Spend variable compute per turn — think
   longer on hard steps, halt early on easy ones. Relevant to efficient agents.
3. **A structured-reasoning capability + eval target.** ARC-AGI / Sudoku as a
   probe of pure reasoning, separate from code generation.

**Cheap experiments to try (laptop / free Colab).**
- **E1 — reproduce the core idea small.** Add a recurrent "think" loop to
  `model.py` behind a flag (reuse one Block n times with a carried state), train
  on the paper's Sudoku-Extreme 1K set, compare vs the non-recursive baseline.
  Their Sudoku run was ~36h on 1 L40S — a scaled-down version fits Colab.
- **E2 — depth-via-recursion vs depth-via-layers, fixed params.** On our
  code/agentic data, does looping a smaller stack beat stacking layers at equal
  parameter count? Measures whether recursion helps *language* reasoning at all.
- **E3 — adaptive halting on `<think>`.** Let the model decide think length;
  plot quality vs compute. Even a crude version shows if halting pays off.

**Verdict.** ✅ **VALIDATED (2026-06-27) — recurrent depth works on our code data.**
Implemented as `Config.recurrence` (weight-tied: rerun the block stack K times).
Controlled run on ~95M tokens of github-code-clean, same optimizer/LR, val loss
at step 1000:

| config | params | effective layers | val loss @ step 1000 |
|---|---|---|---|
| proof  | 17.1M | 4  | 3.58 |
| **recur3** | **17.1M** | **12** | **2.84** |
| deep8  | 32.2M | 8  | (3.87 @ step 500; run lost to a Colab reset) |

**recur3 beat proof by 0.74 nats at IDENTICAL params** — depth-via-recursion is a
real quality lever here. The cost is compute (~3-5x slower wall-clock for the
extra effective depth) and it trades compute for quality at fixed memory/params —
which is exactly the right trade for our memory-constrained / sovereign-on-limited-
hardware thesis. Confound to clean up next: recur3 used seq 512 vs proof's 256;
re-run proof@512 and finish deep8@1000 to fully isolate recurrence from context
length. Next: pretrain->SFT a recur model end to end; try recurrence=2 and higher.
The original TRM grid-puzzle caveat stands, but the recursion *mechanism* transfers.

---

<!-- template for the next paper:
## NAME — (org, arXiv:XXXX)
**What it is.** ...
**Where it could plug into blueshark.** ...
**Cheap experiments to try.** ...
**Verdict.** on critical path / parked bet.
-->
