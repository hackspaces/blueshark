# STATUS — where we are, what's next

Last updated: 2026-06-28. Read this first when resuming (incl. from Claude Code web).

## Latest: coherence ACHIEVED + SFT diagnosis (2026-06-28)
Trained the full pretrain→SFT pipeline on a rented RTX 4090 (~$6). The `coherent`
config (132M, recurrence 2, ctx 1024, 16k Python vocab) pretrained on ~150M tokens
of codeparrot Python → **val 1.33, writes idiomatic Python (real coherence)**; then
SFT on Python SWE-agent traces → val 1.42, does the reason→tool-command loop. Both
in GitHub release **`coherent-v1`**; final model in the viewer's `coherent` preset.

**Probed the model → tweak list:**
- It learned the loop *internals* perfectly (`</think>`→`<tool_call>` P=1.0) but **not how to ENTER** the loop (issue→`<think>` P=0.0001) — entry is ~15× rarer + follows varied issue text. **FIX (done, in `parse_swe_trajectories.py`):** gate the turn behind a fixed `<plan>` marker → `{issue}<plan><think>…`, a constant→constant transition that's learnable AND gates agentic mode (kills style-bleed into plain code). Re-run SFT with the new parser to get it.
- **Inference convention:** agentic turn = prompt with `{issue}<plan>` (or `<think>`); plain code = no marker. (Verified: prompting into the turn makes the loop + tokens fire correctly on the current model.)
- **Still open (need GPU/scale):** correctness is a capability gap (greedy is confidently wrong) → needs **RL with verifiable rewards** (indeval/SWE-Gym). No Indic/multilingual (Python-only vocab) → needs **Indic data + 32-64k Indic vocab** in the next pretrain. Some memorization of the narrow corpus → more/diverse data.

## What blueshark is
Reference architecture for a sovereign agentic coding model: fine-grained MoE +
shared expert + MLA, aux-loss-free balancing, reserved agentic tokens
`<plan> <think> </think> <tool_call> <tool_result>`. Goal right now: get a small
but genuinely **coherent Python** model, then scale.

## What's done (all on `main`)
- **Live viewer** (`viz/`): zero-dep activation visualizer — MLA attention, MoE
  routing, residual stream, generation (with repetition penalty), config-preset
  switcher, agentic-token view. Run: `.venv/bin/python viz/server.py` → http://127.0.0.1:7860
- **Data pipeline** (`pipeline/`): `parse_swe_trajectories.py` (SWE traces →
  `<think>/<tool_call>/<tool_result>` segments + loss masks) → `tokenize_pack.py`
  (memmapped ids+mask shards) → `train.py` (streamed masked-SFT, warmup+cosine,
  eval, checkpoint/resume, `--init` warm-start, structured logging).
- **Config registry** (`configs.py`): proof, finegrained, recur3, deep8, coherent.
- **Colab notebook** (`colab/blueshark_train.ipynb`).

## Validated findings (the science so far)
1. **Data-bound, not arch-bound** — 17M model overfits a tiny corpus; quality
   comes from data + training, not more steps on too little.
2. **pretrain → SFT works** — code-pretrain then masked SFT: val 3.17 → 2.31.
3. **Recurrent depth wins** — `recurrence` reruns the block stack K times for K×
   effective depth at ZERO extra params. recur3 (12 eff. layers) val **2.84 vs
   proof 3.58** at identical params. See `EXPERIMENTS.md`.
4. **Narrow domain = coherence** — broad (600-lang) data made a tiny model produce
   mush; narrow (Python) data is how a small model gets coherent (TinyStories logic).
5. **Decoding matters** — repetition penalty killed the degenerate loops.

## Models (safe in GitHub Releases — see MODELS.md)
Release `tier0-v1`: `pretrain.pt`, `sft_final.pt`, `recur3_codepretrain.pt`,
`tokenizer.json`. Corpora not stored (re-downloadable). Never HuggingFace.

## NEXT ACTION — the coherence run (full recipe in PLAN_COHERENCE.md)
Renting a **RunPod RTX 4090** (~$0.69/hr, ~$7-10 total). SSH public key for the
Mac is generated (`~/.ssh/id_ed25519`); paste the `.pub` into the pod.

On the pod (PyTorch 2.8 image), run:
```
git clone https://github.com/hackspaces/blueshark.git && cd blueshark
pip install tokenizers pyarrow
# 1. pull codeparrot/codeparrot-clean (Python-only, ungated), build ~2-3B-token text
# 2. retrain BPE (vocab 16384, keep 5 special tokens) on the Python corpus
# 3. tokenize_pack.py --text  -> py_pretrain shards
# 4. train.py --config coherent --data py_pretrain ... (pretrain, frequent ckpt)
# 5. train.py --config coherent --init <pretrain best> --data <python agentic sft>
# 6. sample with rep-penalty 1.3 / top-p 0.9; eval indeval
```
Then: upload best.pt + tokenizer to a new release tag (e.g. `coherent-v1`), pull
into `data/` locally, restart the viewer, select the `coherent` preset.

`coherent` config = 132M total / 57M active, 8 layers × recurrence 2 = 16 effective,
ctx 1024.

## Open experiment threads (cheap, optional)
- Isolate recurrence from context length (rerun proof@seq512, finish deep8@1000).
- Sweep recurrence = 2 / 4.

## Branches
`main` = canonical (everything). `feat/training-pipeline` = preserved history.
