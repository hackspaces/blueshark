# Coherence Run — sizing & recipe

**Goal:** a small but genuinely **coherent Python** model — the first output we
look at should read like real Python, not multilingual mush.

**Strategy:** narrow the domain (Python only) + stack every lever we validated +
buy real GPU hours. Coherence = model capacity matched to distribution
complexity, trained on enough tokens. We shrink the distribution (Python) AND
raise capacity/tokens (rented GPU) at the same time.

## Why this works (from our own experiments, not theory)
- **Narrow data → tiny models look coherent.** TinyStories: a ~10M model writes
  coherent stories because the world is small. Our mush came from training on
  github-code-clean's 600 languages with a 17M budget.
- **Validated levers:** pretrain→SFT (val 3.17→2.31), **recurrent depth**
  (recur3 2.84 vs proof 3.58 at identical params), repetition-penalty decoding
  (killed the degenerate loops). All carry forward.

## The model: `coherent` preset (in configs.py)
- MoE + MLA, d_model 512, 8 layers × **recurrence 2 = 16 effective layers**,
  8 experts top-2 + 1 shared, d_ff 1024, ctx 1024.
- **132M total / 57M active params.** Big enough for coherent Python, small
  enough to train in hours and still run in the local viewer.

## Data (Python-only, ungated — no HF token needed)
- **Pretrain:** `codeparrot/codeparrot-clean` (Python only, ungated, ~50GB).
  Target **~2–3B tokens** (past Chinchilla-optimal ~1.1B on purpose — overtraining
  a small model on a narrow domain is exactly what buys coherence).
- **SFT:** the Nebius SWE-agent trajectories filtered to Python, masked
  (think/tool_call trainable, observations not).
- **Tokenizer:** retrain byte-level BPE, vocab 16384, on the Python corpus,
  keeping the 5 agentic tokens.

## Recipe (reuses the existing pipeline unchanged)
1. retrain tokenizer on Python  →  `data/tokenizer.json`
2. `tokenize_pack.py --text` the Python corpus  →  packed shards
3. `train.py --config coherent` pretrain ~2–3B tokens (warmup+cosine, frequent ckpt)
4. `train.py --config coherent --init <pretrain best>` SFT on Python agentic data
5. eval: held-out loss + `indeval` + sample with rep-penalty 1.3 / top-p 0.9

## GPU sizing (single GPU is enough)
| Tier | GPU | Model / tokens | Time | Rough cost* |
|---|---|---|---|---|
| **Recommended** | 1× A100 80GB | coherent (57M act) / ~2.5B | ~12–18 h | **~$25–40** |
| Faster | 1× H100 | same | ~6–9 h | ~$20–30 |
| Dirt-cheap proof | 1× RTX 4090 24GB | ~40M / ~1B | ~6–8 h | ~$5–8 |

*Spot prices on Vast.ai / RunPod, ~early-2026; **verify live before renting.**
Throughput assumption ~40–60k tok/s for a 57M-active model on A100.

## Provider setup (Vast.ai or RunPod, PyTorch image)
```
git clone -b main https://github.com/hackspaces/blueshark.git && cd blueshark
pip install torch tokenizers pyarrow numpy
# pull codeparrot-clean (Python) shards, retrain tokenizer, pack, then:
python pipeline/train.py --config coherent --data py_pretrain --out runs/coh \
  --steps <N> --batch 32 --seq 1024 --lr 3e-3 --eval-every 500 --save-every 500
```
Resume-safe (`--resume`) if the box restarts.

## Storage / safety — everything on GitHub
- **Code:** merged to `main`.
- **Checkpoints:** GitHub **Releases**, one tag per run (holds up to 2GB/asset,
  no LFS). See `MODELS.md`. Upload best.pt + tokenizer at the end (and
  periodically mid-run). **Never HuggingFace.** Kaggle as optional backup.
- **Corpora:** not stored — re-downloadable from source.
