# Model registry

Every meaningful checkpoint lives in a **GitHub Release** on this repo (big
binaries, up to 2GB/asset, no LFS, never HuggingFace). Corpora are not stored —
they re-download from source. The local `data/` dir is gitignored working space.

## Release `tier0-v1`
First end-to-end trained blueshark models. Proof config = 17.1M params unless noted.

| Asset | Trained on | Val loss | Notes |
|---|---|---|---|
| `pretrain.pt` | ~95M tok github-code-clean | 2.413 | code-pretrained base |
| `sft_final.pt` | warm-start + 31M tok Nebius SWE traces (masked) | 2.312 | pretrain→SFT, agentic tokenizer |
| `recur3_codepretrain.pt` | ~67M tok code, **recurrence 3** (12 eff. layers) | 3.852 | the recurrent-depth win (2.84 vs 3.58 vs proof on the controlled run) |
| `tokenizer.json` | — | — | byte-level BPE, vocab 8192, 5 agentic tokens |

Each `.pt` is `{state_dict, config, step, val_loss}`. To load: `build_config(config, vocab)` then `load_state_dict(state_dict)`.

## Not preserved (toy / superseded)
- The first stdlib runs (800-step, 5000-step, vocab 4096) were laptop scratch and
  overwritten — intentionally not kept (they were the "Python texture" toys).

## Coming: the coherence run (see PLAN_COHERENCE.md)
- New release tag (e.g. `coherent-v1`) with the `coherent` preset checkpoints
  (132M total / 57M active, Python-only). Upload best.pt + tokenizer.

## How to bring any model into the local viewer
1. download the asset + its tokenizer from the release
2. drop into `data/` as the matching preset's ckpt (`configs.py` -> `ckpt` field) + `data/tokenizer.json`
3. restart `viz/server.py`, pick the preset in the dropdown
