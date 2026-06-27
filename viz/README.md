# blueshark live architecture viewer

A local, zero-dependency web app that runs a **real forward pass** of the
MoE + MLA model on text you type and draws what happens inside, layer by layer.

## Run

```bash
.venv/bin/python viz/server.py        # then open http://127.0.0.1:7860
```

Type a prompt, press **Run forward pass**. No pip installs, no CDN: it uses
only the Python stdlib plus the `torch`/`tokenizers` the repo already has.

## What it shows

- **Flow column** (left): tokens -> embedding -> Block 0..N (MLA attention +
  MoE routing, with a live per-expert load mini-bar) -> next-token output.
  Click any attention or MoE cell to inspect it.
- **MLA attention map**: per-head, causal (lower-triangular) attention weights
  over the token sequence; hover any cell for the exact weight.
- **MoE routing**: which top-k of the experts each token fires to (brightness =
  gate weight), the per-expert load with the max/mean balance ratio, and a note
  that the shared expert is always on.
- **Residual stream**: hidden-state norm growing through depth (embed -> top).
- **Output**: the top-12 next-token distribution.

## Weights

On start it loads `data/viz_ckpt.pt` if present (**trained weights** badge ->
patterns are learned). Otherwise it randomly initialises the model (**random
init** badge -> the mechanism is still exact, just not learned).

To produce a quick demo checkpoint:

```bash
.venv/bin/python viz/train_ckpt.py    # ~2 min on Apple Silicon, saves data/viz_ckpt.pt
```

## Files

- `server.py` - stdlib HTTP server; loads the model, serves the page and `/infer`.
- `capture.py` - runs the forward pass and records every activation. The
  `recompute_*` helpers mirror `MLA.forward` / `MoE.forward` in `model.py`; if
  that math changes, update them so the displayed numbers stay the model's own.
- `index.html` - the single-page UI (all HTML/CSS/JS inline).
- `train_ckpt.py` - quick training run that saves a checkpoint for the viewer.
