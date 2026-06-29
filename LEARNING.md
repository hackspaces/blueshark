# How blueshark learns — the math, the gauges, the decisions

The point of this doc: stop guessing at tweaks. Every architecture/training
decision maps to a *measurable quantity* in our runs. Read the gauge, then decide.

## The one law
The model does **compression by prediction** — it minimizes cross-entropy:

> L = − ⟨ log P(next token | context) ⟩   (nats)

`ln(vocab)` is the know-nothing baseline. Every nat of loss it sheds = structure
it captured and can now predict. **Lower loss = more knowledge in the weights.**
It descends L by gradient descent: `θ ← θ − η·∇L` — a ball rolling downhill on a
billion-dimensional loss surface. Every other knob just keeps that ball rolling
fast without flying off.

## The three laws and their gauges

### Law 1 — the descent must stay stable → gauge: gradient norm
`‖∇L‖` (logged as `grad_norm`) is how hard we're pushing. Steady (~0.5–1.5) = the
surface is smooth where we are. Frequent spikes / large max = hitting cliffs =
instability. **Normalization (RMSNorm, QK-norm), LR, and the warmup→cosine schedule
all exist to condition the surface** so we can take bigger steps safely.
- **Decides:** QK-norm, LR, warmup, gradient clipping.
- **Read it:** `diagnose.py` → "law 1". A tweak *helps* if it smooths gnorm and/or
  lets loss drop faster at the same LR.

### Law 2 — loss is a power law in compute → gauge: the scaling slope
The deep result: `L(C) ≈ E + A·C^(−α)` where compute `C ≈ 6·N·D` (active params ×
tokens). On a log-log plot, loss falls in a straight line. An architecture's **floor
E** and **exponent α** are roughly fixed properties — so a config that's lower &
steeper when small *stays* better when big. **This is why the bake-off transfers**,
and why "deep8 beat recur3 at matched compute" is a real signal, not noise.
- **Decides:** depth vs width vs experts vs recurrence; how far scaling will get us.
- **Measure it:** `scaling.py` trains a config at 2–3 sizes and fits E, α. Compare
  configs by their curves; scale only the winner. `bakeoff.py` is the single-size
  version (matched compute = fair ranking).

### Law 3 — capacity must match the data → gauge: the generalization gap
A model has finite capacity (~params). When capacity ≫ the information in the data,
it memorizes noise → **overfitting**, seen as **val loss rising while train loss
falls** (gap = val − train, logged as `gen_gap`). Coherence = capacity matched to
the data's complexity — which is why *narrowing* to Python made a small model
coherent, and why 16M overfit 1.9M tokens.
- **Decides:** more data vs bigger model vs narrower domain; when to stop training.
- **Read it:** `diagnose.py` → "law 3". Widening gap = overfit (more/cleaner data or
  smaller model). Large flat gap = data-bound (more data helps).

### (MoE health) — expert load balance
`imbalance` = max/mean expert load (1.0 = perfect). The aux-loss-free balancer keeps
it near 1; runaway imbalance = experts collapsing. `diagnose.py` reports it.

## The decision table
| Decision | Gauge | Tool |
|---|---|---|
| QK-norm / LR / normalization | grad-norm stability + loss smoothness | `diagnose.py` |
| depth / width / experts / recurrence | scaling slope (E, α) | `scaling.py`, `bakeoff.py` |
| data vs model size vs domain | generalization gap | `diagnose.py` |
| MoE balancer | expert load max/mean | `diagnose.py` |

## What we already learned this way
- **recurrence**: looked great at equal *steps* (2.84 vs 3.58) but **lost at equal
  compute** (4.10, worst) — law 2 corrected the hunch. Dropped from the real config.
- **deep8 (real depth)** won the matched-compute bake-off → `sov300` uses real depth.
- **data-bound**: 16M on 1.9M tokens overfit (law 3) → narrow domain + more data.

## How to use it
1. Run any training → it logs loss, val, `gen_gap`, `grad_norm`, `imbalance`.
2. `python pipeline/diagnose.py <run_dir>` → reads the gauges, gives a verdict.
3. For arch choices, `pipeline/bakeoff.py` (rank at matched compute) then
   `pipeline/scaling.py` (fit the slope) → pick by the numbers.
Every tweak gets a number, not a vibe.
