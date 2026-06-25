# Architecture

Blueshark is a decoder-only transformer with two ideas that define the 2026 frontier recipe: Multi-head Latent Attention (MLA) for cheap long-context attention, and a fine-grained Mixture-of-Experts feed-forward with an always-on shared expert.

## Full model

```mermaid
flowchart TD
  A[token ids] --> B[Embedding]
  B --> C[Block 1]
  C --> D[Block 2 ... Block N]
  D --> E[RMSNorm]
  E --> F[LM head, weight tied to embedding]
  F --> G[logits]
```

## Block

Pre-norm, two residual paths: latent attention, then a sparse MoE feed-forward.

```mermaid
flowchart TD
  x[input] --> n1[RMSNorm]
  n1 --> attn[MLA attention]
  x --> r1(("+"))
  attn --> r1
  r1 --> n2[RMSNorm]
  n2 --> moe[MoE feed-forward]
  r1 --> r2(("+"))
  moe --> r2
  r2 --> out[output]
```

## MLA (Multi-head Latent Attention)

Queries and keys/values are compressed through a small latent before being projected back up. Position information (RoPE) rides on a separate decoupled slice, so the latent that gets cached carries no rotation. This shrinks the KV cache, which is the real memory cost at long context.

```mermaid
flowchart TD
  x[x] --> qa[q_a down-project]
  qa --> qn[q_norm]
  qn --> qb[q_b up-project]
  qb --> qsplit{split}
  qsplit --> qnope[q_nope]
  qsplit --> qrope[q_rope -> RoPE]

  x --> kva[kv_a down-project: the latent cache]
  kva --> kvn[kv_norm]
  kvn --> kvb[kv_b up-project]
  kvb --> ksplit{split}
  ksplit --> knope[k_nope]
  ksplit --> v[v]
  x --> krope[k_rope -> RoPE: shared across heads]

  qnope --> q[q = concat]
  qrope --> q
  knope --> k[k = concat]
  krope --> k

  q --> sc[scores = q k^T / sqrt of head dim]
  k --> sc
  sc --> sm[causal softmax]
  sm --> av[multiply by v]
  v --> av
  av --> o[output projection]
```

## MoE feed-forward

A router scores every token over the routed experts and keeps the top-k. Those fire sparsely. A shared expert runs on every token to hold common knowledge so the routed experts can specialize. A load-balancing auxiliary loss keeps the router from collapsing onto a few experts.

```mermaid
flowchart TD
  x[token] --> router[router]
  router --> topk[top-k select]
  topk --> e1[routed expert i]
  topk --> e2[routed expert j]
  x --> sh[shared expert: always on]
  e1 --> sum(("weighted sum"))
  e2 --> sum
  sh --> sum
  sum --> out[output]
```

Memory is set by the total number of experts, since they all have to sit in memory. Speed and training cost are set by the active experts, since only the top-k plus the shared one run per token. That split is why a large total model can train and serve like a small one.
