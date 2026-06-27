"""Activation capture for the live visualizer.

Runs a real forward pass and records every intermediate we want to draw:
the MLA attention probabilities per head, the MoE routing (affinity, chosen
experts, gate weights, per-expert load), the residual-stream norm at each
checkpoint, and the final next-token distribution.

The two recompute_* helpers mirror MLA.forward and MoE.forward in model.py.
If that math changes, update these to match (they are intentionally a literal
re-derivation so the displayed numbers are the model's own, not an approximation).
"""

import torch

from model import apply_rope


@torch.no_grad()
def recompute_attention(mla, x, cos, sin):
    """Return attention probabilities, shape (H, T, T). Mirrors MLA.forward."""
    cfg, H = mla.cfg, mla.cfg.n_heads
    B, T, _ = x.shape

    q = mla.q_b(mla.q_norm(mla.q_a(x))).view(B, T, H, cfg.qk_head_dim).transpose(1, 2)
    q_nope, q_rope = q.split([cfg.d_nope, cfg.d_rope], dim=-1)

    kv = mla.kv_b(mla.kv_norm(mla.kv_a(x))).view(B, T, H, cfg.d_nope + cfg.d_v).transpose(1, 2)
    k_nope, _ = kv.split([cfg.d_nope, cfg.d_v], dim=-1)
    k_rope = mla.k_rope(x).view(B, T, 1, cfg.d_rope).transpose(1, 2)

    q_rope = apply_rope(q_rope, cos, sin)
    k_rope = apply_rope(k_rope, cos, sin).expand(B, H, T, cfg.d_rope)

    q = torch.cat([q_nope, q_rope], dim=-1)
    k = torch.cat([k_nope, k_rope], dim=-1)

    scores = (q @ k.transpose(-2, -1)) / (cfg.qk_head_dim ** 0.5)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(-1)[0]  # (H, T, T)


@torch.no_grad()
def recompute_moe(moe, x):
    """Return (affinity, topi, gate, counts). Mirrors MoE.forward routing."""
    cfg = moe.cfg
    _, _, D = x.shape
    xf = x.reshape(-1, D)                                   # (T, D)

    affinity = moe.router(xf).sigmoid()                    # (T, E)
    _, topi = (affinity + moe.expert_bias).topk(cfg.n_active_experts, dim=-1)
    gate = affinity.gather(-1, topi)
    gate = gate / (gate.sum(-1, keepdim=True) + 1e-9)

    counts = torch.zeros(cfg.n_routed_experts)
    counts.scatter_add_(0, topi.reshape(-1).cpu(),
                        torch.ones(topi.numel()))
    return affinity, topi, gate, counts


@torch.no_grad()
def gen_tokens(model, ids, n, temp, topk, rep_penalty=1.3, rep_window=128):
    """Autoregressively sample n tokens. Returns (full_ids, steps) where each
    step is {id, prob} for the token that was chosen. A repetition penalty
    divides the logits of recently-emitted tokens (CTRL-style), which stops the
    degenerate loops small/undertrained models fall into."""
    model.eval()
    cfg = model.cfg
    out = list(ids)
    steps = []
    for _ in range(n):
        x = torch.tensor([out[-cfg.max_seq:]], dtype=torch.long)
        logits, _ = model(x)
        logits = logits[0, -1].clone()
        if rep_penalty and rep_penalty != 1.0:
            for tid in set(out[-rep_window:]):
                logits[tid] /= rep_penalty if logits[tid] > 0 else 1.0
                if logits[tid] <= 0:
                    logits[tid] *= rep_penalty
        logits = logits / max(temp, 1e-5)
        probs = logits.softmax(-1)
        if topk and topk > 0:
            v, i = probs.topk(min(topk, probs.numel()))
            pick = i[torch.multinomial(v / v.sum(), 1)].item()
        else:
            pick = torch.multinomial(probs, 1).item()
        steps.append({"id": int(pick), "prob": round(float(probs[pick]), 4)})
        out.append(int(pick))
    return out, steps


@torch.no_grad()
def run_capture(model, ids):
    """Run the model on one token sequence, returning a JSON-able dict of every
    intermediate the visualizer draws."""
    model.eval()
    cfg = model.cfg
    idx = torch.tensor([ids], dtype=torch.long)
    cos, sin = model.cos, model.sin

    x = model.embed(idx)                                   # (1, T, D)
    embed_norm = x[0].norm(dim=-1).tolist()

    layers = []
    for li, blk in enumerate(model.blocks):
        nx = blk.attn_norm(x)
        attn_probs = recompute_attention(blk.attn, nx, cos, sin)   # (H, T, T)
        x = x + blk.attn(nx, cos, sin)
        after_attn = x[0].norm(dim=-1).tolist()

        nf = blk.ffn_norm(x)
        affinity, topi, gate, counts = recompute_moe(blk.moe, nf)
        x = x + blk.moe(nf)
        after_moe = x[0].norm(dim=-1).tolist()

        layers.append({
            "layer": li,
            "attn": [[[round(v, 4) for v in row] for row in head]
                     for head in attn_probs.tolist()],          # (H, T, T)
            "affinity": [[round(v, 4) for v in row] for row in affinity.tolist()],
            "topi": topi.tolist(),                              # (T, k)
            "gate": [[round(v, 4) for v in row] for row in gate.tolist()],
            "counts": [int(c) for c in counts.tolist()],
            "after_attn_norm": [round(v, 3) for v in after_attn],
            "after_moe_norm": [round(v, 3) for v in after_moe],
        })

    logits = model.lm_head(model.norm(x))[0]               # (T, V)
    last = logits[-1]
    probs = last.softmax(-1)
    topp, topix = probs.topk(12)
    predictions = [{"id": int(i), "prob": round(float(p), 4)}
                   for p, i in zip(topp.tolist(), topix.tolist())]

    return {
        "n_tokens": len(ids),
        "embed_norm": [round(v, 3) for v in embed_norm],
        "layers": layers,
        "predictions": predictions,
    }
