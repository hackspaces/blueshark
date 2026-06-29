"""Measure an architecture's SCALING SLOPE — the number that predicts whether it
wins at scale, not just at one size (see LEARNING.md, law 2).

Trains a config at 2-3 sizes on a fixed token budget, records (compute, val_loss),
and fits the power law  L = E + A * C^(-alpha)  (C = 6 * active_params * tokens).
A lower irreducible floor E and steeper alpha = better at scale. Compare configs
by their fitted curves, then scale only the winner.

Runs on the GPU box:
  python pipeline/scaling.py --data sov_pretrain --base sov300 --scales 0.35 0.6 1.0 --tokens 120_000_000
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import Config, Model                                   # noqa: E402
from configs import build_config                                  # noqa: E402
from pipeline.train import lr_at, masked_loss, PackedData, evaluate, get_device  # noqa: E402


def scaled_config(base_cfg, s):
    """Scale the size knobs of a Config by factor s (width, depth, ffn, latents)."""
    rnd = lambda x: max(8, int(round(x / 8) * 8))
    return Config(
        vocab_size=base_cfg.vocab_size,
        d_model=rnd(base_cfg.d_model * s),
        n_layers=max(2, round(base_cfg.n_layers * s)),
        n_heads=base_cfg.n_heads,
        d_nope=base_cfg.d_nope, d_rope=base_cfg.d_rope, d_v=base_cfg.d_v,
        kv_latent=rnd(base_cfg.kv_latent * s), q_latent=rnd(base_cfg.q_latent * s),
        n_routed_experts=base_cfg.n_routed_experts, n_active_experts=base_cfg.n_active_experts,
        n_shared_experts=base_cfg.n_shared_experts, d_ff=rnd(base_cfg.d_ff * s),
        max_seq=base_cfg.max_seq, recurrence=base_cfg.recurrence,
    )


def active_params(model, cfg):
    tot = sum(p.numel() for p in model.parameters())
    one = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
    return tot - (cfg.n_routed_experts - cfg.n_active_experts) * one * cfg.n_layers


def train_one(cfg, data, tokens, batch, seq, lr, device):
    steps = max(50, tokens // (batch * seq))
    warmup = max(1, steps // 20)
    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(0)
    best = float("inf")
    model.train()
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, lr, warmup, steps)
        x, y, m = data.batch("train", batch, min(seq, cfg.max_seq), device, rng)
        logits, _ = model(x)
        loss = masked_loss(logits, y, m)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); model.balance_step()
        if step and step % max(50, steps // 5) == 0:
            best = min(best, evaluate(model, data, batch, min(seq, cfg.max_seq), device, rng, iters=10))
    best = min(best, evaluate(model, data, batch, min(seq, cfg.max_seq), device, rng, iters=20))
    return active_params(model, cfg), best


def fit_powerlaw(C, L):
    """Fit L = E + A*C^(-alpha) by sweeping the floor E and log-log linear fit."""
    C, L = np.array(C, float), np.array(L, float)
    best = None
    for E in np.linspace(0, L.min() * 0.98, 60):
        y = np.log(np.clip(L - E, 1e-6, None)); x = np.log(C)
        a, b = np.polyfit(x, y, 1)            # y = b + a*x  -> alpha=-a, A=exp(b)
        resid = float(np.sum((y - (b + a * x)) ** 2))
        if best is None or resid < best[0]:
            best = (resid, E, -a, math.exp(b))
    _, E, alpha, A = best
    return E, A, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="sov300")
    ap.add_argument("--scales", nargs="+", type=float, default=[0.35, 0.6, 1.0])
    ap.add_argument("--tokens", type=int, default=120_000_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    args = ap.parse_args()

    dev = get_device()
    data = PackedData(args.data)
    base = build_config(args.base, data.vocab)

    pts = []
    for s in args.scales:
        cfg = scaled_config(base, s)
        t0 = time.time()
        N, L = train_one(cfg, data, args.tokens, args.batch, args.seq, args.lr, dev)
        C = 6 * N * args.tokens
        pts.append((s, N, C, L))
        print(f"  scale {s:>4}: active {N/1e6:6.1f}M  C {C:.2e}  val {L:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    E, A, alpha = fit_powerlaw([c for *_, c, _ in pts], [l for *_, l in pts])
    print(f"\n  fit: L = {E:.3f} + {A:.2e} * C^(-{alpha:.3f})")
    print(f"  irreducible floor E={E:.3f}   scaling exponent alpha={alpha:.3f}")
    print("  (lower E and higher alpha = this arch keeps improving with scale)")
    # extrapolate to a 10x-compute run
    C10 = pts[-1][2] * 10
    print(f"  extrapolated val at 10x compute ({C10:.1e}): {E + A * C10 ** (-alpha):.3f}")


if __name__ == "__main__":
    main()
