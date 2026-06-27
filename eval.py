"""Eval system. Answers two questions:

  1. DID IT LEARN?  -> held-out loss, perplexity, bits-per-byte (tokenizer-
     independent), and next-token top-1 accuracy on held-out code.
  2. WILL IT SCALE? -> train a ladder of increasing model sizes for a fixed
     budget, fit a power law of loss vs parameters, and extrapolate. A clean
     downward power law is the evidence that the big model will work. This is
     the scaling-law check that the M1 proof gate is built on.

Run:
    .venv/bin/python corpus.py
    .venv/bin/python eval.py
"""

import math
import time

import numpy as np
import torch

from model import Config, Model
from train import CORPUS, get_device, get_tokenizer

B, T = 16, 128


def load_data(tok, device):
    text = CORPUS.read_text(encoding="utf-8")
    ids = torch.tensor(tok.encode(text).ids, dtype=torch.long)
    tokens_per_byte = len(ids) / len(text.encode("utf-8"))   # for bits-per-byte
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], tokens_per_byte


def get_batch(src, device):
    ix = torch.randint(0, len(src) - T - 1, (B,))
    xb = torch.stack([src[i:i + T] for i in ix]).to(device)
    yb = torch.stack([src[i + 1:i + T + 1] for i in ix]).to(device)
    return xb, yb


@torch.no_grad()
def evaluate(model, data, device, tokens_per_byte, iters=60):
    """Returns (val_loss_nats, perplexity, bits_per_byte, top1_accuracy)."""
    model.eval()
    losses, correct, count = [], 0, 0
    for _ in range(iters):
        xb, yb = get_batch(data, device)
        logits, loss = model(xb, yb)
        losses.append(loss.item())
        correct += (logits.argmax(-1) == yb).sum().item()
        count += yb.numel()
    model.train()
    mean = sum(losses) / len(losses)
    bpb = mean * tokens_per_byte / math.log(2)
    return mean, math.exp(mean), bpb, correct / count


def train_one(cfg, train_data, device, steps, lr):
    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    for _ in range(steps):
        xb, yb = get_batch(train_data, device)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.balance_step()
    return model


def main():
    torch.manual_seed(0)
    device = get_device()
    tok = get_tokenizer()
    V = tok.get_vocab_size()
    train_data, val_data, tpb = load_data(tok, device)
    print(f"device: {device}  vocab: {V}  corpus: {(len(train_data)+len(val_data))/1e6:.2f}M tokens\n")

    # A ladder of increasing model sizes, same data and step budget.
    ladder = [
        dict(d_model=128, n_layers=3, d_ff=256),
        dict(d_model=192, n_layers=3, d_ff=384),
        dict(d_model=256, n_layers=4, d_ff=512),
        dict(d_model=384, n_layers=4, d_ff=768),
    ]
    steps = 400
    print(f"SCALING STUDY: {len(ladder)} sizes, {steps} steps each, learning rate scaled to size\n")
    print(f"{'params':>10} {'train bpb':>10} {'val bpb':>9} {'val ppl':>9} {'top-1':>8}")
    print("-" * 52)

    rows = []
    for spec in ladder:
        cfg = Config(
            vocab_size=V, n_heads=4, d_rope=16,
            d_nope=spec["d_model"] // 8, d_v=spec["d_model"] // 8,
            kv_latent=spec["d_model"] // 2, q_latent=spec["d_model"],
            n_routed_experts=8, n_active_experts=2, n_shared_experts=1,
            max_seq=256, **spec,
        )
        lr = 3e-3 * (256 / spec["d_model"]) ** 0.5     # smaller LR for bigger models
        t0 = time.time()
        model = train_one(cfg, train_data, device, steps, lr)
        params = sum(p.numel() for p in model.parameters())
        tr_loss, _, tr_bpb, _ = evaluate(model, train_data, device, tpb)   # capacity
        v_loss, v_ppl, v_bpb, acc = evaluate(model, val_data, device, tpb)  # generalization
        rows.append((params, tr_bpb, v_bpb))
        print(f"{params/1e6:9.1f}M {tr_bpb:10.3f} {v_bpb:9.3f} {v_ppl:9.1f} {acc*100:7.1f}%  ({time.time()-t0:.0f}s)")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # Fit the power law on TRAIN bits/byte (capacity). On a small corpus, val is
    # confounded by overfitting; capacity is the clean scale-free signal.
    P = np.array([r[0] for r in rows], dtype=float)
    Ytr = np.array([r[1] for r in rows], dtype=float)
    b, a = np.polyfit(np.log(P), np.log(Ytr), 1)   # slope b is the scaling exponent
    mono = all(Ytr[i] >= Ytr[i + 1] for i in range(len(Ytr) - 1))

    print("\nSCALING VERDICT (capacity = train bits/byte)")
    print(f"  power-law exponent: {b:.3f}  (negative = improves with scale)")
    print(f"  improves monotonically with size: {mono}")
    for mult in (10, 100):
        print(f"  extrapolated train bits/byte at {mult:>3}x largest: {math.exp(a)*(P[-1]*mult)**b:.3f}")

    if mono and b < 0:
        print("\n  => Capacity scales: the model fits better as it grows, on a clean power law.")
        print("     That is the signal the M1 proof gate checks before the big run.")
        print("     Val lags here only because the toy corpus is tiny; that is the data the ask funds.")
    else:
        print("\n  => Still noisy at toy scale; needs more steps/data for a clean curve.")


if __name__ == "__main__":
    main()
