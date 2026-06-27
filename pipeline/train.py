"""Proper trainer for blueshark: streamed packed data, masked SFT loss,
warmup+cosine LR, eval, checkpoint/resume, and structured logging.

This is the reusable engine (laptop -> Colab -> rented cluster). It reads the
memory-mapped .ids/.mask bins from tokenize_pack.py, so the corpus never has to
fit in RAM, and it computes loss only where the mask is 1 (the model's own turns
in SFT; everything in pretraining).

Logging (to <out>/):
  run.log         human-readable
  metrics.jsonl   one JSON object per eval point (step, lr, train/val loss, grad_norm, tok_s, imbalance, elapsed)
  TensorBoard     if torch.utils.tensorboard is importable (optional)

Usage:
  .venv/bin/python pipeline/train.py --data swe_sft --config proof --steps 2000 \
      --batch 16 --seq 512 --out runs/sft_proof
  # resume:
  .venv/bin/python pipeline/train.py --data swe_sft --config proof --out runs/sft_proof --resume
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import Model               # noqa: E402
from configs import build_config      # noqa: E402

PACKED = os.path.join(ROOT, "data", "packed")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step, peak, warmup, total, floor_frac=0.1):
    """Linear warmup then cosine decay to floor_frac*peak."""
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if step >= total:
        return peak * floor_frac
    prog = (step - warmup) / max(total - warmup, 1)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


class Logger:
    """Console + run.log + metrics.jsonl (+ optional TensorBoard)."""

    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.out = out
        self.logf = open(os.path.join(out, "run.log"), "a")
        self.metf = open(os.path.join(out, "metrics.jsonl"), "a")
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(out, "tb"))
        except Exception:
            pass

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n"); self.logf.flush()

    def metric(self, step, d):
        rec = {"step": step, **d}
        self.metf.write(json.dumps(rec) + "\n"); self.metf.flush()
        if self.tb:
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)

    def close(self):
        self.logf.close(); self.metf.close()
        if self.tb:
            self.tb.close()


class PackedData:
    """Memory-mapped ids + loss mask, with a train/val tail split."""

    def __init__(self, name, val_frac=0.02):
        meta = json.load(open(os.path.join(PACKED, f"{name}.meta.json")))
        self.vocab = meta["vocab_size"]
        self.ids = np.memmap(os.path.join(PACKED, f"{name}.ids.bin"), dtype=np.uint16, mode="r")
        self.mask = np.memmap(os.path.join(PACKED, f"{name}.mask.bin"), dtype=np.uint8, mode="r")
        n = len(self.ids)
        self.split = int(n * (1 - val_frac))
        self.meta = meta

    def batch(self, which, B, T, device, rng):
        lo, hi = (0, self.split) if which == "train" else (self.split, len(self.ids) - T - 1)
        if hi - lo <= T + 1:                       # tiny corpus: fall back to whole range
            lo, hi = 0, len(self.ids) - T - 1
        ix = rng.integers(lo, hi, size=B)
        x = np.stack([self.ids[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([self.ids[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
        m = np.stack([self.mask[i + 1:i + 1 + T] for i in ix]).astype(np.float32)
        return (torch.from_numpy(x).to(device),
                torch.from_numpy(y).to(device),
                torch.from_numpy(m).to(device))


def masked_loss(logits, y, m):
    """Cross-entropy averaged over masked (trainable) positions only."""
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
    m = m.view(-1)
    denom = m.sum().clamp(min=1.0)
    return (ce * m).sum() / denom


@torch.no_grad()
def evaluate(model, data, B, T, device, rng, iters=20):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y, m = data.batch("val", B, T, device, rng)
        logits, _ = model(x)
        tot += masked_loss(logits, y, m).item()
    model.train()
    return tot / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="packed basename under data/packed/")
    ap.add_argument("--config", default="proof")
    ap.add_argument("--out", required=True, help="run dir for checkpoints + logs")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=0, help="warmup steps (default 5% of --steps)")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default="", help="warm-start weights from this checkpoint (e.g. a pretrained base before SFT)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = Logger(args.out)
    device = get_device()
    warmup = args.warmup or max(1, args.steps // 20)

    data = PackedData(args.data)
    cfg = build_config(args.config, data.vocab)
    if args.seq > cfg.max_seq:
        args.seq = cfg.max_seq
    model = Model(cfg).to(device)

    if args.init and os.path.exists(args.init):
        ck = torch.load(args.init, map_location=device)
        sd = ck.get("model", ck.get("state_dict", ck))
        model.load_state_dict(sd)
        log.log(f"warm-started from {args.init}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)

    start_step, best_val = 0, float("inf")
    latest = os.path.join(args.out, "latest.pt")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step, best_val = ck["step"], ck.get("best_val", float("inf"))
        log.log(f"resumed from step {start_step} (best_val {best_val:.4f})")

    total = sum(p.numel() for p in model.parameters())
    log.log(f"config={args.config} device={device} params={total/1e6:.1f}M "
            f"data={args.data} ({data.meta['mode']}, {data.meta['n_tokens']/1e6:.1f}M tok, "
            f"trainable {data.meta['trainable_frac']*100:.0f}%)")
    log.log(f"steps={args.steps} batch={args.batch} seq={args.seq} lr={args.lr} warmup={warmup}")

    rng = np.random.default_rng(0)
    model.train()
    ema = None
    t0 = time.time()
    last_t = t0

    for step in range(start_step, args.steps):
        lr = lr_at(step, args.lr, warmup, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y, m = data.batch("train", args.batch, args.seq, device, rng)
        logits, _ = model(x)
        loss = masked_loss(logits, y, m)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        model.balance_step()

        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()

        if step % args.log_every == 0:
            now = time.time()
            tok_s = args.batch * args.seq * args.log_every / max(now - last_t, 1e-6)
            last_t = now
            log.log(f"step {step:5d}/{args.steps}  loss {loss.item():.3f} (ema {ema:.3f})  "
                    f"lr {lr:.2e}  gnorm {gnorm:.2f}  {tok_s/1e3:.1f}k tok/s")
            log.metric(step, {"train_loss": round(loss.item(), 4), "ema": round(ema, 4),
                              "lr": lr, "grad_norm": round(float(gnorm), 3),
                              "tok_s": round(tok_s), "imbalance": round(model.load_imbalance(), 3),
                              "elapsed": round(now - t0, 1)})

        if step > 0 and step % args.eval_every == 0:
            val = evaluate(model, data, args.batch, args.seq, device, rng)
            log.log(f"  >>> eval step {step}: val_loss {val:.4f}  (best {best_val:.4f})")
            log.metric(step, {"val_loss": round(val, 4)})
            if val < best_val:
                best_val = val
                torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                            "best_val": best_val, "config": args.config},
                           os.path.join(args.out, "best.pt"))
                log.log(f"  saved best.pt (val {best_val:.4f})")

        if step > 0 and step % args.save_every == 0:
            torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                        "best_val": best_val, "config": args.config}, latest)

    torch.save({"step": args.steps, "model": model.state_dict(), "opt": opt.state_dict(),
                "best_val": best_val, "config": args.config}, latest)
    log.log(f"done. best_val {best_val:.4f}. checkpoints in {args.out}/")
    log.close()


if __name__ == "__main__":
    main()
