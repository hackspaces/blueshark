"""Train a small checkpoint for the visualizer so attention maps, expert
routing, and next-token predictions reflect learned structure instead of noise.

Trains whichever preset from viz/configs.py you name (default: proof) and saves
to that preset's checkpoint path. Quick demo-grade run, just enough that the
patterns in the viewer become legible.

    .venv/bin/python viz/train_ckpt.py                     # trains "proof", default steps
    .venv/bin/python viz/train_ckpt.py finegrained         # another preset
    .venv/bin/python viz/train_ckpt.py proof 5000          # preset + step count
"""
import os, sys, time
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import Model
from configs import PRESETS, DEFAULT, build_config

DATA = os.path.join(ROOT, "data")
TOK = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
V = TOK.get_vocab_size()

def device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def main():
    pid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if pid not in PRESETS:
        sys.exit(f"unknown preset '{pid}'. options: {', '.join(PRESETS)}")
    torch.manual_seed(0)
    dev = device(); print(f"preset: {pid}  device: {dev}")
    cfg = build_config(pid, V)
    model = Model(cfg).to(dev)

    text = open(os.path.join(DATA, "corpus.txt"), encoding="utf-8").read()
    data = torch.tensor(TOK.encode(text).ids, dtype=torch.long)
    print(f"corpus {len(data)/1e6:.2f}M tokens")
    B, T = 24, 128
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 800

    def batch():
        ix = torch.randint(0, len(data)-T-1, (B,))
        xb = torch.stack([data[i:i+T] for i in ix]).to(dev)
        yb = torch.stack([data[i+1:i+T+1] for i in ix]).to(dev)
        return xb, yb

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9,0.95), weight_decay=0.1)
    t0=time.time()
    for s in range(1, steps+1):
        xb,yb = batch()
        _, loss = model(xb, yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); model.balance_step()
        if s==1 or s%100==0:
            print(f"step {s:4d}  loss {loss.item():.3f}  imbalance {model.load_imbalance():.2f}  ({time.time()-t0:.0f}s)")

    out = os.path.join(DATA, PRESETS[pid]["ckpt"])
    torch.save({"state_dict": model.cpu().state_dict()}, out)
    print("saved", out)

if __name__ == "__main__":
    main()
