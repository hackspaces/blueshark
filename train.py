"""Small training recipe.

Trains our own byte-level BPE tokenizer on the corpus, then trains a sub-1B
MoE + MLA model (the architecture from model.py) on the Mac GPU (MPS) and
samples from it. The point is to prove the whole loop works end to end on a
small, good-quality corpus with a tokenizer we trained ourselves.

Run:
    .venv/bin/python corpus.py     # build data/corpus.txt
    .venv/bin/python train.py
"""

import time
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from model import Config, Model, SPECIAL_TOKENS

DATA = Path(__file__).parent / "data"
CORPUS = DATA / "corpus.txt"
TOKJSON = DATA / "tokenizer.json"
VOCAB = 4096


def get_tokenizer():
    if TOKJSON.exists():
        return Tokenizer.from_file(str(TOKJSON))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train([str(CORPUS)], trainer)
    tok.save(str(TOKJSON))
    return tok


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def sample(model, tok, device, prompt="def ", n=200, temp=0.8, topk=40):
    model.eval()
    x = torch.tensor([tok.encode(prompt).ids], device=device)
    for _ in range(n):
        logits, _ = model(x[:, -model.cfg.max_seq:])
        logits = logits[0, -1] / temp
        v, i = logits.topk(topk)
        nxt = i[torch.multinomial(torch.softmax(v, -1), 1)]
        x = torch.cat([x, nxt.view(1, 1)], dim=1)
    model.train()
    return tok.decode(x[0].tolist())


def main():
    torch.manual_seed(0)
    device = get_device()
    print(f"device: {device}")

    tok = get_tokenizer()
    V = tok.get_vocab_size()
    print(f"tokenizer: byte-level BPE we trained, vocab {V}")

    data = torch.tensor(tok.encode(CORPUS.read_text(encoding="utf-8")).ids, dtype=torch.long)
    print(f"corpus: {len(data)/1e6:.2f}M tokens")
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = Config(
        vocab_size=V, d_model=256, n_layers=4, n_heads=4,
        d_nope=32, d_rope=16, d_v=32, kv_latent=128, q_latent=256,
        n_routed_experts=8, n_active_experts=2, n_shared_experts=1,
        d_ff=512, max_seq=256,
    )
    model = Model(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"model: {total/1e6:.1f}M params, MoE(8 experts, top-2, +1 shared) + MLA, {cfg.n_layers} layers\n")

    B, T = 16, 128

    def batch(src):
        ix = torch.randint(0, len(src) - T - 1, (B,))
        xb = torch.stack([src[i:i + T] for i in ix]).to(device)
        yb = torch.stack([src[i + 1:i + T + 1] for i in ix]).to(device)
        return xb, yb

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    steps = 1000
    t0 = time.time()
    for s in range(1, steps + 1):
        xb, yb = batch(train_data)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.balance_step()                      # aux-loss-free balancer
        if s == 1 or s % 100 == 0:
            with torch.no_grad():
                vx, vy = batch(val_data)
                _, vl = model(vx, vy)
            print(f"step {s:4d}  train {loss.item():.3f}  val {vl.item():.3f}  ({time.time()-t0:.0f}s)")

    print(f"\nexpert imbalance max/mean: {model.load_imbalance():.2f}")
    print("\n--- sample (prompt: 'def ') ---")
    print(sample(model, tok, device, prompt="def "))


if __name__ == "__main__":
    main()
