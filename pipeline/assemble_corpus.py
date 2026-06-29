"""Assemble the real-run pretraining corpus (~6-8B tokens) on the training box.

Streams each ingredient from HuggingFace, mixes at target token budgets, retrains
an Indic-aware byte-level BPE tokenizer on a sample of the mix, and packs the
whole thing to memmapped shards for pipeline/train.py. Designed to run on the
rented GPU box (bandwidth + disk + HF token), not the laptop.

Gated sources (The Stack) need an HF token + accepting terms on the dataset page:
    export HF_TOKEN=...            # never commit this
    huggingface-cli login --token $HF_TOKEN   # or rely on HF_TOKEN env
    pip install datasets tokenizers numpy
    python pipeline/assemble_corpus.py --name sov_pretrain

India ingredient: ships with the repo's data/india/india_text.txt (mined from the
tender DBs by extract_india_corpus.py) — copy it to the box first.

NOTE on Indic vocab: code/web/math/tender-English have NO Devanagari/Tamil script.
For a truly Indic-aware tokenizer add an Indic-script source (ai4bharat/sangraha)
to SOURCES — otherwise the vocab covers Indian *English* but not Indian *scripts*.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKED = os.path.join(ROOT, "data", "packed")
INDIA = os.path.join(ROOT, "data", "india", "india_text.txt")
HF_TOKEN = os.environ.get("HF_TOKEN")
VOCAB = 48000
SPECIAL = ["<plan>", "<think>", "</think>", "<tool_call>", "<tool_result>"]

# Each source: token budget (how many tokens to draw), HF path/config, text field.
# Tune budgets to your total; ~6.5B here. local=True reads a file instead of HF.
SOURCES = [
    {"name": "code",  "budget": 4_000_000_000, "hf": "bigcode/the-stack-dedup",
     "data_dir": "data/python", "field": "content", "gated": True},
    {"name": "web",   "budget": 1_300_000_000, "hf": "HuggingFaceFW/fineweb-edu",
     "config": "sample-10BT", "field": "text", "gated": False},
    {"name": "math",  "budget": 700_000_000, "hf": "HuggingFaceTB/finemath",
     "config": "finemath-4plus", "field": "text", "gated": False},
    {"name": "india", "budget": 400_000_000, "local": INDIA},   # ~75M tokens, repeated to budget
    # {"name":"indic","budget":1_000_000_000,"hf":"ai4bharat/sangraha","config":"verified","field":"text"},  # for Indic SCRIPT
]


def text_stream(src):
    """Yield text strings from a source (HF streaming or a local file, looping to budget)."""
    if src.get("local"):
        while True:  # loop the (smaller) local file until its budget is met
            with open(src["local"], encoding="utf-8") as f:
                for line in f:
                    yield line
            if not src.get("loop", True):
                break
        return
    from datasets import load_dataset
    kw = {"streaming": True, "split": "train"}
    if src.get("config"):
        kw["name"] = src["config"]
    if src.get("data_dir"):
        kw["data_dir"] = src["data_dir"]
    if src.get("gated"):
        kw["token"] = HF_TOKEN
    for row in load_dataset(src["hf"], **kw):
        t = row.get(src["field"])
        if t:
            yield t + "\n<|endoftext|>\n"


def train_tokenizer(out_path, sample_bytes=300_000_000):
    """Train the Indic-aware BPE on a sample drawn across all sources."""
    if os.path.exists(out_path):
        print("tokenizer exists:", out_path); return
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    sample = os.path.join(PACKED, "_tok_sample.txt")
    per = sample_bytes // len(SOURCES)
    with open(sample, "w", encoding="utf-8") as f:
        for src in SOURCES:
            got = 0
            for t in text_stream(src):
                f.write(t); got += len(t)
                if got >= per:
                    break
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train([sample], tr); tok.save(out_path)
    os.remove(sample)
    print("tokenizer trained, vocab", tok.get_vocab_size())


def pack(name, tok):
    ids_f = open(os.path.join(PACKED, f"{name}.ids.bin"), "wb")
    mask_f = open(os.path.join(PACKED, f"{name}.mask.bin"), "wb")
    total = 0
    for src in SOURCES:
        got = 0
        for t in text_stream(src):
            ids = tok.encode(t).ids
            arr = np.asarray(ids, dtype=np.uint16)
            ids_f.write(arr.tobytes())
            mask_f.write(np.ones(len(ids), dtype=np.uint8).tobytes())
            got += len(ids); total += len(ids)
            if got >= src["budget"]:
                break
        print(f"  {src['name']}: {got/1e9:.2f}B tokens", flush=True)
    ids_f.close(); mask_f.close()
    import json
    json.dump({"n_tokens": total, "vocab_size": tok.get_vocab_size(), "mode": "text"},
              open(os.path.join(PACKED, f"{name}.meta.json"), "w"))
    print(f"packed {total/1e9:.2f}B tokens -> {PACKED}/{name}.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sov_pretrain")
    args = ap.parse_args()
    os.makedirs(PACKED, exist_ok=True)
    if any(s.get("gated") for s in SOURCES) and not HF_TOKEN:
        sys.exit("gated source needs HF_TOKEN env var (export HF_TOKEN=...)")
    tok_path = os.path.join(ROOT, "data", "tokenizer.json")
    train_tokenizer(tok_path)
    from tokenizers import Tokenizer
    pack(args.name, Tokenizer.from_file(tok_path))


if __name__ == "__main__":
    main()
