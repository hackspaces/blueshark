"""Tokenize a corpus into a packed binary stream for streamed training.

Two modes:
  --sft   <jsonl>   read segments [[text, trainable], ...] -> ids + per-token loss mask
  --text  <txt>     read plain text -> ids, mask all 1 (pretraining)

Writes (to data/packed/<name>.*):
  <name>.ids.bin    uint16 token ids, contiguous
  <name>.mask.bin   uint8 loss mask (1 = compute loss on this token)
  <name>.meta.json  {n_tokens, vocab_size, mode, trainable_frac}

The trainer memory-maps these, so the corpus never has to fit in RAM.

Usage:
  .venv/bin/python pipeline/tokenize_pack.py --sft data/sft/swe_agentic_sft.jsonl --name swe_sft
  .venv/bin/python pipeline/tokenize_pack.py --text data/corpus.txt --name code_pretrain
"""
import argparse
import json
import os

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "packed")
TOK = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sft", help="segments JSONL from parse_swe_trajectories.py")
    g.add_argument("--text", help="plain text file (pretraining)")
    ap.add_argument("--name", required=True, help="output basename under data/packed/")
    ap.add_argument("--max-records", type=int, default=0, help="cap records (0 = all)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    ids_path = os.path.join(OUT, f"{args.name}.ids.bin")
    mask_path = os.path.join(OUT, f"{args.name}.mask.bin")

    n_tokens = trainable = 0
    ids_f = open(ids_path, "wb")
    mask_f = open(mask_path, "wb")

    def emit(token_ids, mask_val):
        nonlocal n_tokens, trainable
        if not token_ids:
            return
        arr = np.asarray(token_ids, dtype=np.uint16)
        if isinstance(mask_val, (list, np.ndarray)):
            m = np.asarray(mask_val, dtype=np.uint8)
        else:
            m = np.full(len(token_ids), mask_val, dtype=np.uint8)
        ids_f.write(arr.tobytes())
        mask_f.write(m.tobytes())
        n_tokens += len(token_ids)
        trainable += int(m.sum())

    if args.text:
        mode = "text"
        with open(args.text, encoding="utf-8") as f:
            text = f.read()
        # tokenize in chunks to keep memory bounded
        step = 1_000_000
        for i in range(0, len(text), step):
            emit(TOK.encode(text[i:i + step]).ids, 1)
    else:
        mode = "sft"
        with open(args.sft, encoding="utf-8") as f:
            for rn, line in enumerate(f):
                if args.max_records and rn >= args.max_records:
                    break
                rec = json.loads(line)
                for seg_text, trainable_flag in rec["segments"]:
                    emit(TOK.encode(seg_text).ids, 1 if trainable_flag else 0)
                if rn % 1000 == 0 and rn:
                    print(f"  ...{rn} records, {n_tokens/1e6:.1f}M tokens")

    ids_f.close()
    mask_f.close()
    meta = {"n_tokens": n_tokens, "vocab_size": TOK.get_vocab_size(), "mode": mode,
            "trainable_frac": round(trainable / max(n_tokens, 1), 4)}
    with open(os.path.join(OUT, f"{args.name}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"packed {n_tokens/1e6:.2f}M tokens ({mode}), trainable {meta['trainable_frac']*100:.0f}%")
    print(f"  {ids_path}\n  {mask_path}")


if __name__ == "__main__":
    main()
