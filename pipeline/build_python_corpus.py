"""Build a narrow, clean PYTHON-only pretraining corpus + tokenizer + packed
shards for the coherence run (PLAN_COHERENCE.md).

Source: codeparrot/codeparrot-clean (deduplicated GitHub Python, ungated, json.gz).
Narrowing the domain to one language is the lever that lets a small model become
coherent (see RUNS.md / STATUS.md).

  python pipeline/build_python_corpus.py [n_files] [cap_bytes]

Writes: data/py_corpus.txt, data/tokenizer.json (BPE vocab 16384, 5 agentic
tokens), data/packed/py_pretrain.{ids,mask}.bin
"""
import gzip
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000_000
SPECIAL = ["<plan>", "<think>", "</think>", "<tool_call>", "<tool_result>"]
BASE = "https://huggingface.co/datasets/codeparrot/codeparrot-clean/resolve/main/"
CORPUS = "data/py_corpus.txt"


def build_corpus():
    if os.path.exists(CORPUS) and os.path.getsize(CORPUS) >= CAP * 0.9:
        print("corpus already built:", round(os.path.getsize(CORPUS) / 1e6, 1), "MB", flush=True)
        return
    os.makedirs("data/raw", exist_ok=True)
    total = 0
    with open(CORPUS, "w", encoding="utf-8") as out:
        for i in range(1, N + 1):
            fn = f"file-{i:012d}.json.gz"
            dst = "data/raw/" + fn
            if not os.path.exists(dst):
                print("downloading", fn, flush=True)
                urllib.request.urlretrieve(BASE + fn, dst)
            with gzip.open(dst, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        c = json.loads(line).get("content") or ""
                    except Exception:
                        continue
                    if c:
                        out.write(c + "\n<|endoftext|>\n")
                        total += len(c)
                    if total >= CAP:
                        break
            print(f"  after {fn}: {total/1e6:.0f}MB", flush=True)
            if total >= CAP:
                break
    print("corpus:", round(os.path.getsize(CORPUS) / 1e6, 1), "MB", flush=True)


def build_tokenizer():
    if os.path.exists("data/tokenizer.json"):
        print("tokenizer exists", flush=True)
        return
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=16384, special_tokens=SPECIAL,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    tok.train([CORPUS], tr)
    tok.save("data/tokenizer.json")
    print("tokenizer vocab", tok.get_vocab_size(), flush=True)


if __name__ == "__main__":
    build_corpus()
    build_tokenizer()
    out = subprocess.run(["python", "pipeline/tokenize_pack.py", "--text", CORPUS,
                          "--name", "py_pretrain"], capture_output=True, text=True)
    print(out.stdout, out.stderr, flush=True)
    print("PREP_DONE", flush=True)
