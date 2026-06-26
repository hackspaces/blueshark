"""Build a small, good-quality training corpus from the local Python standard
library: clean, documented code plus docstring prose. Self-contained, no
download. This stands in for the real code-plus-web mix at toy scale.
"""

import glob
import os
import random
import sysconfig
from pathlib import Path

OUT = Path(__file__).parent / "data"
TARGET_BYTES = 6_000_000


def main():
    OUT.mkdir(exist_ok=True)
    stdlib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(stdlib, "**", "*.py"), recursive=True))
    random.seed(0)
    random.shuffle(files)

    chunks, total = [], 0
    for f in files:
        if "test" in f:   # include site-packages for more clean code volume
            continue
        try:
            text = Path(f).read_text(encoding="utf-8")
        except Exception:
            continue
        if not (200 < len(text) < 40_000):   # skip tiny stubs and giant files
            continue
        chunks.append(text)
        total += len(text)
        if total >= TARGET_BYTES:
            break

    corpus = "\n\n".join(chunks)
    (OUT / "corpus.txt").write_text(corpus, encoding="utf-8")
    print(f"corpus: {len(corpus)/1e6:.2f} MB from {len(chunks)} clean Python files")


if __name__ == "__main__":
    main()
