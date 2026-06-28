"""Human-review tool for the datagen output — turn raw teacher trajectories into a
human-curated dataset (provenance-tracked, re-verified after edits).

A reviewer reads each generated trajectory, edits the reasoning/code as needed,
and approves or rejects it. Edits are re-graded against the real indeval rule, so
a human can improve a trajectory but can't silently break its correctness. Every
row carries provenance (teacher vs human-edited) so the final set isn't "all Claude".

Run:
    python -m datagen.review.server path/to/train.jsonl      # then open http://127.0.0.1:7861
Saves edits back to the same file. Use datagen/export.py to emit the clean
training set + a provenance breakdown.
"""
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from indeval.schema import Task          # noqa: E402
from indeval.grader import grade         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "7861"))
PATH = sys.argv[1] if len(sys.argv) > 1 else "train.jsonl"
CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def load():
    with open(PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def save(rows):
    with open(PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


ROWS = load()


def regrade(row, assistant_content):
    """Re-run the (possibly edited) code through the real indeval rule check."""
    m = CODE_RE.search(assistant_content)
    if not m:
        return {"solved": False, "passed": 0, "total": 0, "failures": ["no python code block"]}
    task = Task(task_id=f"{row['domain']}_review", domain=row["domain"], title="",
                prompt="", entry_point=row["entry_point"],
                test_program=row["test_program"], reference_solution="", naive_solution="")
    res = grade(task, m.group(1).strip())
    return {"solved": res.solved, "passed": res.passed, "total": res.total, "failures": res.failures}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/data":
            self._send(200, json.dumps(ROWS))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        p = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/grade":
            self._send(200, json.dumps(regrade(ROWS[p["index"]], p["content"])))
        elif self.path == "/save":
            r = ROWS[p["index"]]
            new = p["content"].strip()
            if new != r["messages"][-1]["content"]:
                r["messages"][-1]["content"] = new
                r["source"] = "human-edited"
            r["status"] = p["status"]
            save(ROWS)
            self._send(200, json.dumps({"ok": True, "source": r["source"]}))
        else:
            self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    if not os.path.exists(PATH):
        raise SystemExit(f"no dataset at {PATH} — generate one with build_dataset.py first")
    print(f"\n  datagen review  ->  http://127.0.0.1:{PORT}   ({len(ROWS)} rows from {PATH})\n")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
