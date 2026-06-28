"""Data Studio — one web app to BUILD a verified India-context SFT dataset.

  Generate  : pick domains + count + teacher; the backend runs the datagen loop,
              shelling out to `claude -p` (your Max login) for the claude-code teacher,
              and keeps only trajectories that pass the real indeval rule check.
  Review    : read each kept trajectory, edit reasoning/code, re-verify the edit
              against the rule, approve/reject. Edits flip provenance to human-edited.
  Export    : write the clean SFT JSONL + a teacher-vs-human provenance breakdown.

Run:
    python -m datagen.studio.server            # then open http://127.0.0.1:7870
Working set persists to data/datagen/studio.jsonl across restarts.
"""
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from indeval.schema import Task                      # noqa: E402
from indeval.grader import grade                     # noqa: E402
from datagen.gen import GENERATORS                   # noqa: E402
from datagen.teacher import (MockTeacher, ClaudeCodeTeacher,  # noqa: E402
                             AnthropicAPITeacher, extract_code)
import random                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "7870"))
WORK = os.path.join(ROOT, "data", "datagen", "studio.jsonl")
CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
LOCK = threading.Lock()

_SYSTEM = ("You are a careful Indian software engineer. Reason step by step "
           "(read, reason, act, verify), then give a single correct Python function.")

os.makedirs(os.path.dirname(WORK), exist_ok=True)
ROWS = [json.loads(l) for l in open(WORK)] if os.path.exists(WORK) else []
GEN = {"running": False, "teacher": None, "attempted": 0, "kept": 0, "total": 0,
       "by_domain": {}, "log": [], "error": None}


def persist():
    with open(WORK, "w") as f:
        for r in ROWS:
            f.write(json.dumps(r) + "\n")


def make_teacher(kind, model):
    if kind == "claude-code":
        return ClaudeCodeTeacher(model=model or None)
    if kind == "api":
        return AnthropicAPITeacher(model=model or "claude-opus-4-8")
    from datagen._mock_solutions import SOLUTIONS
    return MockTeacher(SOLUTIONS, quality="good")


def regrade(row, content):
    m = CODE_RE.search(content)
    if not m:
        return {"solved": False, "passed": 0, "total": 0, "failures": ["no python code block"]}
    task = Task(task_id=f"{row['domain']}_review", domain=row["domain"], title="",
                prompt="", entry_point=row["entry_point"], test_program=row["test_program"],
                reference_solution="", naive_solution="")
    r = grade(task, m.group(1).strip())
    return {"solved": r.solved, "passed": r.passed, "total": r.total, "failures": r.failures}


def run_generation(domains, n, teacher, teacher_name):
    GEN.update(running=True, teacher=teacher_name, attempted=0, kept=0,
               total=len(domains) * n, by_domain={d: {"kept": 0, "attempted": 0} for d in domains},
               log=[], error=None)
    rng = random.Random()
    try:
        for d in domains:
            for _ in range(n):
                GEN["attempted"] += 1
                GEN["by_domain"][d]["attempted"] += 1
                prompt, entry_point, test_program = GENERATORS[d](rng)
                try:
                    resp = teacher(prompt, entry_point, d)
                except Exception as e:
                    GEN["log"].append(f"{d}: teacher error: {e}")
                    continue
                code = extract_code(resp)
                if not code:
                    GEN["log"].append(f"{d}: no code block in teacher output")
                    continue
                task = Task(task_id=f"{d}_train", domain=d, title="", prompt=prompt,
                            entry_point=entry_point, test_program=test_program,
                            reference_solution="", naive_solution="")
                res = grade(task, code)
                if not res.solved:
                    GEN["log"].append(f"{d}: dropped (failed rule check {res.passed}/{res.total})")
                    continue
                with LOCK:
                    ROWS.append({
                        "domain": d,
                        "messages": [{"role": "system", "content": _SYSTEM},
                                     {"role": "user", "content": prompt},
                                     {"role": "assistant", "content": resp.strip()}],
                        "verified": True, "checks": f"{res.passed}/{res.total}",
                        "source": "teacher", "status": "pending_review",
                        "entry_point": entry_point, "test_program": test_program,
                    })
                    persist()
                GEN["kept"] += 1
                GEN["by_domain"][d]["kept"] += 1
    except Exception as e:
        GEN["error"] = str(e)
    finally:
        GEN["running"] = False


def summary():
    from collections import Counter
    st, sr, dm = Counter(), Counter(), Counter()
    for r in ROWS:
        st[r.get("status", "pending_review")] += 1
        sr[r.get("source", "teacher")] += 1
        dm[r["domain"]] += 1
    return {"total": len(ROWS), "status": dict(st), "source": dict(sr), "domain": dict(dm)}


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
        elif self.path == "/state":
            self._send(200, json.dumps({
                "domains": list(GENERATORS), "gen": GEN, "summary": summary()}))
        elif self.path == "/rows":
            self._send(200, json.dumps(ROWS))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        p = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/generate":
            if GEN["running"]:
                self._send(409, json.dumps({"error": "generation already running"}))
                return
            domains = [d for d in p.get("domains", []) if d in GENERATORS] or list(GENERATORS)
            n = max(1, min(int(p.get("n", 3)), 100))
            tname = p.get("teacher", "mock")
            teacher = make_teacher(tname, p.get("model", ""))
            threading.Thread(target=run_generation, args=(domains, n, teacher, tname),
                             daemon=True).start()
            self._send(200, json.dumps({"started": True}))
        elif self.path == "/grade":
            self._send(200, json.dumps(regrade(ROWS[p["index"]], p["content"])))
        elif self.path == "/save":
            with LOCK:
                r = ROWS[p["index"]]
                new = p["content"].strip()
                if new != r["messages"][-1]["content"]:
                    r["messages"][-1]["content"] = new
                    r["source"] = "human-edited"
                r["status"] = p["status"]
                persist()
            self._send(200, json.dumps({"ok": True, "source": r["source"]}))
        elif self.path == "/export":
            keep_pending = bool(p.get("keep_pending"))
            from collections import Counter
            src, dom, kept = Counter(), Counter(), 0
            out = os.path.join(ROOT, "data", "datagen", "train.clean.jsonl")
            with open(out, "w") as f:
                for r in ROWS:
                    s = r.get("status", "pending_review")
                    if s == "rejected" or (s == "pending_review" and not keep_pending):
                        continue
                    f.write(json.dumps({"domain": r["domain"], "source": r.get("source", "teacher"),
                                        "messages": r["messages"]}) + "\n")
                    kept += 1
                    src[r.get("source", "teacher")] += 1
                    dom[r["domain"]] += 1
            human = src.get("human-edited", 0)
            self._send(200, json.dumps({"out": out, "kept": kept, "source": dict(src),
                                        "domain": dict(dom),
                                        "human_pct": round(100 * human / max(kept, 1))}))
        else:
            self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    print(f"\n  blueshark Data Studio  ->  http://127.0.0.1:{PORT}   ({len(ROWS)} rows in {WORK})\n")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
