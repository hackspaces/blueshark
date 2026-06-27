"""Zero-dependency local server for the blueshark live-architecture viewer.

Loads the model + the byte-level BPE tokenizer once, then serves a single page
and a /infer endpoint that runs a real forward pass and returns every
intermediate activation. Uses only the Python stdlib (plus torch/tokenizers,
which the repo already depends on) so there is nothing extra to install.

Run:
    .venv/bin/python viz/server.py
    # open http://127.0.0.1:7860

If data/viz_ckpt.pt exists it is loaded (trained weights -> meaningful
patterns); otherwise the model is randomly initialised (still shows the full
mechanism: routing, attention shape, residual growth).
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from urllib.parse import urlparse, parse_qs       # noqa: E402

from model import Model, SPECIAL_TOKENS           # noqa: E402
from configs import PRESETS, DEFAULT, build_config  # noqa: E402
from capture import run_capture, gen_tokens       # noqa: E402

PORT = int(os.environ.get("PORT", "7860"))
MAX_TOKENS = 48
TOKJSON = os.path.join(ROOT, "data", "tokenizer.json")

print("loading tokenizer ...")
TOK = Tokenizer.from_file(TOKJSON)
VOCAB = TOK.get_vocab_size()
# the agentic control tokens, as the tokenizer actually assigns them
SPECIAL_IDS = {TOK.token_to_id(t): t for t in SPECIAL_TOKENS if TOK.token_to_id(t) is not None}

MODELS = {}  # preset_id -> {"model": Model, "meta": dict}


def load_model(pid):
    """Build (and cache) the model for a config preset, loading its checkpoint
    if one exists. Tiny models, so building on demand is instant."""
    if pid not in PRESETS:
        pid = DEFAULT
    if pid in MODELS:
        return MODELS[pid]
    cfg = build_config(pid, VOCAB)
    torch.manual_seed(0)
    model = Model(cfg).to("cpu")
    ckpt = os.path.join(ROOT, "data", PRESETS[pid]["ckpt"])
    trained = os.path.exists(ckpt)
    if trained:
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    one_expert = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
    active = total - (cfg.n_routed_experts - cfg.n_active_experts) * one_expert * cfg.n_layers
    meta = {
        "id": pid, "name": PRESETS[pid]["name"], "blurb": PRESETS[pid]["blurb"],
        "d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
        "n_routed_experts": cfg.n_routed_experts, "n_active_experts": cfg.n_active_experts,
        "n_shared_experts": cfg.n_shared_experts, "d_ff": cfg.d_ff,
        "kv_latent": cfg.kv_latent, "q_latent": cfg.q_latent,
        "vocab_size": VOCAB, "total_params": total, "active_params": active,
        "trained": trained, "special_tokens": SPECIAL_TOKENS,
        "special_ids": {t: TOK.token_to_id(t) for t in SPECIAL_TOKENS},
    }
    MODELS[pid] = {"model": model, "meta": meta}
    print(f"built '{pid}': {total/1e6:.1f}M params, trained={trained}")
    return MODELS[pid]


def configs_list():
    return [{"id": pid, "name": p["name"], "blurb": p["blurb"],
             "trained": os.path.exists(os.path.join(ROOT, "data", p["ckpt"]))}
            for pid, p in PRESETS.items()]


load_model(DEFAULT)  # warm the default so the first request is fast


def token_label(tid):
    piece = TOK.id_to_token(tid)
    if piece is None:
        return f"<{tid}>"
    # byte-level BPE marks a leading space as 'Ġ' and newline as 'Ċ'
    return piece.replace("Ġ", "·").replace("Ċ", "\\n")


def infer(text, config):
    model = load_model(config)["model"]
    enc = TOK.encode(text if text else " ")
    ids = enc.ids[:MAX_TOKENS]
    if not ids:
        ids = [TOK.token_to_id(" ") or 0]
    out = run_capture(model, ids)
    out["tokens"] = [{"id": i, "label": token_label(i), "special": i in SPECIAL_IDS}
                     for i in ids]
    for p in out["predictions"]:
        p["label"] = token_label(p["id"])
        p["special"] = p["id"] in SPECIAL_IDS
    out["truncated"] = len(enc.ids) > MAX_TOKENS
    return out


def generate(text, config, n, temp, topk):
    model = load_model(config)["model"]
    enc = TOK.encode(text if text else " ")
    ids = enc.ids[:MAX_TOKENS]
    if not ids:
        ids = [TOK.token_to_id(" ") or 0]
    n = max(1, min(int(n), 80))
    temp = max(0.05, min(float(temp), 2.0))
    topk = max(0, min(int(topk), 200))
    out_ids, steps = gen_tokens(model, ids, n, temp, topk)

    gen = []
    for j, s in enumerate(steps):
        # decoded text this token actually added (handles spaces/newlines right)
        prev = TOK.decode(out_ids[:len(ids) + j])
        cur = TOK.decode(out_ids[:len(ids) + j + 1])
        piece = cur[len(prev):] or token_label(s["id"])
        gen.append({"text": piece, "prob": s["prob"], "special": s["id"] in SPECIAL_IDS})

    return {"prompt_text": TOK.decode(ids), "gen": gen, "temp": temp, "topk": topk}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif u.path == "/configs":
            self._send(200, json.dumps(configs_list()))
        elif u.path == "/meta":
            cid = parse_qs(u.query).get("config", [DEFAULT])[0]
            self._send(200, json.dumps(load_model(cid)["meta"]))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path not in ("/infer", "/generate"):
            self._send(404, json.dumps({"error": "not found"}))
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            p = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/infer":
                result = infer(p.get("text", ""), p.get("config", DEFAULT))
            else:
                result = generate(p.get("text", ""), p.get("config", DEFAULT),
                                  p.get("n", 40), p.get("temp", 0.8), p.get("topk", 40))
            self._send(200, json.dumps(result))
        except Exception as e:  # surface errors to the UI rather than 500-ing silently
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}))


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\n  blueshark live viewer  ->  http://127.0.0.1:{PORT}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
