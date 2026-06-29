"""Architecture config presets for the live viewer.

Shared by viz/server.py and viz/train_ckpt.py so a checkpoint trained for a
preset always matches the model the viewer rebuilds for it. Add a preset here
and it shows up in the dropdown automatically.
"""
from model import Config

PRESETS = {
    "proof": {
        "name": "Proof · 8 experts, top-2",
        "blurb": "8 routed experts, top-2 + 1 shared, d_ff 512 — the baseline.",
        "ckpt": "viz_ckpt.pt",
        "kwargs": dict(
            d_model=256, n_layers=4, n_heads=4,
            d_nope=32, d_rope=16, d_v=32, kv_latent=128, q_latent=256,
            n_routed_experts=8, n_active_experts=2, n_shared_experts=1,
            d_ff=512, max_seq=256,
        ),
    },
    "finegrained": {
        "name": "Fine-grained · 16 experts, top-4",
        "blurb": "16 finer experts (d_ff 256), top-4 + 1 shared — same active FLOPs and capacity class, finer specialization.",
        "ckpt": "viz_ckpt_finegrained.pt",
        "kwargs": dict(
            d_model=256, n_layers=4, n_heads=4,
            d_nope=32, d_rope=16, d_v=32, kv_latent=128, q_latent=256,
            n_routed_experts=16, n_active_experts=4, n_shared_experts=1,
            d_ff=256, max_seq=256,
        ),
    },
    # --- the coherence run: narrow domain (Python) + every validated lever ---
    "coherent": {
        "name": "Coherent · ~100M MoE+MLA, recurrence 2",
        "blurb": "scaled for genuine coherence on a NARROW (Python-only) corpus: d=512, 8 layers x recurrence 2 = 16 effective, 8 experts top-2, seq 1024. The rented-GPU run.",
        "ckpt": "viz_ckpt_coherent.pt",
        "kwargs": dict(
            d_model=512, n_layers=8, n_heads=8,
            d_nope=64, d_rope=32, d_v=64, kv_latent=256, q_latent=512,
            n_routed_experts=8, n_active_experts=2, n_shared_experts=1,
            d_ff=1024, max_seq=1024, recurrence=2,
        ),
    },

    # --- the real run: ~540M total / ~256M active, REAL DEPTH (bake-off winner:
    #     deep layers beat recurrence at matched compute), no recurrence ---
    "sov300": {
        "name": "Sovereign-300 · 540M total / 256M active, 20 real layers",
        "blurb": "the real-run config: d768, 20 layers (real depth, recurrence dropped per the bake-off), 8 experts top-2 + shared, ctx 1024. Trains on ~6-8B tokens of code+web+math+India on one A100/H100.",
        "ckpt": "viz_ckpt_sov300.pt",
        "kwargs": dict(
            d_model=768, n_layers=20, n_heads=12,
            d_nope=96, d_rope=32, d_v=96, kv_latent=384, q_latent=768,
            n_routed_experts=8, n_active_experts=2, n_shared_experts=1,
            d_ff=1024, max_seq=1024, recurrence=1,
            grad_checkpoint=True,   # 540M won't fit otherwise; same math, less memory
        ),
    },

    # --- experiment presets (aggressive arch bets) ---
    "proof_qk": {
        "name": "Proof + QK-norm (screening variant)",
        "blurb": "the proof baseline with QK-norm on — bake off vs proof to measure if QK-norm lowers loss / smooths gnorm before adopting it.",
        "ckpt": "viz_ckpt_proof_qk.pt",
        "kwargs": dict(
            d_model=256, n_layers=4, n_heads=4,
            d_nope=32, d_rope=16, d_v=32, kv_latent=128, q_latent=256,
            n_routed_experts=8, n_active_experts=2, n_shared_experts=1,
            d_ff=512, max_seq=256, qk_norm=True,
        ),
    },
    "recur3": {
        "name": "Recurrent-depth · 4 layers looped 3x",
        "blurb": "weight-tied recursion: 4 layers run 3x = 12 effective layers at the SAME params, seq 512. TRM-style depth-via-recursion bet.",
        "ckpt": "viz_ckpt_recur3.pt",
        "kwargs": dict(
            d_model=256, n_layers=4, n_heads=4,
            d_nope=32, d_rope=16, d_v=32, kv_latent=128, q_latent=256,
            n_routed_experts=8, n_active_experts=2, n_shared_experts=1,
            d_ff=512, max_seq=512, recurrence=3,
        ),
    },
    "deep8": {
        "name": "Deep · 8 real layers",
        "blurb": "conventional depth: 8 stacked layers (more params), seq 512. The control for the recurrence bet.",
        "ckpt": "viz_ckpt_deep8.pt",
        "kwargs": dict(
            d_model=256, n_layers=8, n_heads=4,
            d_nope=32, d_rope=16, d_v=32, kv_latent=128, q_latent=256,
            n_routed_experts=8, n_active_experts=2, n_shared_experts=1,
            d_ff=512, max_seq=512, recurrence=1,
        ),
    },
}

DEFAULT = "proof"


def build_config(preset_id, vocab_size):
    if preset_id not in PRESETS:
        preset_id = DEFAULT
    return Config(vocab_size=vocab_size, **PRESETS[preset_id]["kwargs"])
