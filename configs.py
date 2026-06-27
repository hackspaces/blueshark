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
}

DEFAULT = "proof"


def build_config(preset_id, vocab_size):
    if preset_id not in PRESETS:
        preset_id = DEFAULT
    return Config(vocab_size=vocab_size, **PRESETS[preset_id]["kwargs"])
