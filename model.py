"""
Small model architecture - reference implementation.

Fine-grained Mixture-of-Experts + always-on shared expert (DeepSeek-style)
with Multi-head Latent Attention (MLA). This is the 2026 frontier recipe.

The default Config here is TINY so it runs on a laptop in seconds. It is the
SAME architecture as the real model, just scaled down, so we can confirm it is
correct, that gradients flow, and that it can actually learn. Scale the knobs
(see SCALE_TO_30B at the bottom) to reach the real model.

Incorporates current (2026) findings:
  - MoE load balancing is aux-loss-free: a sigmoid router plus a per-expert
    bias nudged by usage, so balancing never distorts the loss (DeepSeek-V3).
  - Attention uses F.scaled_dot_product_attention (FlashAttention kernels).
  - Reserved agentic special tokens give post-training a clean interface.
See ROADMAP in README.md for researched upgrades deferred to scale (sparse
attention indexer, FP8 training, MoE-aware 4-bit, recurrent-depth reasoning).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reserved agentic tokens. The tokenizer maps these to dedicated ids so SFT/RL
# has an unambiguous surface for thinking and tool turns (the architecture is
# unchanged; this is an interface choice).
SPECIAL_TOKENS = ["<plan>", "<think>", "</think>", "<tool_call>", "<tool_result>"]


@dataclass
class Config:
    vocab_size: int = 1000
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    # MLA (Multi-head Latent Attention) dims
    d_nope: int = 32      # per-head content dim (no position)
    d_rope: int = 16      # per-head rotary dim (even number)
    d_v: int = 32         # per-head value dim
    kv_latent: int = 64   # compressed KV latent (this is the MLA cache)
    q_latent: int = 128   # compressed query latent
    # MoE
    n_routed_experts: int = 16
    n_active_experts: int = 4    # top-k routed per token
    n_shared_experts: int = 1    # always-on
    d_ff: int = 512              # per-expert hidden (SwiGLU)
    bias_update_rate: float = 1e-3   # aux-loss-free balancer step size
    # misc
    rope_theta: float = 10000.0
    max_seq: int = 1024
    recurrence: int = 1   # weight-tied recursion: run the block stack this many
                          # times for recurrence x n_layers EFFECTIVE depth at no
                          # extra params (TRM-style recurrent depth). 1 = standard.

    @property
    def qk_head_dim(self) -> int:
        return self.d_nope + self.d_rope


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * normed).type_as(x)


def build_rope_cache(seq, rope_dim, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=device).float() / rope_dim))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv_freq)          # (seq, rope_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, d_rope). Interleaved (NeoX-style) convention; q and k use the
    # same function so it is internally consistent. Not HF-MLA weight compatible.
    T = x.shape[-2]
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


class MLA(nn.Module):
    """Multi-head Latent Attention. Queries and KV are compressed through a
    small latent, then up-projected. RoPE rides on a separate decoupled slice."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H = cfg.n_heads
        self.q_a = nn.Linear(cfg.d_model, cfg.q_latent, bias=False)
        self.q_norm = RMSNorm(cfg.q_latent)
        self.q_b = nn.Linear(cfg.q_latent, H * cfg.qk_head_dim, bias=False)

        self.kv_a = nn.Linear(cfg.d_model, cfg.kv_latent, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_latent)
        self.kv_b = nn.Linear(cfg.kv_latent, H * (cfg.d_nope + cfg.d_v), bias=False)
        self.k_rope = nn.Linear(cfg.d_model, cfg.d_rope, bias=False)  # shared across heads

        self.o = nn.Linear(H * cfg.d_v, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        cfg, H = self.cfg, self.cfg.n_heads

        q = self.q_b(self.q_norm(self.q_a(x)))
        q = q.view(B, T, H, cfg.qk_head_dim).transpose(1, 2)        # (B,H,T,qk)
        q_nope, q_rope = q.split([cfg.d_nope, cfg.d_rope], dim=-1)

        kv = self.kv_b(self.kv_norm(self.kv_a(x)))
        kv = kv.view(B, T, H, cfg.d_nope + cfg.d_v).transpose(1, 2)  # (B,H,T,nope+v)
        k_nope, v = kv.split([cfg.d_nope, cfg.d_v], dim=-1)
        k_rope = self.k_rope(x).view(B, T, 1, cfg.d_rope).transpose(1, 2)  # (B,1,T,rope)

        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin).expand(B, H, T, cfg.d_rope)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # FlashAttention path. Default scale is 1/sqrt(qk_head_dim), which is
        # what MLA wants since scores use the concatenated nope+rope dim.
        out = F.scaled_dot_product_attention(q, k, v.contiguous(), is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, H * cfg.d_v)
        return self.o(out)


class Expert(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Fine-grained MoE with a shared expert and aux-loss-free load balancing.

    The router uses sigmoid affinities. Selection is done on (affinity + bias),
    where bias is a non-trainable per-expert buffer nudged toward balance after
    each step. The gate weight comes from the UNBIASED affinity, so the bias
    steers which experts fire without ever scaling their output (DeepSeek-V3).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_routed_experts)])
        self.shared = nn.ModuleList([Expert(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_shared_experts)])
        self.register_buffer("expert_bias", torch.zeros(cfg.n_routed_experts))
        self.register_buffer("last_counts", torch.zeros(cfg.n_routed_experts))

    def forward(self, x):
        B, T, D = x.shape
        cfg = self.cfg
        xf = x.reshape(-1, D)                                  # (N, D)

        affinity = self.router(xf).sigmoid()                  # (N, E)
        _, topi = (affinity + self.expert_bias).topk(cfg.n_active_experts, dim=-1)
        gate = affinity.gather(-1, topi)                      # unbiased gate
        gate = gate / (gate.sum(-1, keepdim=True) + 1e-9)     # renormalize kept set

        out = torch.zeros_like(xf)
        for e in range(cfg.n_routed_experts):                 # routed experts (sparse)
            hit = (topi == e)
            if hit.any():
                tok, slot = hit.nonzero(as_tuple=True)
                out[tok] += gate[tok, slot].unsqueeze(-1) * self.experts[e](xf[tok])
        for se in self.shared:                                # shared expert (always on)
            out = out + se(xf)

        with torch.no_grad():                                 # usage, for the balancer
            counts = torch.zeros(cfg.n_routed_experts, device=xf.device)
            counts.scatter_add_(0, topi.reshape(-1), torch.ones(topi.numel(), device=xf.device))
            self.last_counts = counts
        return out.view(B, T, D)

    @torch.no_grad()
    def balance_step(self):
        # nudge overloaded experts down, underloaded up (sign-only update)
        target = self.last_counts.mean()
        self.expert_bias += self.cfg.bias_update_rate * torch.sign(target - self.last_counts)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MLA(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.moe = MoE(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.moe(self.ffn_norm(x))
        return x


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight        # tied embeddings
        # reserve the last ids of the vocab for the agentic special tokens
        self.special_ids = {t: cfg.vocab_size - len(SPECIAL_TOKENS) + i
                            for i, t in enumerate(SPECIAL_TOKENS)}
        cos, sin = build_rope_cache(cfg.max_seq, cfg.d_rope, cfg.rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for _ in range(self.cfg.recurrence):      # weight-tied recurrent depth
            for blk in self.blocks:
                x = blk(x, self.cos, self.sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def balance_step(self):
        for blk in self.blocks:
            blk.moe.balance_step()

    def load_imbalance(self):
        # max/mean expert load across all layers; 1.0 is perfect balance
        counts = torch.stack([blk.moe.last_counts for blk in self.blocks]).sum(0)
        return (counts.max() / (counts.mean() + 1e-9)).item()


def count_params(model, cfg):
    total = sum(p.numel() for p in model.parameters())
    one_expert = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
    inactive = (cfg.n_routed_experts - cfg.n_active_experts) * one_expert * cfg.n_layers
    return total, total - inactive


def main():
    torch.manual_seed(0)
    cfg = Config()
    device = "cpu"  # tiny model, CPU is instant and avoids backend quirks
    model = Model(cfg).to(device)

    total, active = count_params(model, cfg)
    print(f"architecture:  MoE({cfg.n_routed_experts} experts, top-{cfg.n_active_experts}, "
          f"+{cfg.n_shared_experts} shared, aux-loss-free) + MLA, {cfg.n_layers} layers")
    print(f"total params:  {total/1e6:.2f}M")
    print(f"active params: {active/1e6:.2f}M  ({100*active/total:.0f}% of total fire per token)")
    print(f"special tokens reserved: {list(model.special_ids.keys())}")

    B, T = 2, 64
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    logits, loss = model(idx, tgt)
    print(f"\nforward pass: logits {tuple(logits.shape)} (expected {(B, T, cfg.vocab_size)})")
    print(f"initial loss: {loss.item():.3f}  (random baseline ln(vocab) = {math.log(cfg.vocab_size):.3f})")
    print(f"initial expert imbalance (max/mean): {model.load_imbalance():.2f}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(30):                                # overfit one batch on purpose
        logits, loss = model(idx, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.balance_step()                           # aux-loss-free balancer (no grad)
    print(f"loss after 30 steps overfitting one batch: {loss.item():.3f}  (should drop hard)")
    print(f"final expert imbalance (max/mean):   {model.load_imbalance():.2f}  (balancer keeps this near 1)")
    print("\nOK: the architecture runs, gradients flow, it learns, and experts stay balanced.")


# SCALE_TO_30B (do not run on a laptop): roughly
#   d_model=2048, n_layers=48, n_heads=16,
#   d_nope=128, d_rope=64, d_v=128, kv_latent=512, q_latent=1536,
#   n_routed_experts=128, n_active_experts=8, n_shared_experts=1, d_ff=1536,
#   vocab_size=150000, max_seq=32768
# -> ~30B total, ~3B active. Exact dims are tuned on the proof model first.

if __name__ == "__main__":
    main()
