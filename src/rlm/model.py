"""
RLM — Recurrent Depth Transformer (RDT).

This is a verbatim copy of RLM/open_mythos/main.py, renamed to model.py
for consistency within the versper_mythos namespace.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


@dataclass
class RLMConfig:
    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 4096
    max_loop_iters: int = 16
    prelude_layers: int = 2
    coda_layers: int = 2
    attn_type: str = "mla"
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4
    expert_dim: int = 512
    act_threshold: float = 0.99
    rope_theta: float = 500000.0
    lora_rank: int = 16
    max_output_tokens: int = 4096
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


# ---------------------------------------------------------------------------
# Grouped Query Attention
# ---------------------------------------------------------------------------

class GQAttention(nn.Module):
    def __init__(self, cfg: RLMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}
        if _HAS_FLASH_ATTN:
            orig_dtype = q.dtype
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            dropout_p = self.dropout_p if self.training else 0.0
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=(mask is not None))
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = F.dropout(F.softmax(attn, dim=-1), p=self.dropout_p, training=self.training)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V2 style)
# ---------------------------------------------------------------------------

class MLAttention(nn.Module):
    def __init__(self, cfg: RLMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False)
        self.q_up_rope = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False)
        self.kv_down = nn.Linear(cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(cfg.kv_lora_rank, cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"):
        B, T, _ = x.shape
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., :self.kv_lora_rank]
        k_rope = kv_raw[..., self.kv_lora_rank:]
        k_rope = k_rope.unsqueeze(2).expand(B, T, self.n_heads, self.qk_rope_dim).contiguous()
        k_rope = apply_rope(k_rope, freqs_cis)
        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}
        S = c_kv.shape[1]
        kv = self.kv_up(self.kv_norm(c_kv))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., :self.qk_nope_dim]
        v = kv[..., self.qk_nope_dim:]
        k = torch.cat([k_nope, k_rope], dim=-1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = self.q_head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    def __init__(self, dim: int, expert_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    def __init__(self, cfg: RLMConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))
        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok) for _ in range(self.n_shared)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.view(B * T, D)
        logits = self.router(flat)
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat)
        for i in range(self.topk):
            expert_ids = topk_idx[:, i]
            token_scores = topk_scores[:, i].unsqueeze(-1)
            for eid in range(self.n_experts):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                out[mask] += token_scores[mask] * self.routed_experts[eid](flat[mask])
        for shared in self.shared_experts:
            out = out + shared(flat)
        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Loop-index RoPE
# ---------------------------------------------------------------------------

def loop_index_embedding(h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim))
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Depth-wise LoRA adapter
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.scale = nn.Embedding(max_loops, rank)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        max_t = self.scale.num_embeddings - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale(torch.tensor(t_idx, device=x.device))
        down = self.down(x) * s
        return down @ self.B


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: RLMConfig, use_moe: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"):
        x = x + self.resid_drop(self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key))
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# LTI-stable injection
# ---------------------------------------------------------------------------

class LTIInjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ---------------------------------------------------------------------------
# ACT Halting
# ---------------------------------------------------------------------------

class ACTHalting(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Recurrent Block
# ---------------------------------------------------------------------------

class RecurrentBlock(nn.Module):
    def __init__(self, cfg: RLMConfig):
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = cfg.dim // 8

    def forward(self, h, e, freqs_cis, mask=None, n_loops=None, kv_cache=None):
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape
        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)
        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)
            p = self.act(h)
            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold, remainder, p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h
            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)
            if halted.all() and kv_cache is None:
                break
        return h_out


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class RLM(nn.Module):
    def __init__(self, cfg: RLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis_mla", freqs_mla)
        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(self, input_ids: torch.Tensor, n_loops: Optional[int] = None,
                kv_cache: Optional[dict] = None, start_pos: int = 0) -> torch.Tensor:
        T = input_ids.shape[1]
        device = input_ids.device
        x = self.embed(input_ids)
        freqs_cis = (self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis)[start_pos: start_pos + T]
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None
        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")
        e = x
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)
        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 n_loops: int = 8, temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        for step in range(max_new_tokens):
            cur_ids = input_ids if step == 0 else input_ids[:, -1:]
            start_pos = 0 if step == 0 else prompt_len + step - 1
            logits = self.forward(cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids
