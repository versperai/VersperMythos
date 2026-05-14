"""
Rotary Position Embedding (RoPE).

Provides two APIs:
  1. cos/sin formulation (used by MiniMind)  — precompute_freqs_cis / apply_rotary_pos_emb
  2. complex formulation (used by OpenMythos) — precompute_rope_freqs / apply_rope

Both compute the same rotation mathematically.
"""

import math
from typing import Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────
# MiniMind-style: separate cos/sin tensors
# ─────────────────────────────────────────────────────────────

def precompute_freqs_cis(
    dim: int,
    end: int = 32768,
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute cos/sin tables for Rotary Position Embedding.

    Returns two tensors of shape (end, dim) with duplicated cos/sin values.
    Supports YaRN context-extension scaling.
    """
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim_fn = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim_fn(beta_fast)), 0)
            high = min(math.ceil(inv_dim_fn(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0, 1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors using the
    rotate-half formulation.
    """
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


# ─────────────────────────────────────────────────────────────
# OpenMythos-style: complex-number phasors
# ─────────────────────────────────────────────────────────────

def precompute_rope_freqs(
    max_len: int,
    dim: int,
    theta: float = 500000.0,
) -> torch.Tensor:
    """
    Pre-compute complex rotary phasors.

    Returns a complex64 tensor of shape (max_len, dim // 2) where
    each entry is e^{i * m * theta_i}.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings via complex multiplication.

    Treats the last dimension of x as pairs of (real, imag) and
    multiplies by the pre-computed phasor.
    """
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(-2).to(x.dtype)
