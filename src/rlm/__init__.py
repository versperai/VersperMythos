"""
RLM — Recurrent Depth Transformer (RDT).

Theoretical reconstruction of the Claude Mythos architecture with
LTI-stable state injection, ACT halting, depth-wise LoRA, and
optional GQA/MLA attention.
"""

from src.rlm.model import (
    ACTHalting,
    Expert,
    GQAttention,
    LoRAAdapter,
    LTIInjection,
    MLAttention,
    MoEFFN,
    RLMConfig,
    RLM,
    RecurrentBlock,
    RMSNorm,
    TransformerBlock,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)
from src.rlm.tokenizer import RLMTokenizer
from src.rlm.variants import (
    rlm_1b,
    rlm_1t,
    rlm_3b,
    rlm_10b,
    rlm_50b,
    rlm_100b,
    rlm_500b,
)

__all__ = [
    "RLMConfig",
    "RMSNorm",
    "GQAttention",
    "MLAttention",
    "Expert",
    "MoEFFN",
    "LoRAAdapter",
    "TransformerBlock",
    "LTIInjection",
    "ACTHalting",
    "RecurrentBlock",
    "RLM",
    "precompute_rope_freqs",
    "apply_rope",
    "loop_index_embedding",
    "rlm_1b",
    "rlm_3b",
    "rlm_10b",
    "rlm_50b",
    "rlm_100b",
    "rlm_500b",
    "rlm_1t",
    "RLMTokenizer",
]
