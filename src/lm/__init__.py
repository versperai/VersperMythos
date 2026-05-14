"""
MiniMind — Dense + MoE pure language model.

Supports pretraining, SFT, LoRA, DPO, PPO, GRPO, SPO,
knowledge distillation, and reasoning distillation.
"""

from src.lm.model import (
    LMConfig,
    LMModel,
    LMForCausalLM,
    LMBlock,
    Attention,
    FeedForward,
    MOEFeedForward,
    MoEGate,
    RMSNorm,
)

from src.lm.lora import LoRA, apply_lora, load_lora, save_lora

__all__ = [
    "LMConfig",
    "LMModel",
    "LMForCausalLM",
    "LMBlock",
    "Attention",
    "FeedForward",
    "MOEFeedForward",
    "MoEGate",
    "RMSNorm",
    "LoRA",
    "apply_lora",
    "load_lora",
    "save_lora",
]
