import torch
from torch import nn
from transformers.activations import ACT2FN


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

    Optionally applies dropout to the output.
    """
    def __init__(self, hidden_size: int, intermediate_size: int = None, hidden_act: str = 'silu', dropout: float = 0.0):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
