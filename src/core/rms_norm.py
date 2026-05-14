import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    RMS Layer Normalization.

    Computes: output = weight * x * rsqrt(mean(x^2) + eps)

    Casts to float32 internally for numerical safety in mixed precision,
    then casts back to the original dtype.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)
