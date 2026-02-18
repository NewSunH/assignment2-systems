import torch
from torch import nn, tensor
from torch import Tensor
from jaxtyping import Bool, Float, Int
import math


class RmsNorm(nn.Module):
    gain: Float[Tensor, "d_model"]
    eps: float
    d_model: int

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(
        self, x: Float[Tensor, "batch_size seq_length d_model"]
    ) -> Float[Tensor, "batch_size seq_length d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.gain
        return result.to(in_dtype)
