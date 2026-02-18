from typing import Mapping
from torch import nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
from typing import Any
import torch
import math


class Linear(nn.Module):
    weights: Float[Tensor, "d_out d_in"]

    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty(d_out, d_in, dtype=dtype, device=device)
        )
        sigma = math.sqrt(2 / (d_in + d_out))
        nn.init.trunc_normal_(
            self.weights, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> torch.Tensor:
        y = x @ self.weights.T
        return y
