import torch
from torch import nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
from typing import Callable
import math


def silu(x: Float) -> Float:
    return x * torch.sigmoid(x)


def _round_up_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be > 0")
    return ((int(x) + multiple - 1) // multiple) * multiple


def matched_silu_d_ff(*, d_ff_swiglu: int) -> int:
    """Choose a SiLU MLP hidden size with ~matched params to SwiGLU.

    SwiGLU FFN params (ignoring bias): 3 * d_model * d_ff
    SiLU  FFN params (ignoring bias): 2 * d_model * d_ff_silu

    So choose d_ff_silu ~= 1.5 * d_ff, then round up to a multiple of 64.
    """

    if int(d_ff_swiglu) <= 0:
        raise ValueError("d_ff_swiglu must be > 0")
    raw = math.ceil(1.5 * float(d_ff_swiglu))
    return _round_up_to_multiple(int(raw), 64)


class FFN(nn.Module):
    weight1: Float[Tensor, "d_ff d_model"]
    weight2: Float[Tensor, "d_model d_ff"]
    weight3: Float[Tensor, "d_ff d_model"]
    activation: Callable[[Float], Float]

    def __init__(
        self,
        d_model: int,
        d_ff: int | None,
        activation: Callable[[Float], Float],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff == None:
            raw_d_ff = int(8 * d_model / 3)
            d_ff = ((raw_d_ff + 63) // 64) * 64

        sigma = math.sqrt(2 / (d_ff + d_model))
        self.weight1 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        self.weight2 = nn.Parameter(
            torch.empty(d_model, d_ff, device=device, dtype=dtype)
        )
        self.weight3 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(
            self.weight1, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        nn.init.trunc_normal_(
            self.weight2, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        nn.init.trunc_normal_(
            self.weight3, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        self.activation = activation

    def forward(self, x: Float[Tensor, "d_model"]) -> Float[Tensor, "d_model"]:
        u = x @ self.weight1.T
        v = x @ self.weight3.T
        gate = self.activation(u)
        h = gate * v
        y = h @ self.weight2.T
        return y


class SiLUFFN(nn.Module):
    """Non-gated 2-layer SiLU MLP FFN.

    This is used for the swiglu_ablation experiment.
    """

    weight1: Float[Tensor, "d_ff d_model"]
    weight2: Float[Tensor, "d_model d_ff"]
    activation: Callable[[Float], Float]

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: Callable[[Float], Float],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if int(d_ff) <= 0:
            raise ValueError("d_ff must be > 0")

        sigma = math.sqrt(2 / (d_ff + d_model))
        self.weight1 = nn.Parameter(
            torch.empty(int(d_ff), int(d_model), device=device, dtype=dtype)
        )
        self.weight2 = nn.Parameter(
            torch.empty(int(d_model), int(d_ff), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(
            self.weight1, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        nn.init.trunc_normal_(
            self.weight2, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        self.activation = activation

    def forward(self, x: Float[Tensor, "d_model"]) -> Float[Tensor, "d_model"]:
        h = self.activation(x @ self.weight1.T)
        y = h @ self.weight2.T
        return y
