from torch import nn
from torch import Tensor
from jaxtyping import Float, Int
import torch
from einops import rearrange


class RoPE(nn.Module):
    d_key: int
    max_seq_len: int
    sin: Float[Tensor, "max_seq_len d_key_half"]
    cos: Float[Tensor, "max_seq_len d_key_half"]
    cos_cached: Float[Tensor, "max_seq_len d_key_half"]
    sin_cached: Float[Tensor, "max_seq_len d_key_half"]

    def __init__(
        self,
        theta: float,
        d_key: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_key = d_key
        self.max_seq_len = max_seq_len
        assert d_key % 2 == 0, "d_key must be even."

        pos = torch.arange(max_seq_len, device=device, dtype=dtype)
        inv_freq = theta ** (
            -torch.arange(0, d_key, 2, device=device, dtype=dtype) / d_key
        )

        angles: Float[Tensor, "max_seq_len d_key_half"] = rearrange(
            pos, "max_seq_len -> max_seq_len 1"
        ) * rearrange(inv_freq, "d_key_half -> 1 d_key_half")
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_key"],
        token_positions: Int[Tensor, "... seq_len"],
    ):
        token_positions = token_positions.to(torch.long)

        x = rearrange(x, "... s (p two) -> ... s p two", two=2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x0 = x[..., 0]
        x1 = x[..., 1]

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        y = torch.stack((y0, y1), dim=-1)
        return rearrange(y, "... s p two -> ... s (p two)")
