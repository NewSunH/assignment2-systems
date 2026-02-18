from __future__ import annotations

import einops
from regex import W
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, split
import math
from torch import nn
from einops import einsum
import einops

from .linear import Linear
from .rope import RoPE


def softmax(v: Float[Tensor, "... d"], i: int) -> Float[Tensor, "..."]:
    v_max = torch.max(v, dim=i, keepdim=True).values
    v_stable = v - v_max
    exp_v = torch.exp(v_stable)

    return exp_v / exp_v.sum(dim=i, keepdim=True)


def scaled_dot_product_attention(
    keys: Float[Tensor, "batch_size ... seq_len d_key"],
    queries: Float[Tensor, "batch_size ... seq_len d_key"],
    value: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    d_key = queries.shape[-1]
    scores = (
        einsum(
            queries,
            keys,
            "batch_size ... q d, batch_size ... k d -> batch_size ... q k",
        )
    ) / math.sqrt(d_key)
    if mask is not None:
        # mask == False -> -inf
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    return softmax(scores, -1) @ value


class MultipleHeadSelfAttention(nn.Module):
    d_model: int
    num_heads: int
    d_key: int
    d_value: int

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        assert d_model % num_heads == 0, "d_model % num_heads must be 0"
        self.d_model = d_model
        self.num_heads = num_heads
        super().__init__()
        self.d_key = d_model // num_heads
        self.d_value = d_model // num_heads
        self.W_KQV = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(
            theta=theta,
            d_key=self.d_key,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
        rope: bool = False,
    ):

        KQV = self.W_KQV.forward(x)
        K, Q, V = KQV.chunk(3, dim=-1)

        Q = einops.rearrange(Q, "b l (h d) -> b h l d", h=self.num_heads)
        K = einops.rearrange(K, "b l (h d) -> b h l d", h=self.num_heads)
        V = einops.rearrange(V, "b l (h d) -> b h l d", h=self.num_heads)

        L = x.shape[-2]
        if token_positions == None:
            token_positions = torch.arange(L, device=x.device)

        if rope:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(keys=K, queries=Q, value=V, mask=mask)
        out = einops.rearrange(out, "b h l d -> b l (h d)")

        out = self.W_O.forward(out)
        return out
