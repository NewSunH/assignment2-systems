from __future__ import annotations

import torch
from torch import nn, Tensor
from jaxtyping import Float, Int

from .attention import MultipleHeadSelfAttention as Mhsa
from .attention import softmax
from .normalization import RmsNorm
from .positionwise_feedforward import FFN, SiLUFFN, matched_silu_d_ff, silu
from .embedding import Embedding
from .linear import Linear


from cs336_basics import linear


class TransformerBlock(nn.Module):
    mhsa: Mhsa
    rmsnorm1: nn.Module
    rmsnorm2: nn.Module
    ffn: nn.Module

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        *,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        ffn_variant: str = "swiglu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.use_rope = bool(use_rope)
        self.ffn_variant = str(ffn_variant)
        self.mhsa = Mhsa(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.rmsnorm1 = (
            RmsNorm(d_model=d_model, device=device, dtype=dtype)
            if use_rmsnorm
            else nn.Identity()
        )
        self.rmsnorm2 = (
            RmsNorm(d_model=d_model, device=device, dtype=dtype)
            if use_rmsnorm
            else nn.Identity()
        )
        if self.ffn_variant == "swiglu":
            self.ffn = FFN(
                d_model=d_model, d_ff=d_ff, activation=silu, device=device, dtype=dtype
            )
        elif self.ffn_variant == "silu":
            d_ff_silu = matched_silu_d_ff(d_ff_swiglu=int(d_ff))
            self.ffn = SiLUFFN(
                d_model=d_model,
                d_ff=int(d_ff_silu),
                activation=silu,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError("ffn_variant must be 'swiglu' or 'silu'")

    def forward(
        self, x: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:

        y = x + self.mhsa.forward(self.rmsnorm1.forward(x), rope=self.use_rope)
        z = y + self.ffn.forward(self.rmsnorm2.forward(y))

        return z


class TransformerLm(nn.Module):
    embedding: Embedding
    transformer_blocks: nn.ModuleList
    norm: nn.Module
    lm_head: Linear

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        *,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        ffn_variant: str = "swiglu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.use_rope = bool(use_rope)
        self.ffn_variant = str(ffn_variant)

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    use_rmsnorm=use_rmsnorm,
                    use_rope=use_rope,
                    ffn_variant=ffn_variant,
                    device=device,
                    dtype=dtype,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = (
            RmsNorm(d_model=d_model, device=device, dtype=dtype)
            if use_rmsnorm
            else nn.Identity()
        )
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, token: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        data = self.embedding.forward(token_ids=token)
        for block in self.transformer_blocks:
            assert isinstance(block, TransformerBlock), "幽默ModuleList不是Generic[T]"
            data = block.forward(data)
        data = self.norm.forward(data)
        logits = self.lm_head.forward(data)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: Int[Tensor, "batch_size sequence_length"],
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> Int[Tensor, "batch_size new_sequence_length"]:
        """Autoregressive decoding for this Transformer LM.

        - Automatically truncates the left context to at most `self.context_length`.
        - If `temperature<=0`, uses greedy decoding.
        - Optionally applies top-k and/or top-p (nucleus) sampling.
        - If `eos_token_id` is provided, stops early once all sequences emit EOS.
        """

        if max_new_tokens <= 0:
            return input_ids

        if top_k is not None and top_k <= 0:
            top_k = None
        if top_p is not None and not (0.0 < float(top_p) <= 1.0):
            raise ValueError("top_p must be in (0, 1].")

        generated = input_ids
        batch_size = generated.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

        for _ in range(max_new_tokens):
            idx_cond = generated[:, -self.context_length :]
            logits = self.forward(idx_cond)[:, -1, :]  # (B, vocab)

            if temperature is None or temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / float(temperature)

                if top_k is not None:
                    k = min(int(top_k), logits.shape[-1])
                    topk_vals, _ = torch.topk(logits, k, dim=-1)
                    kth = topk_vals[:, -1].unsqueeze(-1)
                    logits = torch.where(
                        logits < kth, torch.full_like(logits, -torch.inf), logits
                    )

                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(
                        logits, descending=True, dim=-1
                    )
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative prob above top_p (keep at least 1 token)
                    to_remove = cumprobs > float(top_p)
                    to_remove[..., 0] = False
                    sorted_logits = torch.where(
                        to_remove,
                        torch.full_like(sorted_logits, -torch.inf),
                        sorted_logits,
                    )
                    logits = torch.full_like(logits, -torch.inf).scatter(
                        -1, sorted_idx, sorted_logits
                    )

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, int(eos_token_id)),
                    next_token,
                )

            generated = torch.cat([generated, next_token.to(dtype=torch.long)], dim=-1)

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == int(eos_token_id))
                if bool(finished.all()):
                    break

        return generated
