from torch import nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
import torch


class Embedding(nn.Module):
    embedding_matrix: Float[Tensor, "vocab_size d_model"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding_matrix = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        sigma = 1
        nn.init.trunc_normal_(
            self.embedding_matrix, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )

    def forward(
        self, token_ids: Int[Tensor, "batch_size seq_length"]
    ) -> Float[Tensor, "batch_size seq_length d_model"]:
        return self.embedding_matrix[token_ids]
