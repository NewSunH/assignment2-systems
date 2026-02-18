import math
import typing
import torch
from torch import nn, Tensor

from jaxtyping import Float, Int
import numpy as np
import os


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Tensor:
    vocab_dim = -1
    max_logits = logits.max(dim=vocab_dim, keepdim=True).values
    shifted = logits - max_logits
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=vocab_dim))
    target_logits = logits.gather(
        dim=vocab_dim,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    loss = -target_logits + max_logits.squeeze(-1) + log_sum_exp
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                param.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

                if weight_decay != 0:
                    param.data.add_(param.data, alpha=-lr * weight_decay)

        return loss


def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6):
    parameters = list(parameters)

    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad
        total_norm_sq += grad.norm(2).item() ** 2

    total_norm = total_norm_sq**0.5
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)
    return total_norm


def data_loading(
    x: np.typing.NDArray[np.int_],
    batch_size: int,
    context_length: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # We need (input, target) windows of length context_length, where targets are inputs shifted by 1.
    # For a start index s, we access x[s : s+context_length] and x[s+1 : s+1+context_length],
    # so the largest valid s is len(x) - context_length - 1.
    # np.random.randint uses an exclusive high bound, so high must be (largest_valid_s + 1)
    # = len(x) - context_length.
    max_start_exclusive = len(x) - context_length
    starts = np.random.randint(0, max_start_exclusive, size=batch_size)
    inputs = []
    targets = []
    for s in starts:
        inputs.append(x[s : s + context_length])
        targets.append(x[s + 1 : s + 1 + context_length])
    inputs = np.stack(inputs)
    targets = np.stack(targets)
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets


def learning_rate_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if t < warmup_iters:
        # linear warmup
        return lr_max * t / warmup_iters

    elif t <= cosine_cycle_iters:
        # cosine decay
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return lr_min + 0.5 * (1 + math.cos(math.pi * progress)) * (lr_max - lr_min)

    else:
        return lr_min


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["iteration"])
