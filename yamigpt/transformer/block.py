"""Implementations of transformer blocks"""

import torch
import torch.nn as nn

from ..attention import MultiHeadAttention

class FeedForward(nn.Module):
    """Implementation of simple FFN"""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_hidden, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, seq_len, d_model)
        """

        # (batch_size, seq_len, d_hidden)
        x = self.linear1(x)
        # (batch_size, seq_len, d_hidden)
        x = self.relu(x)
        # (batch_size, seq_len, d_model)
        x = self.linear2(x)

        return x

class TransformerBlock(nn.Module):
    """Implementation of Transformer Block"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_hidden_ffn: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(num_heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_hidden_ffn)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        :param x: (batch_size, seq_len, d_model)
        :param mask: (seq_len, seq_len)
            or (batch_size, 1, seq_len, seq_len)
        """

        # (batch_size, seq_len, d_model)
        x_mha, _ = self.mha(self.layer_norm1(x), mask)
        x = x + x_mha
        # (batch_size, seq_len, d_model)
        x = x + self.ffn(self.layer_norm2(x))

        return x
