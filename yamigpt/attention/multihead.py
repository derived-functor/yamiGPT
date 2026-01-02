"""Implementation of Multi-Head Attention"""

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout: float = 0.2
    ):

        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "Model dimension should be divisible by number of heads. " +
                f"d_model={d_model}, num_heads={num_heads}"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self._sqrt_d_head = math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (batch_size, seq_len, d_model)
        :param mask: (batch_size, 1, 1, seq_len)
            or (batch_size, 1, seq_len, seq_len)

        :return: Tuple of output and attention weights.
        """
        assert x.dim() == 3
        assert x.size(-1) == self.d_model

        Q: torch.Tensor = self.W_q(x)
        K: torch.Tensor = self.W_k(x)
        V: torch.Tensor = self.W_v(x)

        batch_size, seq_len, d_model = Q.shape

        # (batch_size, seq_len, h, d_head)
        Q_h = Q.reshape(
            (batch_size, seq_len, self.num_heads, self.d_head)
        )
        K_h = K.reshape(
            (batch_size, seq_len, self.num_heads, self.d_head)
        )
        V_h = V.reshape(
            (batch_size, seq_len, self.num_heads, self.d_head)
        )

        # (batch_size, h, seq_len, d_head)
        Q_h = Q_h.transpose(1, 2)
        K_h = K_h.transpose(1, 2)
        V_h = V_h.transpose(1, 2)

        # Q_h (batch_size, h, seq_len d_head)
        # @ K_h^T (batch_size, h, d_head, seq_len)
        # -> (batch_size, h, seq_len, seq_len)
        scores = torch.matmul(
            Q_h, K_h.transpose(-1, -2)
        ) / self._sqrt_d_head

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # attention_weights (B, h, seq_len, seq_len)
        # @ V_h (B, h, seq_len, d_head)
        # -> (B, h, seq_len, d_head)
        output = torch.matmul(attention_weights, V_h)
        # output (B, seq_len, h, d_head)
        output = output.transpose(1, 2)
        # output (B, seq_len, d_model)
        output = output.reshape(
            (batch_size, seq_len, d_model)
        )

        output = self.W_o(output)

        return output, attention_weights
