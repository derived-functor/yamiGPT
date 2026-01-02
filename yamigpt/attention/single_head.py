"""Single head attention implementation"""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class Attention(nn.Module):
    """Single Head Attention

    :ivar d_model: dimension of input and output model.
    :ivar d_k: dimension of K-vector.
    :ivar d_v: dimension of V-vector.
    :ivar dropout: probability of dropout on train.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(
            d_model, d_k
        )
        self.W_k = nn.Linear(
            d_model, d_k
        )
        self.W_v = nn.Linear(
            d_model, d_v
        )
        # Output projection
        self.W_o = nn.Linear(
            d_v, d_model
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, seq_len, d_model)
        mask: ()
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Q (batch_size, seq_len, d_k) @ K^T (batch_size, d_k, seq_len) ->
        # -> (batch_size, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-1, -2))/(math.sqrt(self.d_k))

        if mask is not None:
            scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention = torch.matmul(attention_weights, V)
        output = self.W_o(attention)

        return output, attention
