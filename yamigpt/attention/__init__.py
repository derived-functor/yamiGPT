"""Attention module"""

from .single_head import Attention
from .multihead import MultiHeadAttention

__all__ = [
    "Attention",
    "MultiHeadAttention"
]
