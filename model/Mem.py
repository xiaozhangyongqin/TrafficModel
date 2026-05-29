"""Compatibility memory modules used by TrafficModel.

The main active memory implementation lives in MemoryTransformer.py.  These
classes are kept lightweight because TrafficModel instantiates them for backward
compatibility, even though the current forward path does not directly call them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTemporalMemory(nn.Module):
    """Lightweight spatial-temporal memory reader.

    Args:
        sequence_length: Historical sequence length.
        num_nodes: Number of nodes.
        model_dim: Hidden dimension.
        memory_slots: Number of memory slots.
        memory_dim: Memory dimension. Projected internally when different from
            model_dim.
    """

    def __init__(
        self,
        sequence_length: int,
        num_nodes: int,
        model_dim: int,
        memory_slots: int = 10,
        memory_dim: int = 64,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim

        self.memory = nn.Parameter(torch.empty(memory_slots, memory_dim))
        nn.init.xavier_uniform_(self.memory)

        self.query_proj = nn.Linear(model_dim, memory_dim)
        self.out_proj = nn.Linear(memory_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D]
        q = self.query_proj(x)
        attn = torch.einsum("btnd,md->btnm", q, self.memory) / (self.memory_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        memory_read = torch.einsum("btnm,md->btnd", attn, self.memory)
        return x + self.out_proj(memory_read)


class MultiScaleAttentionMemory(nn.Module):
    """Compact multi-scale memory attention.

    This module provides a safe implementation for configs that instantiate it.
    It can be used as a residual memory reader on [B, T, N, D] tensors.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        memory_slots: int = 10,
        max_scales: int = 5,
        attention_dim: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_slots = memory_slots
        self.max_scales = max_scales
        self.attention_dim = attention_dim

        self.memory = nn.Parameter(torch.empty(memory_slots, input_dim))
        nn.init.xavier_uniform_(self.memory)

        self.query_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)
        self.residual_proj = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(x)
        attn = torch.einsum("btnd,md->btnm", q, self.memory) / (self.input_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        memory_read = torch.einsum("btnm,md->btnd", attn, self.memory)
        return self.residual_proj(x) + self.out_proj(memory_read)
