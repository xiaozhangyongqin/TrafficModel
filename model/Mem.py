import torch
import torch.nn as nn


class SpatialTemporalMemory(nn.Module):
    """
    Lightweight spatial-temporal memory placeholder.

    Keeps API compatibility with TrafficModel while remaining a no-op
    unless future logic explicitly uses this memory output.
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
        self.memory_bank = nn.Parameter(torch.zeros(memory_slots, memory_dim))
        nn.init.xavier_uniform_(self.memory_bank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MultiScaleAttentionMemory(nn.Module):
    """
    Lightweight multi-scale attention memory placeholder.

    Keeps API compatibility with TrafficModel while remaining a no-op
    unless future logic explicitly uses this memory output.
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
        self.proj = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
