"""Walk-guided region-level spatial modeling for traffic forecasting."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WalkRegionSpatialBlock(nn.Module):
    """Hierarchical spatial encoder: node walks + coarse region GCN.

    Input/Output shape: [B, T, N, D].

    Complexity is approximately O(B*T*N*num_walks*walk_length*D +
    B*T*num_regions^2*D), instead of full node-pair O(B*T*N^2*D).
    """

    def __init__(
        self,
        num_nodes: int,
        model_dim: int,
        num_regions: int = 20,
        walk_length: int = 16,
        num_walks: int = 3,
        supports: Optional[object] = None,
        dropout: float = 0.1,
        dynamic_walk_weight: bool = True,
        region_gcn_layers: int = 1,
        use_residual_gate: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.model_dim = int(model_dim)
        self.num_regions = max(1, min(int(num_regions), self.num_nodes))
        self.walk_length = max(1, min(int(walk_length), self.num_nodes))
        self.num_walks = max(1, int(num_walks))
        self.dynamic_walk_weight = dynamic_walk_weight
        self.region_gcn_layers = max(1, int(region_gcn_layers))

        adj = self._prepare_adjacency(supports, self.num_nodes)
        walk_index = self._build_walk_index(adj, self.walk_length, self.num_walks)
        self.register_buffer("walk_index", walk_index, persistent=True)

        self.node_embedding = nn.Parameter(torch.empty(self.num_nodes, self.model_dim))
        self.region_prototypes = nn.Parameter(torch.empty(self.num_regions, self.model_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        nn.init.xavier_uniform_(self.region_prototypes)

        self.walk_value_proj = nn.Linear(model_dim, model_dim)
        self.walk_output_proj = nn.Linear(model_dim, model_dim)
        self.region_gcn = nn.ModuleList(
            [nn.Linear(model_dim, model_dim, bias=False) for _ in range(self.region_gcn_layers)]
        )
        self.region_output_proj = nn.Linear(model_dim, model_dim)

        self.norm_walk = nn.LayerNorm(model_dim)
        self.norm_region = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

        if use_residual_gate:
            self.fusion_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer("fusion_gate", torch.tensor(1.0), persistent=False)

    @staticmethod
    def _prepare_adjacency(supports: Optional[object], num_nodes: int) -> torch.Tensor:
        if supports is None:
            return torch.eye(num_nodes, dtype=torch.float32)

        adj = supports[0] if isinstance(supports, (list, tuple)) and len(supports) > 0 else supports
        if isinstance(adj, torch.Tensor):
            adj = adj.detach().float().cpu()
        else:
            adj = torch.as_tensor(adj, dtype=torch.float32)

        if adj.dim() == 3:
            adj = adj[0]
        if adj.dim() != 2 or adj.shape[0] != num_nodes or adj.shape[1] != num_nodes:
            return torch.eye(num_nodes, dtype=torch.float32)

        adj = torch.nan_to_num(adj.clone(), nan=0.0, posinf=0.0, neginf=0.0)
        adj.fill_diagonal_(1.0)
        return adj

    @staticmethod
    def _build_walk_index(adj: torch.Tensor, walk_length: int, num_walks: int) -> torch.Tensor:
        num_nodes = adj.shape[0]
        score = adj.float().abs()
        score.fill_diagonal_(score.diag() + 1.0)
        topk = min(walk_length, num_nodes)

        # BFS-like: strongest adjacent candidates first.
        bfs = torch.topk(score, k=topk, dim=-1).indices

        # DFS-like: same candidate set, complementary reversed order.
        dfs = torch.flip(bfs, dims=[1])

        # RW-like: deterministic pseudo-random order, mixed with strong neighbors.
        node_ids = torch.arange(num_nodes).unsqueeze(1)
        offsets = torch.arange(topk).unsqueeze(0)
        rw = (node_ids * 1103515245 + offsets * 12345 + 97).remainder(num_nodes).long()
        if topk > 1:
            half = topk // 2
            rw[:, :half] = bfs[:, :half]

        candidates = [bfs, dfs, rw]
        while len(candidates) < num_walks:
            candidates.append(torch.roll(bfs, shifts=len(candidates), dims=1))
        return torch.stack(candidates[:num_walks], dim=1).long()

    def _soft_region_assignment(self) -> torch.Tensor:
        node = F.normalize(self.node_embedding, dim=-1)
        region = F.normalize(self.region_prototypes, dim=-1)
        return F.softmax(node @ region.transpose(0, 1), dim=-1)

    def _region_adjacency(self, assignment: torch.Tensor) -> torch.Tensor:
        region_adj = assignment.transpose(0, 1) @ assignment
        region_adj = region_adj + torch.eye(self.num_regions, device=assignment.device)
        degree = region_adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return region_adj / degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, num_nodes, dim = x.shape
        if num_nodes != self.num_nodes or dim != self.model_dim:
            raise ValueError(
                f"Expected input [B,T,{self.num_nodes},{self.model_dim}], "
                f"but got [B,T,{num_nodes},{dim}]."
            )

        flat_x = x.reshape(bsz * seq_len, num_nodes, dim)

        # 1) Node-level walk encoding.
        idx = self.walk_index.to(x.device)
        candidates = flat_x.index_select(dim=1, index=idx.reshape(-1))
        candidates = candidates.reshape(
            bsz * seq_len, num_nodes, self.num_walks, self.walk_length, dim
        )
        candidates = self.walk_value_proj(candidates)

        if self.dynamic_walk_weight:
            query = flat_x.unsqueeze(2).unsqueeze(3)
            attn = (query * candidates).sum(dim=-1) / (dim ** 0.5)
            attn = F.softmax(attn, dim=-1)
            walk_context = (attn.unsqueeze(-1) * candidates).sum(dim=3).mean(dim=2)
        else:
            walk_context = candidates.mean(dim=3).mean(dim=2)

        walk_context = self.walk_output_proj(walk_context)
        walk_context = self.norm_walk(flat_x + self.dropout(walk_context))

        # 2) Region-level graph convolution.
        assignment = self._soft_region_assignment().to(x.device)
        denom = assignment.sum(dim=0).view(1, self.num_regions, 1).clamp_min(1e-6)
        region_h = torch.einsum("nr,bnd->brd", assignment, walk_context) / denom
        region_adj = self._region_adjacency(assignment)

        for layer in self.region_gcn:
            region_h = torch.einsum("rs,bsd->brd", region_adj, region_h)
            region_h = self.dropout(F.gelu(layer(region_h)))

        node_global = torch.einsum("nr,brd->bnd", assignment, region_h)
        node_global = self.region_output_proj(node_global)
        node_global = self.norm_region(walk_context + self.dropout(node_global))

        gate = torch.sigmoid(self.fusion_gate)
        out = flat_x + gate * (node_global - flat_x)
        return out.reshape(bsz, seq_len, num_nodes, dim)
