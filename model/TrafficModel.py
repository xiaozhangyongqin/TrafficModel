"""
Traffic Forecasting Model with Mamba, memory augmentation, and optional
walk-guided region-level spatial modeling.
"""

import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from .mambaEncoder import Mamba, MambaConfig
from .Mem import SpatialTemporalMemory, MultiScaleAttentionMemory
from .MemoryTransformer import MemoryTransformerLayer
from .WalkRegionSpatial import WalkRegionSpatialBlock


class MambaMemoryBlock(nn.Module):
    """Mamba block with memory augmentation for traffic forecasting."""

    def __init__(
        self,
        model_dim: int,
        sequence_length: int,
        num_nodes: int,
        mlp_ratio: float = 2.0,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        dropout_rate: float = 0.1,
        memory_slots: int = 10,
        memory_dim: int = 64,
        attention_slots: int = 10,
        conv_stride: int = 2,
        conv_kernel: int = 3,
        attention_dim: int = 4,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate

        # Kept for backward compatibility with earlier checkpoints/configs.
        self.spatial_temporal_memory = SpatialTemporalMemory(
            sequence_length=sequence_length,
            num_nodes=num_nodes,
            model_dim=model_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
        )
        self.multi_scale_attention = MultiScaleAttentionMemory(
            input_dim=model_dim,
            output_dim=model_dim,
            memory_slots=attention_slots,
            max_scales=5,
            attention_dim=attention_dim,
        )

        self.mem_att = MemoryTransformerLayer(
            model_dim=model_dim,
            memory_slots=attention_slots,
            memory_dim=model_dim,
            num_heads=1,
            dropout_rate=0.1,
        )

        mamba_config = MambaConfig(
            d_model=model_dim,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
        )
        self.mamba_encoder = Mamba(mamba_config)
        self.layer_norm1 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, num_nodes, model_dim = x.shape
        residual = x

        x_memory = self.mem_att(x)
        x_memory = self.layer_norm1(x_memory) + residual

        # Mamba scans the flattened spatio-temporal token sequence.
        x_reshaped = x_memory.reshape(batch_size, seq_len * num_nodes, model_dim)
        mamba_output = self.mamba_encoder(x_reshaped)
        mamba_output = mamba_output.reshape(batch_size, seq_len, num_nodes, model_dim)
        return mamba_output + residual


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)].unsqueeze(2).expand_as(x).detach()


class LearnableEMA(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, T, N, F]
        T = x.shape[1]
        alpha = torch.sigmoid(self.log_alpha)
        t = torch.arange(T, device=x.device).float()
        weights = alpha * (1 - alpha + 1e-6) ** t
        weights = weights.flip(dims=[0])
        weights = weights / weights.sum()
        recent_summary = torch.sum(x * weights.view(1, T, 1, 1), dim=1, keepdim=True)
        return recent_summary.repeat([1, T, 1, 1])


class DataEncoding(nn.Module):
    def __init__(self, in_dim, hid_dim, timelr=True, activation="relu"):
        super().__init__()
        assert activation in ["gelu", "relu"]
        self.learnable_ema = LearnableEMA(T=12)
        self.timelr = timelr
        in_units = in_dim * 3 if timelr else in_dim
        self.linear1 = nn.Linear(in_units, hid_dim)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        # x: [B, T, N, F]
        if self.timelr:
            seq_len = x.shape[1]
            latest_x = x[:, -1:, :, :].repeat([1, seq_len, 1, 1])
            recent_window = min(3, seq_len)
            recent_avg = x[:, -recent_window:, :, :].mean(dim=1, keepdim=True).repeat([1, seq_len, 1, 1])
            data = torch.cat([x, recent_avg, latest_x], dim=-1)
        else:
            data = x
        data = self.linear1(data)
        data = self.activation(data)
        data = self.linear2(data)
        return data


class TrafficForecastingModel(nn.Module):
    """Traffic forecasting model.

    The new walk-region module is controlled by ``use_walk_region_spatial``.
    When enabled, it performs local node walk aggregation and coarse region GCN
    before the existing Mamba-memory backbone.
    """

    def __init__(
        self,
        num_nodes: int,
        input_steps: int = 12,
        output_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 12,
        dow_embedding_dim: int = 12,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 12,
        supports=None,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        mlp_ratio: float = 2.0,
        use_mixed_proj: bool = True,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        dropout_mamba: float = 0.15,
        num_mamba_layers: int = 1,
        dropout_adaptive: float = 0.3,
        memory_slots: int = 10,
        memory_dim: int = 64,
        attention_slots: int = 10,
        conv_stride: int = 2,
        conv_kernel: int = 3,
        attention_dim: int = 24,
        use_walk_region_spatial: bool = False,
        walk_num_regions: int = 20,
        walk_length: int = 16,
        walk_num_walks: int = 3,
        walk_region_dropout: float = 0.1,
        walk_dynamic_weight: bool = True,
        walk_region_gcn_layers: int = 1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.use_walk_region_spatial = use_walk_region_spatial

        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_dataEnco = DataEncoding(input_dim, input_embedding_dim)

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(torch.empty(input_steps, num_nodes, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)

        if spatial_embedding_dim > 0:
            self.spatial_embedding = nn.Embedding(num_nodes, spatial_embedding_dim)
            self.sp_drop = nn.Dropout(0.1)

        if self.use_walk_region_spatial:
            self.walk_region_spatial = WalkRegionSpatialBlock(
                num_nodes=num_nodes,
                model_dim=self.model_dim,
                num_regions=walk_num_regions,
                walk_length=walk_length,
                num_walks=walk_num_walks,
                supports=supports,
                dropout=walk_region_dropout,
                dynamic_walk_weight=walk_dynamic_weight,
                region_gcn_layers=walk_region_gcn_layers,
            )
        else:
            self.walk_region_spatial = nn.Identity()

        self.mamba_memory_blocks = nn.ModuleList(
            [
                MambaMemoryBlock(
                    model_dim=self.model_dim,
                    sequence_length=input_steps,
                    num_nodes=num_nodes,
                    mlp_ratio=mlp_ratio,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout_rate=dropout_mamba,
                    memory_slots=memory_slots,
                    memory_dim=memory_dim,
                    attention_slots=attention_slots,
                    conv_stride=conv_stride,
                    conv_kernel=conv_kernel,
                    attention_dim=attention_dim,
                )
                for _ in range(num_mamba_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout_adaptive)

        self.encoder_projection = nn.Linear(input_steps * self.model_dim, self.model_dim)

        self.encoder_layers = nn.ModuleList(
            [
                Mlp(
                    in_features=self.model_dim,
                    hidden_features=int(self.model_dim * mlp_ratio),
                    act_layer=nn.ReLU,
                    drop=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Linear(self.model_dim, output_steps * output_dim)

    def _create_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]

        traffic_features = x[..., : self.input_dim]
        embedded_features = self.input_dataEnco(traffic_features)
        feature_list = [embedded_features]

        if self.tod_embedding_dim > 0:
            tod_index = torch.clamp((tod * self.steps_per_day).long(), min=0, max=self.steps_per_day - 1)
            feature_list.append(self.tod_embedding(tod_index))

        if self.dow_embedding_dim > 0:
            dow_index = torch.clamp(dow.long(), min=0, max=6)
            feature_list.append(self.dow_embedding(dow_index))

        if self.adaptive_embedding_dim > 0:
            adaptive_emb = self.adaptive_embedding.expand(batch_size, *self.adaptive_embedding.shape)
            feature_list.append(self.dropout(adaptive_emb))

        if self.spatial_embedding_dim > 0:
            batch, _, num_nodes, _ = x.shape
            spatial_indexs = torch.arange(num_nodes, dtype=torch.long, device=x.device)
            spatial_emb = self.spatial_embedding(spatial_indexs).unsqueeze(0).unsqueeze(1)
            feature_list.append(self.sp_drop(spatial_emb.repeat(batch, self.input_steps, 1, 1)))

        return torch.cat(feature_list, dim=-1)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        embedded_x = self._create_embeddings(x)
        embedded_x = self.walk_region_spatial(embedded_x)

        for mamba_block in self.mamba_memory_blocks:
            embedded_x = mamba_block(embedded_x)

        encoder_input = embedded_x.transpose(1, 2).flatten(-2)
        encoded_features = self.encoder_projection(encoder_input)

        for encoder_layer in self.encoder_layers:
            encoded_features = encoded_features + encoder_layer(encoded_features)

        output = self.output_projection(encoded_features).view(
            batch_size, self.num_nodes, self.output_steps, self.output_dim
        )
        return output.transpose(1, 2)


TrafficModel = TrafficForecastingModel
