
"""
Traffic Forecasting Model with Mamba and Memory Augmentation

This module implements a traffic forecasting model that combines:
- Mamba (State Space Model) for sequence modeling
- Memory augmentation for capturing long-term dependencies
- Multi-scale attention for spatial-temporal feature extraction

Author: Traffic Model Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp

from .mambaEncoder import Mamba, MambaConfig
from .Mem import SpatialTemporalMemory, MultiScaleAttentionMemory
from .MemoryTransformer import MemoryTransformerLayer
import math
# from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D

class MambaMemoryBlock(nn.Module):
    """
    Mamba block with memory augmentation for traffic forecasting.
    
    This block combines Mamba's state space modeling capabilities with
    memory augmentation to capture both short-term and long-term dependencies
    in traffic data.
    
    Args:
        model_dim (int): Model dimension
        sequence_length (int): Input sequence length
        num_nodes (int): Number of traffic nodes
        mlp_ratio (float): MLP expansion ratio
        d_state (int): State dimension for Mamba
        d_conv (int): Convolution kernel size for Mamba
        expand (int): Expansion factor for Mamba
        dropout_rate (float): Dropout rate
        memory_slots (int): Number of memory slots
        memory_dim (int): Memory dimension
        attention_slots (int): Number of attention memory slots
        conv_stride (int): Convolution stride
        conv_kernel (int): Convolution kernel size
        attention_dim (int): Attention dimension
    """
    
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
        attention_dim: int = 4
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        
        # Memory components
        self.spatial_temporal_memory = SpatialTemporalMemory(
            sequence_length=sequence_length,
            num_nodes=num_nodes,
            model_dim=model_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim
        )
        
        self.multi_scale_attention = MultiScaleAttentionMemory(
            input_dim=model_dim,
            output_dim=model_dim,
            memory_slots=attention_slots,
            max_scales = 5,
            attention_dim=attention_dim
        )
        self.mem_att = MemoryTransformerLayer(
            model_dim=model_dim,
            memory_slots=attention_slots,
            memory_dim=model_dim,
            num_heads=1,
            dropout_rate=0.1
        )
        
        # Mamba encoder
        mamba_config = MambaConfig(
            d_model=model_dim,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand
        )
        self.mamba_encoder = Mamba(mamba_config)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Mamba memory block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_nodes, model_dim)
            
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        batch_size, seq_len, num_nodes, model_dim = x.shape
        
        # Store residual connection
        residual = x
        
        x_memory = self.mem_att(x)
        x_memory = self.layer_norm1(x_memory) + residual
        # Apply Mamba encoder
        # Reshape for Mamba: (batch_size, seq_len * num_nodes, model_dim)
        x_reshaped = (x_memory).reshape(batch_size, seq_len * num_nodes, model_dim)
        # x_reshaped = x.reshape(batch_size, seq_len * num_nodes, model_dim)
        mamba_output = self.mamba_encoder(x_reshaped)
        mamba_output = mamba_output.reshape(batch_size, seq_len, num_nodes, model_dim)
        
        # Combine memory and Mamba outputs
        output = mamba_output + residual 
        # output = x_memory
        return output

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
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()
class LearnableEMA(nn.Module):
    def __init__(self, T):
        super().__init__()
        # 可学习的衰减率（log-space 防止负值）
        self.log_alpha = nn.Parameter(torch.zeros(1))  # 控制记忆长度

    def forward(self, x):
        # x: [B, T, N, F]
        T = x.shape[1]
        
        # 将 alpha 限制在 (0,1)
        alpha = torch.sigmoid(self.log_alpha)  # 学习到的平滑系数

        # 构造指数权重：[w0, w1, ..., w_{T-1}]
        # wt = alpha * (1-alpha)^t
        t = torch.arange(T, device=x.device).float()  # [T]
        weights = alpha * (1 - alpha + 1e-6) ** t     # [T]
        weights = weights.flip(dims=[0])              # 倒序，最近的权重最大
        weights = weights / weights.sum()             # 归一化

        # 加权平均：对时间维度加权
        recent_summary = torch.sum(x * weights.view(1, T, 1, 1), dim=1, keepdim=True)
        
        return recent_summary.repeat([1, T, 1, 1])
class DataEncoding(nn.Module):
    def __init__(self, in_dim, hid_dim, timelr=True, activation='relu'):
        super().__init__()
        assert activation in ['gelu', 'relu']
        self.learnable_ema = LearnableEMA(T=12)
        self.timelr =timelr
        in_units = in_dim*3 if timelr else in_dim
        self.linear1 = nn.Linear(in_units, hid_dim)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        # x_ema = self.learnable_ema(x)
        if self.timelr:
            latestX = x[:, -1:, :, :].repeat([1, 12, 1, 1])
            recent_avg = x[:, -3:, :, :].mean(dim=1, keepdim=True).repeat([1, 12, 1, 1])
            data = torch.cat([x, recent_avg, latestX], dim=-1)
        else:
            data = x
        data = self.linear1(data)
        data = self.activation(data)
        data = self.linear2(data)
        return data

class TrafficForecastingModel(nn.Module):
    """
    Traffic Forecasting Model with Mamba and Memory Augmentation.
    
    This model combines multiple components for accurate traffic forecasting:
    - Input embeddings for traffic features
    - Time-of-day and day-of-week embeddings
    - Adaptive embeddings for node-specific patterns
    - Mamba-based sequence modeling with memory augmentation
    - Multi-layer perceptron for final prediction
    
    Args:
        num_nodes (int): Number of traffic nodes
        input_steps (int): Number of input time steps
        output_steps (int): Number of output time steps
        steps_per_day (int): Number of time steps per day
        input_dim (int): Input feature dimension
        output_dim (int): Output feature dimension
        input_embedding_dim (int): Input embedding dimension
        tod_embedding_dim (int): Time-of-day embedding dimension
        dow_embedding_dim (int): Day-of-week embedding dimension
        spatial_embedding_dim (int): Spatial embedding dimension
        adaptive_embedding_dim (int): Adaptive embedding dimension
        supports (list, optional): Graph adjacency matrices
        num_layers (int): Number of MLP layers
        dropout_rate (float): Dropout rate
        mlp_ratio (float): MLP expansion ratio
        use_mixed_proj (bool): Whether to use mixed projection
        d_state (int): Mamba state dimension
        d_conv (int): Mamba convolution kernel size
        expand (int): Mamba expansion factor
        dropout_mamba (float): Mamba dropout rate
        num_mamba_layers (int): Number of Mamba layers
        dropout_adaptive (float): Adaptive embedding dropout rate
        memory_slots (int): Number of memory slots
        memory_dim (int): Memory dimension
        attention_slots (int): Number of attention memory slots
        conv_stride (int): Convolution stride
        conv_kernel (int): Convolution kernel size
        attention_dim (int): Attention dimension
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
        attention_dim: int = 24
    ):
        super().__init__()
        
        # Store configuration
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
        
        # Calculate total model dimension
        self.model_dim = (
            input_embedding_dim +
            tod_embedding_dim +
            dow_embedding_dim +
            spatial_embedding_dim +
            adaptive_embedding_dim
        )
        # self.model_dim = input_embedding_dim
        
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        
        # Input projection
        # self.input_projection = nn.Linear(input_dim, input_embedding_dim)
        self.input_dataEnco = DataEncoding(input_dim, input_embedding_dim)

        
        # Time embeddings
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
            
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
            
        # Adaptive embeddings
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(input_steps, num_nodes, adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        # Spatial embeddings
        if spatial_embedding_dim > 0:
            self.spatial_embedding = nn.Embedding(num_nodes, spatial_embedding_dim)
            self.sp_drop = nn.Dropout(0.1)
            
        # Mamba memory blocks
        # MMB = MambaMemoryBlock(
        #         model_dim=self.model_dim,
        #         sequence_length=input_steps,
        #         num_nodes=num_nodes,
        #         mlp_ratio=mlp_ratio,
        #         d_state=d_state,
        #         d_conv=d_conv,
        #         expand=expand,
        #         dropout_rate=dropout_mamba,
        #         memory_slots=memory_slots,
        #         memory_dim=memory_dim,
        #         attention_slots=attention_slots,
        #         conv_stride=conv_stride,
        #         conv_kernel=conv_kernel,
        #         attention_dim=attention_dim
        #     )
        self.mamba_memory_blocks = nn.ModuleList([
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
                attention_dim=attention_dim
            )
            for _ in range(num_mamba_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_adaptive)
        
        # Encoder projection
        self.encoder_projection = nn.Linear(
            input_steps * self.model_dim,
            self.model_dim
        )
        
        # MLP encoder layers
        self.encoder_layers = nn.ModuleList([
            Mlp(
                in_features=self.model_dim,
                hidden_features=int(self.model_dim * mlp_ratio),
                act_layer=nn.ReLU,
                drop=dropout_rate
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(
            self.model_dim,
            output_steps * output_dim
        )
        
    def _create_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create embeddings from input features.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_steps, num_nodes, input_dim+2)
            
        Returns:
            torch.Tensor: Embedded features of shape (batch_size, input_steps, num_nodes, model_dim)
        """
        batch_size = x.shape[0]
        
        # Extract time features
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]  # Time of day
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]  # Day of week
            
        # Extract traffic features
        traffic_features = x[..., :self.input_dim]
        
        # Project traffic features
        embedded_features = self.input_dataEnco(traffic_features)
        
        # Create feature list
        feature_list = [embedded_features]
        
        # Add time-of-day embedding
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
            feature_list.append(tod_emb)
            
        # Add day-of-week embedding
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow.long())
            feature_list.append(dow_emb)

            
        # Add adaptive embedding
        if self.adaptive_embedding_dim > 0:
            adaptive_emb = self.adaptive_embedding.expand(
                batch_size, *self.adaptive_embedding.shape
            )
            feature_list.append(self.dropout(adaptive_emb))
        
        # Add Spatial Embdediding
        if self.spatial_embedding_dim > 0:
            batch, _,  num_nodes, _ = x.shape
            spatial_indexs = torch.LongTensor(torch.arange(num_nodes)).to(x.device)  # (N,)
            spatial_emb = self.spatial_embedding(spatial_indexs).unsqueeze(0).unsqueeze(1)  # (1, 1, N, spatial_embedding_dim)
            feature_list.append(self.sp_drop(spatial_emb.repeat(batch, self.input_steps, 1, 1)))
            # feature_list.append(self.sp_drop(spatial_emb))

        # Concatenate all features
        return torch.cat(feature_list, dim=-1)
        # return sum(feature_list)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the traffic forecasting model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_steps, num_nodes, input_dim+2)
            
        Returns:
            torch.Tensor: Predicted traffic of shape (batch_size, output_steps, num_nodes, output_dim)
        """
        batch_size = x.shape[0]
        
        # Create embeddings
        embedded_x = self._create_embeddings(x)

        # Apply Mamba memory blocks
        for mamba_block in self.mamba_memory_blocks:
            embedded_x = mamba_block(embedded_x)
            
        # Project to encoder dimension
        # Reshape: (batch_size, num_nodes, input_steps * model_dim)
        encoder_input = embedded_x.transpose(1, 2).flatten(-2)
        encoded_features = self.encoder_projection(encoder_input)
        
        # Apply encoder layers with residual connections
        for encoder_layer in self.encoder_layers:
            encoded_features = encoded_features + encoder_layer(encoded_features)
            
        # Generate output predictions
        output = self.output_projection(encoded_features).view(
            batch_size, self.num_nodes, self.output_steps, self.output_dim
        )
        
        # Transpose to (batch_size, output_steps, num_nodes, output_dim)
        return output.transpose(1, 2)


# Alias for backward compatibility
TrafficModel = TrafficForecastingModel
