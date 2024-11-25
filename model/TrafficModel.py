import torch.nn as nn
import torch
import numpy as np
from mambaEncoder import mambaEncoder, ormamba
from timm.models.vision_transformer import Mlp
from positional_encodings.torch_encodings import PositionalEncoding2D
from Mem import STDMN, MSAM


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self):
        super().__init__()

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        tp_enc_2d = PositionalEncoding2D(num_feat).to(input_data.device)
        input_data += tp_enc_2d(input_data)
        return input_data


class MambaMem(nn.Module):

    def __init__(self, model_dim, T, num_nodes, mlp_ratio, d_state=8, d_conv=4,
                 expand=2, dropout_m=0.15, dropout=0.1, N_m=10, D_m=64, N_k =10, conv_stride=2, conv_k=3, dim_k=4):
        super().__init__()
        self.model_dim = model_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.mem = STDMN(
            T=T,
            num_nodes=num_nodes,
            model_dim=model_dim,
            mem_num=N_m,
            mem_dim=D_m
        )
        self.conv = MSAM(
            input_dim=model_dim,
            output_dim=model_dim,
            mem_num=N_k,
            conv_stride=conv_stride,
            conv_k=conv_k,
            dim_k=dim_k
        )
        self.mamba = mambaEncoder(
            model_dim=self.model_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout_m
        )
        self.ln1 = nn.LayerNorm(self.model_dim)
        self.ln2 = nn.LayerNorm(self.model_dim)

    def forward(self, x):
        batch_size, T, N, D = x.shape
        residual = x
        x = torch.cat([x, self.conv(x)], dim=1)
        x = self.mem(x) + residual
        x = self.mamba(x.reshape(batch_size, T * N, self.model_dim)).reshape(x.shape)
        return x


class TrafficModel(nn.Module):
    def __init__(
            self,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=12,
            dow_embedding_dim=12,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=12,
            supports=None,
            num_layers=3,
            dropout=0.1,
            mlp_ratio=2,
            use_mixed_proj=True,
            d_state=8,
            d_conv=4,
            expand=2,
            dropout_m=0.15,
            num_layers_ma=1,
            dropout_a=0.3,
            N_m=10,
            D_m=64,
            N_k =10,
            conv_stride=2,
            conv_k=3,
            dim_k=24
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        #         self.pos = PositionalEncoding()
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        self.mambaformer = nn.ModuleList(
            [
                MambaMem(
                    model_dim=self.model_dim,
                    T=in_steps,
                    num_nodes=num_nodes,
                    mlp_ratio=mlp_ratio,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout_m=dropout_m,
                    dropout=dropout,
                    N_m=N_m,
                    D_m=D_m,
                    N_k =N_k,
                    conv_stride=conv_stride,
                    conv_k=conv_k,
                    dim_k=dim_k
                )
                for _ in range(num_layers_ma)
            ]
        )

        self.dropout = nn.Dropout(dropout_a)

        self.encoder_proj = nn.Linear(
            in_steps * self.model_dim,
            self.model_dim,
        )

        self.encoder = nn.ModuleList(
            [
                Mlp(
                    in_features=self.model_dim,
                    hidden_features=int(self.model_dim * mlp_ratio),
                    act_layer=nn.ReLU,
                    drop=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(self.model_dim, out_steps * output_dim)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
            # T_i_D_emb1 = self.T_i_D_emb[(x[..., 0, 1] * 288).type(torch.LongTensor)]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
            # D_i_W_emb1 = self.D_i_W_emb[(x[..., 0, 2]).type(torch.LongTensor)]
        x = x[..., : self.input_dim]
        #         x = x + self.pos(x)

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = torch.tensor([]).to(x)
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features = torch.concat([features, tod_emb], -1)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features = torch.concat([features, dow_emb], -1)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features = torch.concat([features, self.dropout(adp_emb)], -1)
        x = torch.cat(
            [x] + [features], dim=-1
        )  # (batch_size, in_steps, num_nodes, model_dim)
        for mam in self.mambaformer:
            x = mam(x)
        #         print(x.shape)
        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
        for layer in self.encoder:
            x = x + layer(x)
        # (batch_size, in_steps, num_nodes, model_dim)

        out = self.output_proj(x).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        return out
