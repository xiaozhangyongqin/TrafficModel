from turtle import mode
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryMultiAttention(nn.Module):
    """
    Memory module with dynamic read/write operations.
    Supports multi-head attention for richer retrieval.
    
    Args:
        memory_slots (int): Number of memory slots
        memory_dim (int): Dimension of each memory slot
        num_heads (int): Number of attention heads
    """
    def __init__(self, memory_slots: int, memory_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_rate)

        # 初始化记忆矩阵
        self.memory_bank = nn.Parameter(torch.FloatTensor(memory_slots, memory_dim))
        nn.init.xavier_normal_(self.memory_bank)

        # QKV 投影
        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)

        # 写入门控
        self.write_gate = nn.Linear(memory_dim, memory_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, L, N, D] (batch, seq_len, num_nodes, dim)
        Returns:
            memory_aug: [B, L, N, D]
        """
        B, L, N, D = x.shape
        Q = self.query_proj(x).view(B, L, N, self.num_heads, D // self.num_heads)

        # --- Multi-head Attention Read ---
        K = self.key_proj(self.memory_bank).view(self.memory_slots, self.num_heads, D // self.num_heads)
        V = self.value_proj(self.memory_bank).view(self.memory_slots, self.num_heads, D // self.num_heads)

        # 注意力计算
        attn_scores = torch.einsum("blnhd,mhd->blnhm", Q, K) / (D ** 0.5)  # (B, L, H, N, M)

        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))  # 
        memory_read = torch.einsum("blnhm,mhd->blnhd", attn_weights, V)  # 
        memory_read = memory_read.reshape(B, L, N, D)

       # --- Gated Write (更新记忆) ---
        if self.training:
            # 计算全局输入表示用于记忆更新
            global_input = x.mean(dim=(0, 1, 2))  # (D,) 全局平均
            
            # 计算记忆使用统计
            with torch.no_grad():
                memory_usage = attn_weights.mean(dim=(0, 1, 2, 3))  # (M,) 每个记忆槽的平均使用率
                
            # 计算写入信号
            write_signal = torch.tanh(self.write_gate(global_input))  # (D,)
            
            # 选择性更新记忆库
            for i in range(self.memory_slots):
                if memory_usage[i] > 0.01:  # 只更新被使用的记忆槽
                    # 计算更新强度
                    update_strength = 0.1 * memory_usage[i]
                    
                    # 更新记忆
                    self.memory_bank.data[i] = (
                        (1 - update_strength) * self.memory_bank.data[i] + 
                        update_strength * write_signal
                    )

        return memory_read + x


class MemoryTransformerLayer(nn.Module):
    """
    Enhanced Spatial-Temporal Memory Network for traffic forecasting.
    Combines input projection + dynamic memory augmentation.
    """
    def __init__(
        self,
        model_dim: int = 64,
        memory_slots: int = 10,
        memory_dim: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        # 1. Memory-based Multi-Head Attention
        self.memory_attention = MemoryMultiAttention(
            memory_slots=memory_slots,
            memory_dim=model_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        # 2 ffn
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(model_dim * 2, model_dim)
        )
        
        # 3. Layer Normalization
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, L, N, D] input tensor
        Returns:
            out: [B, L, N, D]
        """
        atten_out = self.memory_attention(x)
        x = self.norm1(x + atten_out)

        ffn_out = self.feed_forward(x)
        x = self.norm2(x + ffn_out)
        return x


        

if __name__ == "__main__":
    model = MemoryTransformerLayer(
        model_dim=64,
        memory_slots=10,
        memory_dim=64,
        num_heads=4,
        dropout_rate=0.1
    )
    x = torch.randn(32, 12, 170, 64)
    out = model(x)
    print(out.shape)

