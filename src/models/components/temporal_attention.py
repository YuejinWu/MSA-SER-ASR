"""
=== 全局时间注意力模块 (Temporal Attention) ===

设计动机：
    标准 HuBERT 输出为逐帧特征序列，不同时间帧对情感判别的重要性不同。
    本模块通过可学习的缩放点积注意力机制，自动学习每帧的重要性权重，
    生成全局上下文向量，同时输出注意力权重用于可解释性分析。

输入:  (batch, seq_len, hidden_dim)  — HuBERT 输出的帧级特征
输出:  (batch, hidden_dim)          — 注意力加权后的全局特征向量
       (batch, seq_len)             — 平均注意力权重（用于可视化）
"""

import torch
import torch.nn as nn
from typing import Tuple


class TemporalAttention(nn.Module):
    """
    全局时间注意力模块，基于缩放点积注意力机制。

    计算流程：
        1. 通过 Linear 层将输入特征投影到低维空间，生成 Query 和 Key
        2. 计算 QK^T 得到注意力分数矩阵，除以 sqrt(d_k) 缩放
        3. 对注意力分数做 softmax 归一化
        4. 以原始特征作为 Value，按注意力权重加权求和
        5. 对所有注意力头取平均，输出逐帧注意力权重
    """

    def __init__(self, hidden_dim: int = 768, attention_dim: int = 64):
        """
        Args:
            hidden_dim:  输入特征维度（HuBERT 为 768）
            attention_dim: Q/K 投影的目标维度（降低计算量）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # Q、K 投影层：将 768 维特征压缩到 64 维，加速注意力计算
        self.query_proj = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, attention_dim, bias=False)

        # 缩放因子：避免点积值过大导致 softmax 梯度消失
        self.scale = attention_dim ** 0.5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim) HuBERT 帧级特征

        Returns:
            context:    (batch, hidden_dim) 注意力加权后的全局特征
            attn_weights: (batch, seq_len) 平均逐帧注意力权重
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Step 1: 生成 Query 和 Key
        Q = self.query_proj(x)  # (batch, seq_len, attention_dim)
        K = self.key_proj(x)    # (batch, seq_len, attention_dim)

        # Step 2: 计算缩放点积注意力分数矩阵
        # Q @ K^T → (batch, seq_len, seq_len)，除以 sqrt(d_k) 缩放
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch, seq_len, seq_len)

        # Step 3: Softmax 归一化（每行和为 1）
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # (batch, seq_len, seq_len)

        # Step 4: 对 Value（原始特征）加权求和
        # 每个查询位置对所有 key 位置的 value 加权平均
        context = torch.matmul(attn_weights, x)
        # (batch, seq_len, hidden_dim)

        # 对序列维度取平均，得到全局上下文向量
        context = context.mean(dim=1)  # (batch, hidden_dim)

        # Step 5: 计算用于可视化的平均注意力权重
        # 对查询维度取平均，得到每个时间帧被"关注"的平均程度
        avg_attn = attn_weights.mean(dim=1)
        # (batch, seq_len)

        return context, avg_attn
