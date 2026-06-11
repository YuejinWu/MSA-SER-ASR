"""
=== 原型对比学习损失函数 ===

核心公式（SupCon 风格）：
    L_PCL(z, p, y) = -log( exp(sim(z, p_y) / τ) / Σ_i exp(sim(z, p_i) / τ) )

其中：
    - z: 样本的融合特征（已 L2 归一化）
    - p: 原型向量（已 L2 归一化）
    - y: 样本标签
    - τ: 温度系数（temperature），控制分布锐度
    - sim(a, b) = a · b / (||a|| * ||b||) = 余弦相似度

设计动机：
    通过将特征拉向其类别原型，同时推离其他类别原型，
    在特征空间中形成类内紧凑、类间分离的表示结构，
    提高分类边界清晰度和泛化能力。
"""

import torch
import torch.nn.functional as F


def prototype_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    计算原型对比学习损失。

    Args:
        features:    (batch, hidden_dim) 样本特征向量
        labels:      (batch,) 整数标签 [0, num_classes-1]
        prototypes:  (num_classes, hidden_dim) 原型向量
        temperature: 温度系数 τ，默认 0.1

    Returns:
        loss: 标量损失值
    """
    # L2 归一化：使向量点积等于余弦相似度
    features_norm = F.normalize(features, p=2, dim=-1)
    prototypes_norm = F.normalize(prototypes, p=2, dim=-1)

    # 余弦相似度矩阵: sim[i, j] = cos(feature_i, prototype_j)
    sim_matrix = torch.matmul(features_norm, prototypes_norm.T)
    # (batch, num_classes)

    # 除以温度系数
    sim_matrix = sim_matrix / temperature

    # Log-Softmax: log( exp(sim) / Σ exp(sim) )
    log_probs = F.log_softmax(sim_matrix, dim=-1)

    # 取每个样本对应其真实类别的对数概率
    # log_probs[i, labels[i]] → 正样本的对数概率
    loss = -log_probs[range(len(labels)), labels].mean()

    return loss
