"""
=== 原型对比学习模块 (Prototype Layer) ===

设计动机：
    传统交叉熵损失只关注样本级别的分类正确性，忽略了类内紧凑性和类间分离性。
    原型对比学习（PCL）通过维护一组可学习的类别原型向量，将样本特征拉向其
    对应类别的原型，同时推离其他类别原型，从而在特征空间中形成更清晰的决策边界。

    原型更新策略：
        每个 epoch 结束后，在训练集上计算每类样本的融合特征均值，
        使用指数移动平均（EMA）更新原型，避免单批次噪声干扰。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PrototypeLayer(nn.Module):
    """
    原型对比学习层，维护 num_classes 个原型向量。

    原型向量的初始值为全零，随着训练进行通过 EMA 方式
    逐步逼近各类别样本在特征空间中的聚类中心。
    """

    def __init__(
        self,
        num_classes: int = 8,
        hidden_dim: int = 768,
        temperature: float = 0.1,
        momentum: float = 0.9,
    ):
        """
        Args:
            num_classes:  情感类别数（RAVDESS 为 8）
            hidden_dim:   特征维度（熔合后为 768）
            temperature:  PCL 温度系数 τ（越小，分布越尖锐）
            momentum:     EMA 更新动量系数
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.momentum = momentum

        # 原型向量：(num_classes, hidden_dim)，使用 buffer 存储（不参与梯度优化）
        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, hidden_dim),
        )
        # 标记原型是否已被初始化（第一次更新后置 True）
        self.register_buffer("prototypes_initialized", torch.tensor(False))

    def compute_pcl_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算原型对比学习损失 L_PCL。

        公式：
            L_PCL = -1/N * Σ_i log( exp(sim(z_i, p_{y_i}) / τ) /
                                  Σ_j exp(sim(z_i, p_j) / τ) )
        其中 sim(a, b) = a · b / (||a|| * ||b||) 为余弦相似度。

        Args:
            features: (batch, hidden_dim) 融合特征向量
            labels:   (batch,) 标签 [0, num_classes-1]

        Returns:
            pcl_loss: 标量损失值
        """
        if not self.prototypes_initialized:
            # 原型尚未初始化，返回零损失，让模型先通过 CE 学习基本特征
            return torch.tensor(0.0, device=features.device)

        # L2 归一化特征和原型，使点积等价于余弦相似度
        features_norm = F.normalize(features, p=2, dim=-1)       # (batch, hidden_dim)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)  # (num_classes, hidden_dim)

        # 计算余弦相似度矩阵（点积形式，因为已归一化）
        # sim_matrix[i, j] = cosine_sim(z_i, p_j)
        sim_matrix = torch.matmul(features_norm, prototypes_norm.T)
        # (batch, num_classes)

        # 除以温度系数 τ，控制分布的平滑程度
        sim_matrix = sim_matrix / self.temperature

        # 计算对比损失：每个样本的正样本是它所属类别的原型
        # log_softmax 等价于 log(exp(sim) / sum(exp(sim)))
        log_probs = F.log_softmax(sim_matrix, dim=-1)

        # 取出每个样本对应其真实标签的对数概率
        pcl_loss = -log_probs[range(len(labels)), labels].mean()

        return pcl_loss

    def update_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        momentum: float | None = None,
    ):
        """
        使用 EMA 更新原型向量。

        更新公式：
            p_c^{new} = momentum * p_c^{old} + (1 - momentum) * mean(z ∈ class c)

        Args:
            features: (N, hidden_dim) 训练集的融合特征
            labels:   (N,) 标签
            momentum: EMA 动量；若为 None，使用构造时的默认值
        """
        if momentum is None:
            momentum = self.momentum

        with torch.no_grad():
            for c in range(self.num_classes):
                # 筛选当前类别的样本特征
                class_mask = (labels == c)
                if class_mask.sum() == 0:
                    continue  # 该类别无样本，跳过

                class_features = features[class_mask]  # (n_c, hidden_dim)
                # 计算类别均值
                class_mean = class_features.mean(dim=0)  # (hidden_dim,)

                if self.prototypes_initialized:
                    # EMA 更新：平滑地移动原型
                    self.prototypes[c] = (
                        momentum * self.prototypes[c]
                        + (1 - momentum) * class_mean
                    )
                else:
                    # 首次更新：直接赋值
                    self.prototypes[c] = class_mean

            # 标记原型已初始化
            self.prototypes_initialized = torch.tensor(True, device=self.prototypes.device)
