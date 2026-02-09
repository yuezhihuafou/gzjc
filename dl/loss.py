import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    ArcFace / Additive Angular Margin Loss 的 Logits 计算模块。

    数学流程:
        1. 对特征 emb 和 权重 W 做 L2 归一化
        2. 计算余弦相似度: cosine = emb_norm @ W_norm^T
        3. 对真实类别对应的 cosine 做:
               theta = arccos(cosine)
               theta_m = theta + m
               cosine_m = cos(theta_m)
        4. 将替换后的 cosine_m 放回原 logits 中
        5. 乘以缩放因子 s，返回给 CrossEntropyLoss 使用
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb:   (B, in_features) 特征向量
            label: (B,)           long 型标签
        Returns:
            logits: (B, out_features) 供 CrossEntropyLoss 使用
        """
        # 1. L2 归一化
        emb_norm = F.normalize(emb, p=2, dim=1)
        W_norm = F.normalize(self.weight, p=2, dim=1)

        # 2. 余弦相似度
        cosine = F.linear(emb_norm, W_norm)  # (B, C)

        # 3. 添加角度间隔
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # 只在真实类别上替换为 phi
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


__all__ = ["ArcMarginProduct"]


