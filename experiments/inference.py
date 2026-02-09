import os
import sys
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dl.data_loader import get_dataloaders
from dl.sound_data_loader import get_sound_dataloaders
from dl.model import build_backbone

# 类别 id → 可读名（与 train.py / CWRU 约定一致）
CLASS_ID_TO_NAME = {0: "Normal", 1: "B", 2: "IR", 3: "OR"}


@torch.no_grad()
def compute_class_centroids(
    backbone: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在给定的数据集上计算每个类别的特征中心 (L2 归一化后)。
    返回:
        centroids: (C, D)
        labels:    (C,) 对应的类别 id
    """
    backbone.eval()

    all_feats = []
    all_labels = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        feats = backbone(x)               # (B, D)
        feats = F.normalize(feats, dim=1)  # L2 归一化

        all_feats.append(feats)
        all_labels.append(y)

    all_feats = torch.cat(all_feats, dim=0)   # (N, D)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)

    unique_labels = all_labels.unique()
    centroids = []
    for c in unique_labels:
        mask = all_labels == c
        centroid = all_feats[mask].mean(dim=0)
        centroid = F.normalize(centroid, dim=0)
        centroids.append(centroid)

    centroids = torch.stack(centroids, dim=0)  # (C, D)
    return centroids, unique_labels.cpu()


@torch.no_grad()
def predict_with_centroids(
    backbone: torch.nn.Module,
    centroids: torch.Tensor,
    centroid_labels: torch.Tensor,
    x: torch.Tensor,
    threshold: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用类别中心进行 open-set 推理。
    Args:
        backbone: 训练好的骨干网络
        centroids: (C, D) L2 归一化后的中心
        centroid_labels: (C,) 中心对应的类别标签
        x: (B, 2, L) 输入信号
        threshold: 余弦相似度阈值
    Returns:
        pred_labels: (B,) 预测标签，未知类记为 -1
        max_sim:     (B,) 最大相似度
    """
    device = next(backbone.parameters()).device
    backbone.eval()

    x = x.to(device)
    feats = backbone(x)
    feats = F.normalize(feats, dim=1)  # (B, D)

    # 余弦相似度: (B, D) @ (D, C) = (B, C)
    sims = torch.matmul(feats, centroids.t())
    max_sim, idx = sims.max(dim=1)

    pred_labels = centroid_labels[idx].to(device)
    # 低于阈值的标记为未知类 -1
    unknown_mask = max_sim < threshold
    pred_labels = pred_labels.clone()
    pred_labels[unknown_mask] = -1

    return pred_labels.cpu(), max_sim.cpu()


def main():
    """
    示例：加载训练好的 Backbone，基于训练集计算类别中心，
    然后在测试集上做 open-set 推理，并给出未知类判别。
    """
    parser = argparse.ArgumentParser(description='使用训练好的模型进行推理')
    parser.add_argument('--data_source', type=str, choices=['cwru', 'sound'], default='cwru',
                        help='数据源: cwru 或 sound')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批大小 (默认: 256 for cwru, 8 for sound)')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='余弦相似度阈值，低于此值判为 Unknown')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/backbone.pth',
                        help='Backbone 权重文件路径')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"数据源: {args.data_source}")

    # 根据数据源设置默认 batch_size
    if args.batch_size is None:
        batch_size = 8 if args.data_source == 'sound' else 256
    else:
        batch_size = args.batch_size

    # 加载数据
    if args.data_source == 'sound':
        print("\n加载声音能量曲线数据...")
        train_loader, val_loader, test_loader = get_sound_dataloaders(
            batch_size=batch_size, shuffle=True
        )
    else:
        print("\n加载 CWRU 处理后的数据...")
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=batch_size, shuffle=True
        )

    # 构建并加载骨干网络
    backbone = build_backbone(in_channels=2, embedding_dim=512).to(device)
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"未找到 backbone 权重文件: {ckpt_path}，请先运行 train.py 进行训练并保存模型。"
        )
    backbone.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"已加载模型权重: {ckpt_path}")

    # 1. 计算训练集上的类别中心
    centroids, centroid_labels = compute_class_centroids(backbone, train_loader, device)
    print("类别中心已计算完毕。")
    # 输出类别 id 与可读名
    labels_np = centroid_labels.numpy()
    names_str = ", ".join(f"{int(l)}={CLASS_ID_TO_NAME.get(int(l), str(int(l)))}" for l in labels_np)
    print(f"类别映射: {names_str}")

    # 2. 在测试集上做 open-set 推理示意
    total = 0
    known_correct = 0
    unknown_count = 0

    with torch.no_grad():
        for x, y in test_loader:
            preds, sims = predict_with_centroids(
                backbone,
                centroids,
                centroid_labels,
                x,
                threshold=args.threshold,
            )

            # 统计已知类的分类准确率
            known_mask = preds != -1
            known_correct += (preds[known_mask] == y[known_mask]).sum().item()
            unknown_count += (preds == -1).sum().item()
            total += y.size(0)

    print(f"测试样本总数: {total}")
    print(f"判为 Unknown 的数量: {unknown_count}")
    if total - unknown_count > 0:
        print(
            f"在被判为已知类的样本中，分类准确率: "
            f"{known_correct / (total - unknown_count):.4f}"
        )


if __name__ == "__main__":
    main()


