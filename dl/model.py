import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    """1D ResNet 的 BasicBlock，将 Conv2d/BN2d 全部替换为 1D 版本。"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18_1D_Backbone(nn.Module):
    """
    修改版 1D ResNet-18 作为特征提取骨干网络。

    输入: (B, 2, L)
    输出: 512 维特征向量 (B, 512)，不包含分类层。
    """

    def __init__(self, in_channels: int = 2, embedding_dim: int = 512):
        super().__init__()
        self.in_planes = 64

        # stem
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet-18 四个 stage
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        # 全局池化 + 线性投影到 512 维 embedding
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Linear(512 * BasicBlock1D.expansion, embedding_dim, bias=False)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_planes,
                    planes * BasicBlock1D.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * BasicBlock1D.expansion),
            )

        layers = [BasicBlock1D(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * BasicBlock1D.expansion

        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, L) 双通道李群特征序列
        Returns:
            embedding: (B, 512) L2 前的特征向量
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局池化： (B, C, L') -> (B, C, 1)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # (B, C)

        x = self.embedding(x)  # (B, embedding_dim)
        x = self.bn_embedding(x)

        return x


def build_backbone(in_channels: int = 2, embedding_dim: int = 512) -> nn.Module:
    """
    便捷构造函数，供训练和推理脚本使用。
    """
    return ResNet18_1D_Backbone(in_channels=in_channels, embedding_dim=embedding_dim)


__all__ = ["BasicBlock1D", "ResNet18_1D_Backbone", "build_backbone"]


