"""
声音能量曲线数据加载器 - 用于深度学习训练

将 tools/load_sound.py 加载的声音数据转换为深度学习格式：
- 输入：volume (能量) 和 density (密度) 曲线
- 输出：双通道张量 (2, L)，可直接用于 1D-CNN + ArcFace 训练
"""
import os
import json
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.load_sound import SoundDataLoader


class SoundDataset(Dataset):
    """
    声音能量曲线数据集
    
    从 xlsx 文件加载声音数据，转换为双通道格式 (volume, density)
    形状: (2, L)，其中 L 通常是 3000
    """
    
    def __init__(
        self,
        sound_loader: SoundDataLoader,
        sample_names: List[str],
        labels: np.ndarray,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
    ):
        """
        Args:
            sound_loader: SoundDataLoader 实例
            sample_names: 样本名称列表，如 ['97_Normal_0', '234_0', ...]
            labels: 对应的标签数组 (N,)
            channel_mean: (2,) 通道均值，用于标准化
            channel_std: (2,) 通道标准差，用于标准化
        """
        self.sound_loader = sound_loader
        self.sample_names = sample_names
        self.labels = labels.astype(np.int64)
        self.channel_mean = channel_mean.reshape(2, 1)
        self.channel_std = channel_std.reshape(2, 1)
        
    def __len__(self) -> int:
        return len(self.sample_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_name = self.sample_names[idx]
        label = self.labels[idx]
        
        # 加载声音曲线
        curves = self.sound_loader.load_sound_curves(sample_name)
        if curves is None:
            raise ValueError(f"无法加载样本: {sample_name}")
        
        # 提取 volume 和 density，堆叠为双通道 (2, L)
        volume = curves['volume']  # (L,)
        density = curves['density']  # (L,)
        
        # 确保长度一致
        min_len = min(len(volume), len(density))
        volume = volume[:min_len]
        density = density[:min_len]
        
        x = np.stack([volume, density], axis=0)  # (2, L)
        
        # Z-Score 标准化：按通道独立
        x = (x - self.channel_mean) / (self.channel_std + 1e-8)
        
        x_tensor = torch.from_numpy(x.astype(np.float32))
        y_tensor = torch.tensor(label, dtype=torch.long)
        
        return x_tensor, y_tensor


def _load_labels_from_metadata(
    sample_names: List[str],
    metadata_path: str = "cwru_processed/metadata.json",
) -> Optional[np.ndarray]:
    """
    从 metadata.json 中根据文件名匹配标签
    
    Args:
        sample_names: 样本名称列表，如 ['97_Normal_0', '234_0', ...]
        metadata_path: metadata.json 路径
        
    Returns:
        labels: (N,) 标签数组，如果无法匹配则返回 None
    """
    if not os.path.exists(metadata_path):
        print(f"Warning: metadata.json 不存在: {metadata_path}")
        return None
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 构建文件名 -> 标签的映射
    name_to_label = {}
    for item in metadata:
        filename = item['filename'].replace('.mat', '')
        label = item.get('fault_label', -1)
        name_to_label[filename] = label
    
    # 匹配样本名称到标签
    labels = []
    unmatched = []
    for name in sample_names:
        if name in name_to_label:
            labels.append(name_to_label[name])
        else:
            # 尝试去掉后缀匹配（如 '97_Normal_0' -> '97_Normal'）
            base_name = '_'.join(name.split('_')[:-1])
            if base_name in name_to_label:
                labels.append(name_to_label[base_name])
            else:
                unmatched.append(name)
                labels.append(-1)  # 未知标签
    
    if unmatched:
        print(f"Warning: {len(unmatched)} 个样本无法匹配标签: {unmatched[:5]}...")
    
    labels = np.array(labels)
    if (labels == -1).any():
        print(f"Warning: 存在 {np.sum(labels == -1)} 个未匹配的样本")
    
    return labels


def get_sound_dataloaders(
    batch_size: int = 64,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    sound_data_dir: str = "声音能量曲线数据",
    metadata_path: str = "cwru_processed/metadata.json",
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建基于声音数据的 train/val/test DataLoader
    
    Args:
        batch_size: 批大小
        split_ratio: (train, val, test) 比例
        sound_data_dir: 声音数据目录
        metadata_path: metadata.json 路径，用于获取标签
        num_workers: DataLoader 工作进程数
        shuffle: 是否打乱
        pin_memory: 是否使用 pin_memory
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 初始化声音数据加载器
    sound_loader = SoundDataLoader(sound_data_dir)
    available_samples = sound_loader.get_available_files()
    
    if len(available_samples) == 0:
        raise ValueError(f"未找到任何声音数据样本在目录: {sound_data_dir}")
    
    print(f"找到 {len(available_samples)} 个声音数据样本")
    
    # 从 metadata 加载标签
    labels = _load_labels_from_metadata(available_samples, metadata_path)
    if labels is None:
        raise ValueError("无法从 metadata.json 加载标签，请检查文件路径")
    
    # 过滤掉标签为 -1 的样本（未匹配的）
    valid_mask = labels != -1
    valid_samples = [s for s, m in zip(available_samples, valid_mask) if m]
    valid_labels = labels[valid_mask]
    
    if len(valid_samples) == 0:
        raise ValueError("没有有效的样本（所有样本都无法匹配标签）")
    
    print(f"有效样本数: {len(valid_samples)} (已过滤 {len(available_samples) - len(valid_samples)} 个未匹配样本)")
    
    n_samples = len(valid_samples)
    
    # 计算各子集大小
    train_ratio, val_ratio, test_ratio = split_ratio
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"split_ratio 之和必须为 1，当前为: {split_ratio}")
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    # 生成打乱后的索引
    all_indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed=42)
        rng.shuffle(all_indices)
    
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train : n_train + n_val]
    test_indices = all_indices[n_train + n_val :]
    
    # 在训练集上计算通道均值/方差（需要先加载数据）
    print("计算训练集统计量...")
    train_volumes = []
    train_densities = []
    
    for idx in train_indices:
        sample_name = valid_samples[idx]
        curves = sound_loader.load_sound_curves(sample_name)
        if curves is not None:
            train_volumes.append(curves['volume'])
            train_densities.append(curves['density'])
    
    if len(train_volumes) == 0:
        raise ValueError("训练集为空，无法计算统计量")
    
    # 计算均值和标准差
    all_volumes = np.concatenate(train_volumes)
    all_densities = np.concatenate(train_densities)
    
    channel_mean = np.array([np.mean(all_volumes), np.mean(all_densities)])
    channel_std = np.array([np.std(all_volumes), np.std(all_densities)])
    
    print(f"通道均值: {channel_mean}")
    print(f"通道标准差: {channel_std}")
    
    # 构建数据集
    train_samples = [valid_samples[i] for i in train_indices]
    val_samples = [valid_samples[i] for i in val_indices]
    test_samples = [valid_samples[i] for i in test_indices]
    
    train_labels = valid_labels[train_indices]
    val_labels = valid_labels[val_indices]
    test_labels = valid_labels[test_indices]
    
    train_dataset = SoundDataset(sound_loader, train_samples, train_labels, channel_mean, channel_std)
    val_dataset = SoundDataset(sound_loader, val_samples, val_labels, channel_mean, channel_std)
    test_dataset = SoundDataset(sound_loader, test_samples, test_labels, channel_mean, channel_std)
    
    # DataLoader 通用参数（num_workers>0 时保持 worker 进程不退出）
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader


__all__ = ["SoundDataset", "get_sound_dataloaders"]

