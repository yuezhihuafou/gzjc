import os
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class LieGroupDataset(Dataset):
    """
    李群特征数据集

    默认从 `cwru_processed/signals.npy` 与 `cwru_processed/labels.npy` 加载数据，
    形状假定为:
        signals: (N, 2, L)
        labels:  (N,)

    按照规范，对每个样本进行 **按通道独立的 Z-Score 标准化**:
        x_norm[c] = (x[c] - mean[c]) / std[c]
    其中 mean/std 在 **训练子集** 上按通道统计得到，并在 val/test 上复用。
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
    ) -> None:
        super().__init__()

        self.signals = signals
        self.labels = labels.astype(np.int64)
        self.indices = indices.astype(np.int64)

        # (2,) 通道均值与方差
        self.channel_mean = channel_mean.reshape(2, 1)
        self.channel_std = channel_std.reshape(2, 1)

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        x = self.signals[real_idx]  # (2, L)
        y = self.labels[real_idx]

        # Z-Score 标准化：按通道独立
        x = (x - self.channel_mean) / (self.channel_std + 1e-8)

        x_tensor = torch.from_numpy(x.astype(np.float32))
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor


def _load_numpy_arrays(
    base_dir: str = "cwru_processed",
    signals_name: str = "signals.npy",
    labels_name: str = "labels.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从指定目录加载 numpy 数据。
    """
    signals_path = os.path.join(base_dir, signals_name)
    labels_path = os.path.join(base_dir, labels_name)

    if not os.path.exists(signals_path):
        raise FileNotFoundError(f"signals 文件不存在: {signals_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels 文件不存在: {labels_path}")

    signals = np.load(signals_path)  # 期望形状: (N, 2, L)
    labels = np.load(labels_path)    # 期望形状: (N,)

    # 兼容旧版/单通道数据：(N, L) 或 (N, 1, L) -> (N, 2, L)
    if signals.ndim == 2:
        signals = np.stack([signals, signals], axis=1)
    elif signals.ndim == 3 and signals.shape[1] == 1:
        signals = np.concatenate([signals, signals], axis=1)

    if signals.ndim != 3 or signals.shape[1] != 2:
        raise ValueError(
            f"signals 期望形状为 (N, 2, L)，实际为: {signals.shape}"
        )
    if labels.ndim != 1 or labels.shape[0] != signals.shape[0]:
        raise ValueError(
            f"labels 形状与 signals 不匹配: signals={signals.shape}, labels={labels.shape}"
        )

    return signals, labels


def get_dataloaders(
    batch_size: int = 64,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    base_dir: str = "cwru_processed",
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 train/val/test 三个 DataLoader。

    标准化策略:
        - 先在训练子集上计算通道维度的 mean/std:
            mean_c = mean(signals_train[:, c, :])
            std_c  = std(signals_train[:, c, :])
        - 所有子集都使用同一组 mean/std。
    """
    signals, labels = _load_numpy_arrays(base_dir=base_dir)

    n_samples = signals.shape[0]

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

    # 在训练集上计算通道均值 / 方差
    # signals_train: (N_train, 2, L)
    signals_train = signals[train_indices]
    # 沿 (样本, 长度) 维度求平均 -> (2,)
    channel_mean = signals_train.mean(axis=(0, 2))
    channel_std = signals_train.std(axis=(0, 2))

    # 构建 Dataset
    train_dataset = LieGroupDataset(signals, labels, train_indices, channel_mean, channel_std)
    val_dataset = LieGroupDataset(signals, labels, val_indices, channel_mean, channel_std)
    test_dataset = LieGroupDataset(signals, labels, test_indices, channel_mean, channel_std)

    # DataLoader 通用参数（num_workers>0 时保持 worker 进程不退出，减少每 epoch 重启开销）
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

    return train_loader, val_loader, test_loader


def _filter_samples_by_fault_label(samples: list) -> list:
    """过滤掉 fault_label==-1 的样本，保证 cls 任务仅用有标签样本（与 NPZ 一一对应）。"""
    filtered = []
    for s in samples:
        npz_path = s['npz_path']
        try:
            with np.load(npz_path) as data:
                lb = int(data['fault_label']) if 'fault_label' in data else -1
        except Exception:
            lb = -1
        if lb >= 0:
            filtered.append(s)
    return filtered


def get_sound_api_cache_dataloaders(
    batch_size: int = 64,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    cache_dir: str = "datasets/sound_api/cache_npz",
    index_path: Optional[str] = None,
    task: str = 'hi',
    horizon: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建基于 sound_api_cache 的 train/val/test DataLoader
    
    使用 bearing-level split：按 bearing_id 分组切分，保证同一 bearing 的所有样本在同一子集。
    task=='cls' 时会过滤掉 fault_label==-1 的样本，保证分类标签与 NPZ 一一对应。
    
    Args:
        batch_size: 批大小
        split_ratio: (train, val, test) 比例
        cache_dir: NPZ 缓存目录
        index_path: index.csv 路径（可选，如果提供则从索引加载，否则扫描目录）
        task: 任务类型 ('hi', 'risk', 'cls')
        horizon: 风险预测的时间窗口（仅用于 risk 任务）
        num_workers: DataLoader 工作进程数
        pin_memory: 是否使用 pin_memory
        seed: 随机种子（用于 bearing-level split）
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from dl.sound_api_cache_dataset import (
        SoundAPICacheDataset,
        load_samples_from_index,
        load_samples_from_cache_dir,
        split_by_bearing,
        resolve_fault_label_from_f_source,
    )
    
    # 加载样本列表
    if index_path and os.path.exists(index_path):
        print(f"从索引文件加载: {index_path}")
        samples = load_samples_from_index(index_path, cache_dir)
    else:
        print(f"扫描缓存目录: {cache_dir}")
        samples = load_samples_from_cache_dir(cache_dir)
    
    if len(samples) == 0:
        raise ValueError(f"未找到任何样本在目录: {cache_dir}")
    
    print(f"找到 {len(samples)} 个样本")
    
    # task=='cls' 时用思路 B 从 NPZ→JSON→.f→sidecar 无损还原标签，再过滤
    if task == 'cls':
        print("cls 任务：通过 NPZ→JSON→.f→sidecar 反推补回标签（思路 B）...")
        for s in samples:
            s['fault_label'] = resolve_fault_label_from_f_source(s['npz_path'])
        n_before = len(samples)
        samples = [s for s in samples if s['fault_label'] >= 0]
        dropped = n_before - len(samples)
        if dropped > 0:
            print(f"  已排除无法从 .f sidecar 解析标签的样本: {dropped} 个")
        if len(samples) == 0:
            raise ValueError(
                "无有效分类标签。请确认 NPZ 由 build_sound_api_cache 从 convert_mc_to_api_json 的 JSON 构建，"
                "且 .f 侧有 sidecar .json 含 health_label/fault_label。"
            )
        print(f"  保留有标签样本: {len(samples)} 个")
    
    # bearing-level split（打印分配概况）
    train_samples, val_samples, test_samples = split_by_bearing(
        samples, split_ratio=split_ratio, seed=seed, verbose=True
    )
    
    print(f"数据集划分 (bearing-level):")
    print(f"  训练集: {len(train_samples)} 样本 ({len(set(s['bearing_id'] for s in train_samples))} bearings)")
    print(f"  验证集: {len(val_samples)} 样本 ({len(set(s['bearing_id'] for s in val_samples))} bearings)")
    print(f"  测试集: {len(test_samples)} 样本 ({len(set(s['bearing_id'] for s in test_samples))} bearings)")
    
    # 在训练集上计算通道均值/方差（跳过损坏/不完整 NPZ，如上传不完整导致的 EOFError）
    print("计算训练集统计量...")
    train_volumes = []
    train_densities = []
    bad_npz_paths = set()

    for sample in train_samples:
        npz_path = sample['npz_path']
        try:
            with np.load(npz_path) as data:
                x = data['x'].astype(np.float32)  # (2, 3000)
            train_volumes.append(x[0])  # log1p(volume)
            train_densities.append(x[1])  # density
        except (EOFError, OSError, ValueError) as e:
            bad_npz_paths.add(str(npz_path))

    if bad_npz_paths:
        n_bad = len(bad_npz_paths)
        train_samples = [s for s in train_samples if str(s['npz_path']) not in bad_npz_paths]
        val_samples = [s for s in val_samples if str(s['npz_path']) not in bad_npz_paths]
        test_samples = [s for s in test_samples if str(s['npz_path']) not in bad_npz_paths]
        print(f"已跳过 {n_bad} 个损坏/不完整的 NPZ 文件（如 EOFError: No data left in file）")
        print(f"  过滤后: 训练 {len(train_samples)}, 验证 {len(val_samples)}, 测试 {len(test_samples)}")

    if len(train_volumes) == 0:
        raise ValueError("训练集为空或全部 NPZ 损坏，无法计算统计量。请检查上传的 NPZ 是否完整。")
    
    all_volumes = np.concatenate(train_volumes)
    all_densities = np.concatenate(train_densities)
    
    channel_mean = np.array([np.mean(all_volumes), np.mean(all_densities)])
    channel_std = np.array([np.std(all_volumes), np.std(all_densities)])
    
    print(f"通道均值: {channel_mean}")
    print(f"通道标准差: {channel_std}")
    
    # 构建数据集
    train_dataset = SoundAPICacheDataset(
        cache_dir, train_samples, task=task, horizon=horizon,
        channel_mean=channel_mean, channel_std=channel_std
    )
    val_dataset = SoundAPICacheDataset(
        cache_dir, val_samples, task=task, horizon=horizon,
        channel_mean=channel_mean, channel_std=channel_std
    )
    test_dataset = SoundAPICacheDataset(
        cache_dir, test_samples, task=task, horizon=horizon,
        channel_mean=channel_mean, channel_std=channel_std
    )
    
    # DataLoader 通用参数（num_workers>0 时保持 worker 进程不退出）
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    
    # 对于包含 meta 的数据集，需要自定义 collate_fn
    def collate_fn(batch):
        """处理包含 meta 的批次"""
        x_list, y_list, meta_list = zip(*batch)
        x_batch = torch.stack(x_list)
        y_batch = torch.stack(y_list) if task != 'cls' else torch.tensor(y_list)
        return x_batch, y_batch, meta_list
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader


__all__ = ["LieGroupDataset", "get_dataloaders", "get_sound_api_cache_dataloaders"]

