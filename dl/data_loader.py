import os
import csv
import json
import re
from typing import Tuple, Optional, Dict, List

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
        condition_ids: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        self.signals = signals
        self.labels = labels.astype(np.int64)
        self.indices = indices.astype(np.int64)
        self.condition_ids = condition_ids if condition_ids is not None else None

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
    split_mode: str = "random",
    master_labels_path: Optional[str] = None,
    label_source_policy: str = "any",
    label_version: str = "latest",
    test_condition_id: Optional[str] = None,
    seed: int = 42,
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
    condition_ids = None

    # 优先从 master_labels 读取统一真值标签（single source of truth）
    if master_labels_path and os.path.exists(master_labels_path):
        labels, condition_ids = _load_labels_from_master_table(
            master_labels_path=master_labels_path,
            n_samples=signals.shape[0],
            fallback_labels=labels,
            label_source_policy=label_source_policy,
            label_version=label_version,
        )

    n_samples = signals.shape[0]

    # 计算各子集大小
    train_ratio, val_ratio, test_ratio = split_ratio
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"split_ratio 之和必须为 1，当前为: {split_ratio}")

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    if split_mode not in ("random", "leave_one_condition_out"):
        raise ValueError("split_mode 仅支持 'random' 或 'leave_one_condition_out'")

    if split_mode == "leave_one_condition_out":
        if condition_ids is None:
            raise ValueError(
                "split_mode=leave_one_condition_out 需要 master_labels 中包含 condition_id 字段"
            )
        train_indices, val_indices, test_indices = _split_leave_one_condition_out(
            condition_ids=condition_ids,
            split_ratio=split_ratio,
            test_condition_id=test_condition_id,
            seed=seed,
        )
    else:
        # 生成打乱后的索引
        all_indices = np.arange(n_samples)
        if shuffle:
            rng = np.random.default_rng(seed=seed)
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
    train_dataset = LieGroupDataset(
        signals, labels, train_indices, channel_mean, channel_std, condition_ids=condition_ids
    )
    val_dataset = LieGroupDataset(
        signals, labels, val_indices, channel_mean, channel_std, condition_ids=condition_ids
    )
    test_dataset = LieGroupDataset(
        signals, labels, test_indices, channel_mean, channel_std, condition_ids=condition_ids
    )

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


def _load_labels_from_master_table(
    master_labels_path: str,
    n_samples: int,
    fallback_labels: np.ndarray,
    label_source_policy: str = "any",
    label_version: str = "latest",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从统一标签表读取 fault_binary 与 condition_id。
    支持按 label_source / label_version 过滤，不满足过滤条件时退回 fallback_labels。
    """
    rows: List[Dict[str, str]] = []
    with open(master_labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        raise ValueError(f"master_labels 为空: {master_labels_path}")

    selected = rows
    if label_source_policy != "any":
        selected = [r for r in selected if r.get("label_source", "") == label_source_policy]
        if len(selected) == 0:
            raise ValueError(
                f"master_labels 过滤后为空: label_source_policy={label_source_policy}"
            )

    if label_version != "latest":
        selected = [r for r in selected if r.get("label_version", "") == label_version]
        if len(selected) == 0:
            raise ValueError(
                f"master_labels 过滤后为空: label_version={label_version}"
            )

    by_sample_id = {}
    for r in selected:
        sid = r.get("sample_id")
        if sid is not None and sid != "":
            try:
                by_sample_id[int(sid)] = r
            except ValueError:
                continue

    out_labels = np.asarray(fallback_labels, dtype=np.int64).copy()
    out_condition_ids = np.array(["unknown"] * n_samples, dtype=object)

    if len(by_sample_id) >= n_samples:
        for i in range(n_samples):
            r = by_sample_id.get(i)
            if r is None:
                continue
            fb = r.get("fault_binary")
            if fb not in (None, ""):
                out_labels[i] = int(float(fb))
            out_condition_ids[i] = r.get("condition_id", "unknown") or "unknown"
    else:
        if len(selected) != n_samples:
            raise ValueError(
                f"master_labels 样本数不匹配: selected={len(selected)}, n_samples={n_samples}"
            )
        for i, r in enumerate(selected):
            fb = r.get("fault_binary")
            if fb not in (None, ""):
                out_labels[i] = int(float(fb))
            out_condition_ids[i] = r.get("condition_id", "unknown") or "unknown"

    return out_labels, out_condition_ids


def _split_leave_one_condition_out(
    condition_ids: np.ndarray,
    split_ratio: Tuple[float, float, float],
    test_condition_id: Optional[str] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按工况切分：测试集固定一个 condition，验证集固定另一个 condition。"""
    unique_conditions = np.unique(condition_ids)
    if unique_conditions.shape[0] < 2:
        raise ValueError("leave_one_condition_out 至少需要 2 个不同工况")

    cond_to_idx: Dict[str, np.ndarray] = {
        str(cond): np.where(condition_ids == cond)[0] for cond in unique_conditions
    }
    sorted_conditions = sorted(
        cond_to_idx.keys(),
        key=lambda c: cond_to_idx[c].shape[0],
        reverse=True,
    )

    if test_condition_id is None:
        test_condition_id = sorted_conditions[0]
    if test_condition_id not in cond_to_idx:
        raise ValueError(f"test_condition_id 不存在: {test_condition_id}")

    remaining = [c for c in sorted_conditions if c != test_condition_id]
    val_condition_id = remaining[0] if len(remaining) > 0 else None
    if val_condition_id is None:
        raise ValueError("无法构建验证集工况，至少需要 2 个工况")

    test_indices = cond_to_idx[test_condition_id]
    val_indices = cond_to_idx[val_condition_id]

    train_indices = []
    for c in remaining[1:]:
        train_indices.append(cond_to_idx[c])
    if len(train_indices) == 0:
        # 仅两个工况时，在剩余工况中随机切分 train/val
        base = cond_to_idx[val_condition_id].copy()
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(base)
        n_train = int(base.shape[0] * (split_ratio[0] / (split_ratio[0] + split_ratio[1])))
        train_indices = base[:n_train]
        val_indices = base[n_train:]
    else:
        train_indices = np.concatenate(train_indices, axis=0)

    return (
        train_indices.astype(np.int64),
        val_indices.astype(np.int64),
        test_indices.astype(np.int64),
    )


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
    split_mode: str = "bearing",
    condition_map_path: Optional[str] = None,
    condition_policy: str = "xjtu_3cond",
    test_condition_id: Optional[str] = None,
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
    
    # task in {'risk','cls'} 时强制真实标签：
    # 优先使用 NPZ 内 fault_label，缺失时再走 NPZ→JSON→.f→sidecar 回填。
    if task in ('risk', 'cls'):
        print(f"{task} 任务：强制使用真实标签（优先 NPZ.fault_label，缺失回退 sidecar）...")
        fallback_cnt = 0
        for s in samples:
            fault_label = -1
            npz_path = s['npz_path']
            try:
                with np.load(npz_path) as data:
                    if 'fault_label' in data:
                        fault_label = int(data['fault_label'])
            except Exception:
                fault_label = -1
            if fault_label < 0:
                fault_label = resolve_fault_label_from_f_source(npz_path)
                fallback_cnt += 1
            s['fault_label'] = fault_label
        n_before = len(samples)
        samples = [s for s in samples if s['fault_label'] >= 0]
        dropped = n_before - len(samples)
        if dropped > 0:
            print(f"  已排除无法从 .f sidecar 解析标签的样本: {dropped} 个")
        print(f"  sidecar 回退解析次数: {fallback_cnt}")
        if len(samples) == 0:
            raise ValueError(
                "无有效真实标签。请确认 NPZ 由 build_sound_api_cache 从 convert_mc_to_api_json 的 JSON 构建，"
                "且 .f 侧有 sidecar .json 含 health_label/fault_label。"
            )
        n_fault = sum(1 for s in samples if int(s['fault_label']) > 0)
        n_normal = len(samples) - n_fault
        print(f"  保留有标签样本: {len(samples)} 个")
        print(f"  标签分布: normal={n_normal}, fault={n_fault}")
    
    # bearing-level split / leave-one-condition-out
    if split_mode == "leave_one_condition_out":
        cond_map = _load_condition_map(condition_map_path) if condition_map_path else {}
        for s in samples:
            s["condition_id"] = _resolve_sound_condition_id(
                sample=s,
                condition_map=cond_map,
                condition_policy=condition_policy,
            )
        train_samples, val_samples, test_samples = _split_sound_samples_by_condition(
            samples=samples,
            test_condition_id=test_condition_id,
            split_ratio=split_ratio,
            seed=seed,
        )
    else:
        train_samples, val_samples, test_samples = split_by_bearing(
            samples, split_ratio=split_ratio, seed=seed, verbose=True
        )
    
    split_tag = "condition-level" if split_mode == "leave_one_condition_out" else "bearing-level"
    print(f"数据集划分 ({split_tag}):")
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
        drop_last=True,
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


def _load_condition_map(path: str) -> Dict[str, str]:
    """加载 bearing_id -> condition_id 映射，支持 .json / .csv。"""
    map_path = os.path.abspath(path)
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"condition_map 文件不存在: {map_path}")
    mapping: Dict[str, str] = {}
    if map_path.lower().endswith(".json"):
        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            mapping = {str(k): str(v) for k, v in data.items()}
        elif isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    continue
                if "bearing_id" in row and "condition_id" in row:
                    mapping[str(row["bearing_id"])] = str(row["condition_id"])
    else:
        with open(map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "bearing_id" in row and "condition_id" in row:
                    mapping[str(row["bearing_id"])] = str(row["condition_id"])
    if len(mapping) == 0:
        raise ValueError(f"condition_map 为空或格式不正确: {map_path}")
    return mapping


def _infer_xjtu_condition_from_bearing_id(bearing_id: str) -> str:
    """
    XJTU 工况推断（默认策略）：
    每 5 个 bearing 为一个条件组，按 3 组循环映射到 cond_1/2/3。
    """
    m = re.search(r"\d+", str(bearing_id))
    if not m:
        return "unknown"
    bid = int(m.group(0))
    cond_idx = ((bid - 1) // 5) % 3 + 1
    return f"cond_{cond_idx}"


def _resolve_sound_condition_id(
    sample: Dict,
    condition_map: Dict[str, str],
    condition_policy: str = "xjtu_3cond",
) -> str:
    bid = str(sample.get("bearing_id", ""))
    if bid in condition_map:
        return condition_map[bid]
    if condition_policy == "xjtu_3cond":
        return _infer_xjtu_condition_from_bearing_id(bid)
    return "unknown"


def _split_sound_samples_by_condition(
    samples: List[Dict],
    test_condition_id: Optional[str],
    split_ratio: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """按 condition_id 切分：留一条件测试，留一条件验证，其余训练。"""
    cond_groups: Dict[str, List[Dict]] = {}
    for s in samples:
        cid = str(s.get("condition_id", "unknown"))
        cond_groups.setdefault(cid, []).append(s)

    condition_ids = sorted(cond_groups.keys())
    if len(condition_ids) < 2:
        raise ValueError("leave_one_condition_out 至少需要 2 个 condition_id")

    if test_condition_id is None:
        # 默认选择样本数最多的 condition 作为测试集
        test_condition_id = sorted(condition_ids, key=lambda c: len(cond_groups[c]), reverse=True)[0]
    if test_condition_id not in cond_groups:
        raise ValueError(f"test_condition_id 不存在: {test_condition_id}")

    remaining = [c for c in condition_ids if c != test_condition_id]
    val_condition_id = remaining[0]

    train_samples: List[Dict] = []
    for cid in remaining[1:]:
        train_samples.extend(cond_groups[cid])
    val_samples = list(cond_groups[val_condition_id])
    test_samples = list(cond_groups[test_condition_id])

    if len(train_samples) == 0:
        # 只有两个工况时，在验证工况内部按比例拆一部分给 train
        rng = np.random.default_rng(seed=seed)
        idx = np.arange(len(val_samples))
        rng.shuffle(idx)
        train_ratio = split_ratio[0] / max(1e-8, (split_ratio[0] + split_ratio[1]))
        n_train = int(len(val_samples) * train_ratio)
        train_idx = set(idx[:n_train].tolist())
        new_train, new_val = [], []
        for i, s in enumerate(val_samples):
            (new_train if i in train_idx else new_val).append(s)
        train_samples, val_samples = new_train, new_val

    print("\n" + "=" * 80)
    print("数据集划分概况 (Leave-One-Condition-Out)")
    print("=" * 80)
    print(f"测试工况: {test_condition_id}, 验证工况: {val_condition_id}")
    print(f"训练集: {len(train_samples)} | 验证集: {len(val_samples)} | 测试集: {len(test_samples)}")
    print("=" * 80)
    return train_samples, val_samples, test_samples


__all__ = ["LieGroupDataset", "get_dataloaders", "get_sound_api_cache_dataloaders"]
