"""
Sound API Cache Dataset - 从 NPZ 缓存加载数据（只读 NPZ）

支持任务：
- hi: 健康指数回归，y_hi = 1 - t/(T-1)
- risk: 风险预测二分类，给定 horizon=K，tf=floor(0.3*T)，y=1 if t+K>=tf else 0
- cls: 分类任务，标签可从 NPZ 或通过思路 B（NPZ→JSON→.f→sidecar）无损还原

注意：
- 本模块只读取 NPZ 文件，不读取 JSON 或 xlsx（思路 B 时会读 JSON 与 sidecar 补标签）
- 默认缓存目录：datasets/sound_api/cache_npz/
"""
import os
import csv
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from core.json_io import loadFirstJson


def resolve_fault_label_from_f_source(npz_path: Path) -> int:
    """
    思路 B：利用 NPZ/JSON 与 .f 的一一对应，从 NPZ 反推到 .f 的 sidecar，补回标签（无损还原）。
    
    流程：NPZ.source_path（JSON 路径）→ 读 JSON.metadata.source_path（.f 路径）→
          sidecar = .f 同目录同 stem 的 .json → 读 sidecar 的 health_label / fault_label。
    
    Returns:
        标签值（>=0 有效），无法解析时返回 -1。
    """
    try:
        with np.load(npz_path) as data:
            if "source_path" not in data:
                return -1
            sp = data["source_path"]
            json_path = Path(sp.item() if hasattr(sp, "item") else str(sp))
    except Exception:
        return -1
    if not json_path.is_absolute() and not json_path.exists():
        json_path = Path(os.path.abspath(str(json_path)))
    if not json_path.exists():
        return -1
    try:
        obj = loadFirstJson(json_path)
        meta = obj.get("metadata") or {}
        f_path_str = meta.get("source_path")
        if not f_path_str:
            return -1
        f_path = Path(f_path_str)
        sidecar_path = f_path.with_suffix(".json")
        if not sidecar_path.exists():
            return -1
        with open(sidecar_path, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        label = sidecar.get("fault_label", sidecar.get("health_label", -1))
        if label is None:
            return -1
        val = int(label)
        return val if val >= 0 else -1
    except Exception:
        return -1


class SoundAPICacheDataset(Dataset):
    """
    从 NPZ 缓存加载声音数据
    
    数据格式：
    - x: (2, 3000) float32，x[0]=log1p(volume), x[1]=density
    - meta: 包含 bearing_id, t, T 等信息
    """
    
    def __init__(
        self,
        cache_dir: str,
        samples: List[Dict],
        task: str = 'hi',
        horizon: Optional[int] = None,
        channel_mean: Optional[np.ndarray] = None,
        channel_std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            cache_dir: NPZ 缓存目录
            samples: 样本列表，每个元素包含 {'bearing_id', 't', 'T', 'npz_path'}
            task: 任务类型 ('hi', 'risk', 'cls')
            horizon: 风险预测的时间窗口（仅用于 risk 任务）
            channel_mean: (2,) 通道均值，用于标准化
            channel_std: (2,) 通道标准差，用于标准化
        """
        self.cache_dir = Path(cache_dir)
        self.samples = samples
        self.task = task
        self.horizon = horizon
        
        # 标准化参数
        if channel_mean is not None:
            self.channel_mean = channel_mean.reshape(2, 1)
        else:
            self.channel_mean = None
        
        if channel_std is not None:
            self.channel_std = channel_std.reshape(2, 1)
        else:
            self.channel_std = None
        
        # 验证任务参数
        if task == 'risk' and horizon is None:
            raise ValueError("risk 任务需要指定 horizon 参数")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        npz_path = sample['npz_path']
        
        # 加载 NPZ（上下文内复制所需数组，避免句柄泄漏）
        try:
            with np.load(npz_path) as data:
                x = data['x'].astype(np.float32)  # (2, 3000)
                if 'fault_label' in data:
                    fault_label_npz = int(data['fault_label'])
                else:
                    fault_label_npz = -1
        except (EOFError, OSError, ValueError) as e:
            raise RuntimeError(f"NPZ 损坏或未完整上传，无法加载: {npz_path}") from e
        
        # 分类标签：优先用思路 B 在 data_loader 中已解析的 sample['fault_label']，否则用 NPZ
        if self.task == 'cls':
            fault_label = sample.get('fault_label', fault_label_npz)
            if fault_label == -1 and fault_label_npz >= 0:
                fault_label = fault_label_npz
        else:
            fault_label = fault_label_npz
        
        # 标准化
        if self.channel_mean is not None and self.channel_std is not None:
            x = (x - self.channel_mean) / (self.channel_std + 1e-8)
        
        # 计算标签
        bearing_id = sample['bearing_id']
        t = sample['t']
        T = sample['T']
        
        if self.task == 'hi':
            # 健康指数：y_hi = 1 - t/(T-1)
            if T > 1:
                y = 1.0 - t / (T - 1)
            else:
                y = 1.0  # T=1 的边界情况
            y = np.float32(y)
        
        elif self.task == 'risk':
            # 风险预测：给定 horizon=K，tf=floor(0.3*T)，y=1 if t+K>=tf else 0
            tf = int(np.floor(0.3 * T))
            y = 1.0 if (t + self.horizon >= tf) else 0.0
            y = np.float32(y)
        
        elif self.task == 'cls':
            # 分类任务：使用思路 B 还原或 NPZ 中的 fault_label（与 .f 一一对应、无损）
            y = np.int64(fault_label)
        
        else:
            raise ValueError(f"未知任务类型: {self.task}")
        
        # 元数据
        meta = {
            'bearing_id': bearing_id,
            't': t,
            'T': T,
            'npz_path': str(npz_path)
        }
        
        # 转换为 Tensor
        x_tensor = torch.from_numpy(x)
        if self.task == 'cls':
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)
        
        return x_tensor, y_tensor, meta


def load_samples_from_cache_dir(cache_dir: str) -> List[Dict]:
    """
    从缓存目录加载所有样本（只读 NPZ）
    
    扫描 datasets/sound_api/cache_npz/{bearing_id}/{t:06d}.npz
    构建样本列表，包含 bearing_id, t, T, npz_path
    
    注意：本函数只读取 NPZ 文件，不读取 JSON 或 xlsx
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"缓存目录不存在: {cache_dir}")
    
    # 按 bearing_id 分组
    bearing_groups = defaultdict(list)
    
    for bearing_dir in cache_path.iterdir():
        if not bearing_dir.is_dir():
            continue
        
        bearing_id = bearing_dir.name
        
        # 扫描该 bearing 下的所有 npz 文件
        npz_files = sorted(bearing_dir.glob('*.npz'))
        
        for npz_file in npz_files:
            # 从文件名解析 t（格式：000123.npz）
            try:
                t = int(npz_file.stem)
            except ValueError:
                continue
            
            bearing_groups[bearing_id].append({
                'bearing_id': bearing_id,
                't': t,
                'npz_path': npz_file
            })
    
    # 计算每个 bearing 的 T（总样本数）
    samples = []
    for bearing_id, group in bearing_groups.items():
        # 按 t 排序
        group.sort(key=lambda x: x['t'])
        T = len(group)
        
        # 为每个样本添加 T
        for item in group:
            item['T'] = T
            samples.append(item)
    
    return samples


def load_samples_from_index(index_path: str, cache_dir: str) -> List[Dict]:
    """
    从 index.csv 加载样本列表（更可靠，因为 index.csv 已经校验过 t 连续性）
    
    Args:
        index_path: index.csv 路径
        cache_dir: NPZ 缓存目录（默认: datasets/sound_api/cache_npz）
    
    注意：本函数只读取 NPZ 文件，不读取 JSON 或 xlsx
    """
    cache_path = Path(cache_dir)
    
    # 读取 index.csv
    records = []
    with open(index_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'bearing_id': row['bearing_id'],
                't': int(row['t'])
            })
    
    # 按 bearing_id 分组，计算 T
    bearing_groups = defaultdict(list)
    for record in records:
        bearing_groups[record['bearing_id']].append(record['t'])
    
    # 构建样本列表
    samples = []
    for bearing_id, t_list in bearing_groups.items():
        t_list.sort()
        T = len(t_list)
        
        for t in t_list:
            npz_path = cache_path / bearing_id / f"{t:06d}.npz"
            if not npz_path.exists():
                continue  # 跳过不存在的文件
            
            samples.append({
                'bearing_id': bearing_id,
                't': t,
                'T': T,
                'npz_path': npz_path
            })
    
    return samples


def split_by_bearing(
    samples: List[Dict],
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    按 bearing_id 分组切分数据集（bearing-level split）
    
    Args:
        samples: 样本列表
        split_ratio: (train, val, test) 比例
        seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        train_samples, val_samples, test_samples
    """
    # 按 bearing_id 分组
    bearing_groups = defaultdict(list)
    for sample in samples:
        bearing_groups[sample['bearing_id']].append(sample)
    
    # 获取所有 bearing_id
    bearing_ids = sorted(bearing_groups.keys())
    
    # 随机打乱 bearing_id
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(bearing_ids)
    
    # 计算各子集的 bearing 数量
    n_bearings = len(bearing_ids)
    train_ratio, val_ratio, test_ratio = split_ratio
    
    n_train = int(n_bearings * train_ratio)
    n_val = int(n_bearings * val_ratio)
    n_test = n_bearings - n_train - n_val
    
    train_bearings = bearing_ids[:n_train]
    val_bearings = bearing_ids[n_train:n_train + n_val]
    test_bearings = bearing_ids[n_train + n_val:]
    
    # 构建样本列表
    train_samples = []
    val_samples = []
    test_samples = []
    
    for bearing_id, group in bearing_groups.items():
        if bearing_id in train_bearings:
            train_samples.extend(group)
        elif bearing_id in val_bearings:
            val_samples.extend(group)
        elif bearing_id in test_bearings:
            test_samples.extend(group)
    
    # 打印分配概况
    if verbose:
        print("\n" + "=" * 80)
        print("数据集划分概况 (Bearing-level Split)")
        print("=" * 80)
        print(f"总样本数: {len(samples)}")
        print(f"总 bearing 数: {n_bearings}")
        print(f"划分比例: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
        print()
        print(f"训练集: {len(train_samples)} 样本, {len(train_bearings)} bearings")
        if train_bearings:
            train_samples_per_bearing = [len(bearing_groups[b]) for b in train_bearings]
            print(f"  平均每 bearing: {np.mean(train_samples_per_bearing):.1f} 样本")
            print(f"  样本范围: {min(train_samples_per_bearing)} - {max(train_samples_per_bearing)}")
        print()
        print(f"验证集: {len(val_samples)} 样本, {len(val_bearings)} bearings")
        if val_bearings:
            val_samples_per_bearing = [len(bearing_groups[b]) for b in val_bearings]
            print(f"  平均每 bearing: {np.mean(val_samples_per_bearing):.1f} 样本")
            print(f"  样本范围: {min(val_samples_per_bearing)} - {max(val_samples_per_bearing)}")
        print()
        print(f"测试集: {len(test_samples)} 样本, {len(test_bearings)} bearings")
        if test_bearings:
            test_samples_per_bearing = [len(bearing_groups[b]) for b in test_bearings]
            print(f"  平均每 bearing: {np.mean(test_samples_per_bearing):.1f} 样本")
            print(f"  样本范围: {min(test_samples_per_bearing)} - {max(test_samples_per_bearing)}")
        print("=" * 80)
    
    return train_samples, val_samples, test_samples


__all__ = [
    "SoundAPICacheDataset",
    "resolve_fault_label_from_f_source",
    "load_samples_from_cache_dir",
    "load_samples_from_index",
    "split_by_bearing"
]
