#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CWRU 轴承故障数据集批量加载脚本
===============================
此脚本用于批量读取 CWRU (Case Western Reserve University) 轴承数据集的 .mat 文件，
提取振动信号和元数据，便于后续集成到故障诊断智能模型中。

功能:
    - 递归扫描数据集目录中的所有 .mat 文件
    - 自动解析文件路径推断故障类型、采样率、负载等元数据
    - 支持导出为 numpy 格式便于深度学习模型使用
    - 支持数据分段切片，生成固定长度样本
    - 提供 CWRUDataset 类可直接用于 PyTorch DataLoader

使用示例:
    # 命令行使用
    python load_cwru.py ../CWRU-dataset-main ./output --segment-length 2048
    
    # Python 代码中使用
    from load_cwru import CWRUDataLoader
    loader = CWRUDataLoader('CWRU-dataset-main')
    X, y, meta = loader.load_all()
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
from scipy.io import loadmat

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 故障类型映射
# =============================================================================
FAULT_LABELS = {
    'Normal': 0,
    'B': 1,      # Ball fault (滚动体故障)
    'IR': 2,     # Inner Race fault (内圈故障)
    'OR': 3,     # Outer Race fault (外圈故障)
}

FAULT_NAMES = {
    0: 'Normal',
    1: 'Ball',
    2: 'Inner Race',
    3: 'Outer Race',
}

# 故障直径 (mil -> mm)
FAULT_DIAMETERS = {
    '007': 0.007 * 25.4,  # 0.1778 mm
    '014': 0.014 * 25.4,  # 0.3556 mm
    '021': 0.021 * 25.4,  # 0.5334 mm
    '028': 0.028 * 25.4,  # 0.7112 mm
}

# 负载对应转速 (HP -> RPM)
LOAD_RPM = {
    0: 1797,
    1: 1772,
    2: 1750,
    3: 1730,
}


# =============================================================================
# 核心函数
# =============================================================================
def find_signal_key(mat_dict: Dict) -> Optional[str]:
    """
    从 .mat 文件字典中找到振动信号的键名。
    CWRU 数据集中，驱动端信号键名包含 'DE'，风扇端包含 'FE'。
    """
    priority_patterns = ['DE_time', 'FE_time', 'BA_time', '_time']
    
    # 首先按优先级查找
    for pattern in priority_patterns:
        for key in mat_dict.keys():
            if pattern in key and not key.startswith('__'):
                return key
    
    # 如果没找到，选择最长的数组
    max_len = 0
    best_key = None
    for key, value in mat_dict.items():
        if key.startswith('__'):
            continue
        try:
            arr = np.asarray(value)
            if arr.ndim >= 1:
                length = arr.size
                if length > max_len:
                    max_len = length
                    best_key = key
        except:
            continue
    
    return best_key


def load_mat_signal(filepath: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    加载 .mat 文件并提取振动信号。
    
    Args:
        filepath: .mat 文件路径
        
    Returns:
        (signal, key): 信号数组和对应的键名，失败返回 (None, None)
    """
    try:
        mat_data = loadmat(filepath)
    except Exception as e:
        logger.warning(f"加载文件失败 {filepath}: {e}")
        return None, None
    
    key = find_signal_key(mat_data)
    if key is None:
        logger.warning(f"未找到信号数据 {filepath}")
        return None, None
    
    signal = np.asarray(mat_data[key]).flatten().astype(np.float32)
    return signal, key


def find_signal_keys(mat_dict: Dict, sensor_locations: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    查找所有可用的时间序列信号键，返回 [(location, key)] 列表。
    优先匹配 'DE_time'、'FE_time'、'BA_time'，其次匹配包含 '_time' 的键。
    可通过 sensor_locations 指定需要的通道集合，例如 ['DE','FE']。
    """
    candidates: List[Tuple[str, str]] = []
    loc_order = ['DE', 'FE', 'BA']
    loc_patterns = {
        'DE': ['DE_time', 'DE-'],
        'FE': ['FE_time', 'FE-'],
        'BA': ['BA_time', 'BA-'],
    }

    def add_candidate(location: str, key: str):
        if sensor_locations and location not in sensor_locations:
            return
        candidates.append((location, key))

    # 先按明确的键名匹配
    for key in mat_dict.keys():
        if key.startswith('__'):
            continue
        for loc in loc_order:
            for pat in loc_patterns[loc]:
                if pat in key:
                    add_candidate(loc, key)
                    break

    # 再补充 '_time' 模式
    for key in mat_dict.keys():
        if key.startswith('__'):
            continue
        if '_time' in key and all(key != k for _, k in candidates):
            # 未能确定具体位置，标记为 'DE' 优先或 'UNK'
            add_candidate('DE', key)

    # 去重并按 loc_order 排序
    seen = set()
    unique_candidates: List[Tuple[str, str]] = []
    for loc in loc_order:
        for l, k in candidates:
            if l == loc and (l, k) not in seen:
                unique_candidates.append((l, k))
                seen.add((l, k))
    return unique_candidates


def load_mat_signals(filepath: str, sensor_locations: Optional[List[str]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    加载 .mat 文件并提取多通道振动信号。

    Returns:
        signals: {location: 1D np.ndarray}
        keys_map: {location: key_name}
    """
    try:
        mat_data = loadmat(filepath)
    except Exception as e:
        logger.warning(f"加载文件失败 {filepath}: {e}")
        return {}, {}

    pairs = find_signal_keys(mat_data, sensor_locations=sensor_locations)
    signals: Dict[str, np.ndarray] = {}
    keys_map: Dict[str, str] = {}
    for loc, key in pairs:
        try:
            arr = np.asarray(mat_data[key]).flatten().astype(np.float32)
            signals[loc] = arr
            keys_map[loc] = key
        except Exception as e:
            logger.warning(f"解析键失败 {filepath} 键={key}: {e}")
            continue
    return signals, keys_map


def parse_filepath_metadata(root: str, filepath: str) -> Dict[str, Any]:
    """
    从文件路径解析元数据。
    
    路径结构示例:
        12k_Drive_End_Bearing_Fault_Data/IR/007/105_0.mat
        Normal/97_Normal_0.mat
    
    Args:
        root: 数据集根目录
        filepath: .mat 文件完整路径
        
    Returns:
        元数据字典
    """
    rel_path = os.path.relpath(filepath, root).replace('\\', '/')
    parts = rel_path.split('/')
    filename = os.path.basename(filepath)
    
    meta = {
        'filepath': filepath,
        'relative_path': rel_path,
        'filename': filename,
        'sampling_rate': None,
        'sensor_location': None,
        'fault_type': None,
        'fault_label': None,
        'fault_diameter': None,
        'load_hp': None,
        'rpm': None,
        'or_position': None,  # 外圈故障位置: @3, @6, @12
    }
    
    # 解析顶层文件夹 (采样率和传感器位置)
    top_folder = parts[0] if len(parts) > 0 else ''
    
    # 采样率: 12k 或 48k
    sr_match = re.search(r'(\d+)k', top_folder)
    if sr_match:
        meta['sampling_rate'] = int(sr_match.group(1)) * 1000
    
    # 传感器位置
    if 'Drive_End' in top_folder:
        meta['sensor_location'] = 'DE'
    elif 'Fan_End' in top_folder:
        meta['sensor_location'] = 'FE'
    
    # 正常数据特殊处理
    if 'Normal' in top_folder or 'Normal' in filename:
        meta['fault_type'] = 'Normal'
        meta['fault_label'] = FAULT_LABELS['Normal']
        meta['fault_diameter'] = 0.0
        meta['sampling_rate'] = meta['sampling_rate'] or 12000
        # 解析负载: 97_Normal_0.mat -> load = 0
        load_match = re.search(r'_(\d)\.mat$', filename)
        if load_match:
            load = int(load_match.group(1))
            meta['load_hp'] = load
            meta['rpm'] = LOAD_RPM.get(load)
        return meta
    
    # 故障类型: B, IR, OR
    if len(parts) > 1:
        fault_type = parts[1]
        if fault_type in FAULT_LABELS:
            meta['fault_type'] = fault_type
            meta['fault_label'] = FAULT_LABELS[fault_type]
    
    # 故障直径: 007, 014, 021, 028
    if len(parts) > 2:
        diameter_str = parts[2]
        if diameter_str in FAULT_DIAMETERS:
            meta['fault_diameter'] = FAULT_DIAMETERS[diameter_str]
    
    # 外圈故障位置
    if meta['fault_type'] == 'OR':
        # 检查子文件夹 @3, @6, @12
        for p in parts:
            if p.startswith('@'):
                meta['or_position'] = p
                break
        # 或者文件名中包含 @6 等
        or_match = re.search(r'@(\d+)', filename)
        if or_match:
            meta['or_position'] = f"@{or_match.group(1)}"
    
    # 负载: 文件名格式 105_0.mat -> load = 0
    load_match = re.search(r'_(\d)\.mat$', filename)
    if load_match:
        load = int(load_match.group(1))
        meta['load_hp'] = load
        meta['rpm'] = LOAD_RPM.get(load)
    
    return meta


def segment_signal(signal: np.ndarray, segment_length: int, 
                   overlap: float = 0.5) -> np.ndarray:
    """
    将长信号切分为固定长度的片段。
    
    Args:
        signal: 原始信号 (1D)
        segment_length: 每个片段的长度
        overlap: 重叠比例 (0-1)
        
    Returns:
        切分后的信号数组 (N, segment_length)
    """
    step = int(segment_length * (1 - overlap))
    if step <= 0:
        step = 1
    
    segments = []
    for start in range(0, len(signal) - segment_length + 1, step):
        segments.append(signal[start:start + segment_length])
    
    if len(segments) == 0:
        # 信号太短，零填充
        padded = np.zeros(segment_length, dtype=signal.dtype)
        padded[:len(signal)] = signal
        segments.append(padded)
    
    return np.array(segments)


# =============================================================================
# 数据加载器类
# =============================================================================
class CWRUDataLoader:
    """
    CWRU 数据集加载器，用于批量读取和预处理数据。
    
    使用示例:
        loader = CWRUDataLoader('CWRU-dataset-main')
        
        # 加载所有数据
        X, y, meta = loader.load_all()
        
        # 加载并分段
        X, y, meta = loader.load_all(segment_length=2048, overlap=0.5)
        
        # 只加载特定故障类型
        X, y, meta = loader.load_all(fault_types=['IR', 'OR'])
    """
    
    def __init__(self, root_dir: str):
        """
        初始化加载器。
        
        Args:
            root_dir: CWRU 数据集根目录
        """
        self.root_dir = os.path.abspath(root_dir)
        self.mat_files = self._scan_mat_files()
        logger.info(f"扫描到 {len(self.mat_files)} 个 .mat 文件")
    
    def _scan_mat_files(self) -> List[str]:
        """递归扫描所有 .mat 文件"""
        mat_files = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fn in filenames:
                if fn.lower().endswith('.mat'):
                    mat_files.append(os.path.join(dirpath, fn))
        mat_files.sort()
        return mat_files
    
    def load_single(self, filepath: str) -> Dict[str, Any]:
        """
        加载单个 .mat 文件。
        
        Returns:
            包含 'signal', 'meta', 'key' 的字典
        """
        signal, key = load_mat_signal(filepath)
        meta = parse_filepath_metadata(self.root_dir, filepath)
        return {
            'signal': signal,
            'meta': meta,
            'signal_key': key,
        }

    def load_single_multi(self, filepath: str, sensor_locations: Optional[List[str]] = None) -> Dict[str, Any]:
        """加载单个 .mat 文件（多通道）。"""
        signals, keys_map = load_mat_signals(filepath, sensor_locations=sensor_locations)
        meta = parse_filepath_metadata(self.root_dir, filepath)
        return {
            'signals': signals,  # {loc: 1D array}
            'meta': meta,
            'signal_keys': keys_map,  # {loc: key}
        }
    
    def load_all(self, 
                 segment_length: Optional[int] = None,
                 overlap: float = 0.5,
                 fault_types: Optional[List[str]] = None,
                 sampling_rates: Optional[List[int]] = None,
                 normalize: bool = False,
                 multi_channel: bool = False,
                 sensor_locations: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        批量加载所有数据。
        
        Args:
            segment_length: 分段长度，None 表示不分段
            overlap: 分段重叠比例
            fault_types: 筛选故障类型，如 ['Normal', 'IR', 'OR', 'B']
            sampling_rates: 筛选采样率，如 [12000, 48000]
            normalize: 是否对信号进行标准化
            
        Returns:
            X: 信号数组 (N, segment_length) 或信号列表
            y: 标签数组 (N,)
            meta: 元数据列表
        """
        all_signals = []
        all_labels = []
        all_meta = []
        
        for filepath in self.mat_files:
            if multi_channel:
                data = self.load_single_multi(filepath, sensor_locations=sensor_locations)
                signals_dict: Dict[str, np.ndarray] = data['signals']
                meta = data['meta']
                if not signals_dict:
                    continue
                # 筛选
                if fault_types and meta['fault_type'] not in fault_types:
                    continue
                if sampling_rates and meta['sampling_rate'] not in sampling_rates:
                    continue

                label = meta['fault_label']
                if label is None:
                    logger.warning(f"未知故障类型: {filepath}")
                    continue

                # 统一长度（取最短）并按固定通道顺序堆叠；若指定了 sensor_locations，则对缺失通道用零填充
                order = ['DE', 'FE', 'BA']
                available_locs = [loc for loc in order if loc in signals_dict]
                target_locs = [loc for loc in order if (sensor_locations and loc in sensor_locations) or (not sensor_locations and loc in available_locs)]
                if not target_locs:
                    continue
                min_len = min([len(signals_dict[loc]) for loc in available_locs]) if available_locs else 0
                if min_len <= 0:
                    continue
                stacked_list = []
                for loc in target_locs:
                    if loc in signals_dict:
                        arr = signals_dict[loc][:min_len]
                    else:
                        arr = np.zeros(min_len, dtype=np.float32)
                    stacked_list.append(arr)
                stacked = np.stack(stacked_list, axis=0)  # (C, L)

                # 标准化（逐通道）
                if normalize:
                    ch_mean = stacked.mean(axis=1, keepdims=True)
                    ch_std = stacked.std(axis=1, keepdims=True)
                    stacked = (stacked - ch_mean) / (ch_std + 1e-8)

                # 分段
                if segment_length:
                    step = int(segment_length * (1 - overlap))
                    if step <= 0:
                        step = 1
                    if min_len < segment_length:
                        # 零填充到 segment_length
                        pad_width = segment_length - min_len
                        stacked = np.pad(stacked, ((0, 0), (0, pad_width)), mode='constant')
                        min_len = segment_length
                    for start in range(0, min_len - segment_length + 1, step):
                        seg = stacked[:, start:start + segment_length]
                        all_signals.append(seg.astype(np.float32))
                        all_labels.append(label)
                        meta_seg = meta.copy()
                        meta_seg['channels'] = target_locs
                        all_meta.append(meta_seg)
                else:
                    all_signals.append(stacked.astype(np.float32))
                    all_labels.append(label)
                    meta_full = meta.copy()
                    meta_full['channels'] = target_locs
                    all_meta.append(meta_full)
            else:
                data = self.load_single(filepath)
                signal = data['signal']
                meta = data['meta']

                if signal is None:
                    continue

                # 筛选
                if fault_types and meta['fault_type'] not in fault_types:
                    continue
                if sampling_rates and meta['sampling_rate'] not in sampling_rates:
                    continue

                label = meta['fault_label']
                if label is None:
                    logger.warning(f"未知故障类型: {filepath}")
                    continue

                # 标准化
                if normalize:
                    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

                # 分段
                if segment_length:
                    segments = segment_signal(signal, segment_length, overlap)
                    for seg in segments:
                        all_signals.append(seg)
                        all_labels.append(label)
                        all_meta.append(meta.copy())
                else:
                    all_signals.append(signal)
                    all_labels.append(label)
                    all_meta.append(meta)
        
        # 转换为数组
        if segment_length:
            X = np.array(all_signals, dtype=np.float32)
        else:
            X = all_signals  # 不同长度，保持列表
        y = np.array(all_labels, dtype=np.int64)
        
        ch_info = ''
        if multi_channel and segment_length and isinstance(X, np.ndarray):
            ch_info = f", shape={X.shape} (N,C,L)"
        elif segment_length and isinstance(X, np.ndarray):
            ch_info = f", shape={X.shape} (N,L)"
        logger.info(f"加载完成: {len(all_labels)} 个样本{ch_info}, "
                    f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, all_meta
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取各类别的文件数量分布"""
        dist = {}
        for filepath in self.mat_files:
            meta = parse_filepath_metadata(self.root_dir, filepath)
            ft = meta['fault_type'] or 'Unknown'
            dist[ft] = dist.get(ft, 0) + 1
        return dist
    
    def export_numpy(self, out_dir: str, segment_length: Optional[int] = 2048, 
                     overlap: float = 0.5, normalize: bool = True,
                     multi_channel: bool = False,
                     sensor_locations: Optional[List[str]] = None,
                     fault_types: Optional[List[str]] = None,
                     sampling_rates: Optional[List[int]] = None):
        """
        导出为 NumPy 格式，便于深度学习训练。
        
        输出文件:
            - signals.npy: 当分段导出时为数组 (N, segment_length) 或 (N, C, L)；当整段导出时为长度为 N 的列表（每项为 1D 或 2D 数组）
            - labels.npy: 标签数组 (N,)
            - metadata.json: 元数据列表
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        X, y, meta = self.load_all(
            segment_length=segment_length,
            overlap=overlap,
            fault_types=fault_types,
            sampling_rates=sampling_rates,
            normalize=normalize,
            multi_channel=multi_channel,
            sensor_locations=sensor_locations,
        )
        
        np.save(out_dir / 'signals.npy', X)
        np.save(out_dir / 'labels.npy', y)
        
        # 保存元数据 (移除不可序列化的内容)
        meta_clean = []
        for m in meta:
            m_clean = {k: v for k, v in m.items() if isinstance(v, (str, int, float, type(None)))}
            meta_clean.append(m_clean)
        
        with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(meta_clean, f, ensure_ascii=False, indent=2)
        
        # 兼容列表（整段导出）情况下的日志输出
        if isinstance(X, np.ndarray):
            x_shape_info = f"shape={X.shape}"
        else:
            # 列表模式：每个元素可能是 1D(单通道) 或 2D(多通道)
            sample_desc = "list"
            if len(X) > 0:
                try:
                    sample_shape = getattr(X[0], 'shape', None)
                    if sample_shape is not None:
                        sample_desc = f"list, sample_shape={sample_shape}"
                except Exception:
                    pass
            x_shape_info = f"{len(X)} samples ({sample_desc})"

        logger.info(f"数据已导出到 {out_dir}")
        logger.info(f"  - signals.npy: {x_shape_info}")
        logger.info(f"  - labels.npy: shape={y.shape}")
        logger.info(f"  - metadata.json: {len(meta_clean)} 条记录")


# =============================================================================
# PyTorch Dataset (可选)
# =============================================================================
try:
    import torch
    from torch.utils.data import Dataset
    
    class CWRUDataset(Dataset):
        """
        PyTorch Dataset 封装，可直接用于 DataLoader。
        
        使用示例:
            dataset = CWRUDataset('CWRU-dataset-main', segment_length=2048)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            for X, y in dataloader:
                # X: (batch_size, 1, segment_length)
                # y: (batch_size,)
                pass
        """
        
        def __init__(self, root_dir: str, segment_length: int = 2048,
                     overlap: float = 0.5, normalize: bool = True,
                     fault_types: Optional[List[str]] = None,
                     transform=None,
                     multi_channel: bool = False,
                     sensor_locations: Optional[List[str]] = None):
            self.loader = CWRUDataLoader(root_dir)
            self.X, self.y, self.meta = self.loader.load_all(
                segment_length=segment_length,
                overlap=overlap,
                fault_types=fault_types,
                normalize=normalize,
                multi_channel=multi_channel,
                sensor_locations=sensor_locations,
            )
            self.transform = transform
        
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, idx):
            x = self.X[idx]
            y = self.y[idx]
            
            if self.transform:
                x = self.transform(x)
            
            # 保持形状：(C, length)；若是单通道 1D，则添加通道维度
            x = torch.from_numpy(x)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            y = torch.tensor(y, dtype=torch.long)
            
            return x, y
    
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


# =============================================================================
# 命令行接口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='CWRU 轴承故障数据集批量加载工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 扫描数据集并显示统计信息
  python load_cwru.py ../CWRU-dataset-main --info
  
  # 导出为 NumPy 格式
  python load_cwru.py ../CWRU-dataset-main ./output --segment-length 2048
  
  # 只导出内圈和外圈故障
  python load_cwru.py ../CWRU-dataset-main ./output --fault-types IR OR
        """
    )
    
    parser.add_argument('root', help='CWRU 数据集根目录')
    parser.add_argument('out', nargs='?', default='./cwru_processed',
                        help='输出目录 (默认: ./cwru_processed)')
    parser.add_argument('--info', action='store_true',
                        help='只显示数据集信息，不导出')
    parser.add_argument('--segment-length', type=int, default=2048,
                        help='分段长度 (默认: 2048)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='分段重叠比例 (默认: 0.5)')
    parser.add_argument('--fault-types', nargs='+', default=None,
                        help='筛选故障类型: Normal B IR OR')
    parser.add_argument('--sampling-rates', nargs='+', type=int, default=None,
                        help='筛选采样率: 如 12000 48000')
    parser.add_argument('--no-normalize', action='store_true',
                        help='不进行标准化')
    parser.add_argument('--multi-channel', action='store_true',
                        help='启用多通道加载（如 DE/FE/BA）并导出 (N,C,L) 或整段 (C,L)')
    parser.add_argument('--sensor-locations', nargs='+', choices=['DE', 'FE', 'BA'], default=None,
                        help='指定通道位置以加载：DE FE BA（默认自动识别可用通道）')
    parser.add_argument('--no-seg', action='store_true',
                        help='不进行分段，导出整段原始序列（可能长度不一致，signals.npy 为列表）')
    
    args = parser.parse_args()
    
    loader = CWRUDataLoader(args.root)
    
    # 显示数据集信息
    print("\n" + "="*50)
    print("CWRU 数据集统计信息")
    print("="*50)
    print(f"数据集路径: {loader.root_dir}")
    print(f"文件总数: {len(loader.mat_files)}")
    print("\n类别分布:")
    for fault_type, count in sorted(loader.get_class_distribution().items()):
        label = FAULT_LABELS.get(fault_type, '?')
        print(f"  {fault_type:8s} (label={label}): {count} 个文件")
    
    if args.info:
        return
    
    # 导出数据
    print("\n" + "="*50)
    print("开始导出数据...")
    print("="*50)
    
    loader.export_numpy(
        args.out,
        segment_length=(None if args.no_seg else args.segment_length),
        overlap=args.overlap,
        normalize=not args.no_normalize,
        multi_channel=args.multi_channel,
        sensor_locations=args.sensor_locations,
        fault_types=args.fault_types,
        sampling_rates=args.sampling_rates,
    )
    
    print("\n完成!")


if __name__ == '__main__':
    main()
