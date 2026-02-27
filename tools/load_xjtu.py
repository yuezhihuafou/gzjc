#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XJTU-SY 轴承故障数据集批量加载脚本
===============================
此脚本用于批量读取 XJTU-SY (Xi'an Jiaotong University) 轴承数据集的 CSV 文件，
提取振动信号和元数据，便于后续集成到故障诊断智能模型中。

XJTU-SY 数据集结构：
    - 三种工作条件：35Hz12kN、37.5Hz11kN、40Hz10kN
    - 每个工作条件下有多个轴承（Bearing1_1 到 Bearing1_5 等）
    - 每个轴承目录包含多个 CSV 文件（全寿命周期数据）
    - 每个 CSV 文件包含两列：Horizontal_vibration_signals、Vertical_vibration_signals
    - 采样频率：25.6 kHz
    - 每个文件采样长度：1.28 秒（32768 个数据点）

使用示例:
    # 命令行使用
    python load_xjtu.py ../xjtu_dataset/XJTU-SY_Bearing_Datasets ./output --segment-length 2048
    
    # Python 代码中使用
    from load_xjtu import XJTUDataLoader
    loader = XJTUDataLoader('xjtu_dataset/XJTU-SY_Bearing_Datasets')
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
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 工况映射
# =============================================================================
WORKING_CONDITIONS = {
    '35Hz12kN': {'speed_hz': 35, 'load_kn': 12},
    '37.5Hz11kN': {'speed_hz': 37.5, 'load_kn': 11},
    '40Hz10kN': {'speed_hz': 40, 'load_kn': 10},
}

# XJTU 数据集标签策略：
# 根据轴承运行时间/文件编号判断健康状态
# 通常前期文件为健康，后期为故障
# 这里采用简化策略：文件编号前 30% 为健康(0)，后 70% 为故障(1)
HEALTH_LABEL = 0  # 健康
FAULT_LABEL = 1   # 故障


# =============================================================================
# 核心函数
# =============================================================================
def load_csv_signal(filepath: str, multi_channel: bool = False) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    加载 XJTU CSV 文件并提取振动信号。
    
    Args:
        filepath: CSV 文件路径
        multi_channel: 是否返回多通道数据（默认 False 只返回水平通道）
        
    Returns:
        (signal, channels): 信号数组和通道名称列表
            - 单通道：(N,) 和 ['Horizontal']
            - 多通道：(2, N) 和 ['Horizontal', 'Vertical']
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.warning(f"加载文件失败 {filepath}: {e}")
        return None, []
    
    if 'Horizontal_vibration_signals' not in df.columns:
        logger.warning(f"文件格式不正确 {filepath}")
        return None, []
    
    horizontal = df['Horizontal_vibration_signals'].values.astype(np.float32)
    
    if multi_channel and 'Vertical_vibration_signals' in df.columns:
        vertical = df['Vertical_vibration_signals'].values.astype(np.float32)
        # 确保长度一致
        min_len = min(len(horizontal), len(vertical))
        signal = np.stack([horizontal[:min_len], vertical[:min_len]], axis=0)  # (2, N)
        channels = ['Horizontal', 'Vertical']
    else:
        signal = horizontal
        channels = ['Horizontal']
    
    return signal, channels


def parse_filepath_metadata(root: str, filepath: str) -> Dict[str, Any]:
    """
    从文件路径解析元数据。
    
    路径结构示例:
        35Hz12kN/Bearing1_1/1.csv
        37.5Hz11kN/Bearing1_3/50.csv
    
    Args:
        root: 数据集根目录
        filepath: CSV 文件完整路径
        
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
        'dataset': 'XJTU-SY',
        'sampling_rate': 25600,  # 25.6 kHz
        'working_condition': None,
        'speed_hz': None,
        'load_kn': None,
        'bearing_name': None,
        'file_number': None,
        'health_label': None,
    }
    
    # 解析工作条件
    if len(parts) > 0:
        wc = parts[0]
        if wc in WORKING_CONDITIONS:
            meta['working_condition'] = wc
            meta['speed_hz'] = WORKING_CONDITIONS[wc]['speed_hz']
            meta['load_kn'] = WORKING_CONDITIONS[wc]['load_kn']
    
    # 解析轴承名称
    if len(parts) > 1:
        meta['bearing_name'] = parts[1]
    
    # 解析文件编号
    file_num_match = re.search(r'(\d+)\.csv$', filename)
    if file_num_match:
        meta['file_number'] = int(file_num_match.group(1))
    
    return meta


def assign_health_label(bearing_files: List[str], ratio: float = 0.3) -> Dict[str, int]:
    """
    根据文件编号分配健康标签。
    前 ratio 的文件标记为健康，后续为故障。
    
    Args:
        bearing_files: 同一轴承的所有文件路径列表（已排序）
        ratio: 健康文件比例（默认 0.3）
        
    Returns:
        {filepath: label} 映射
    """
    labels = {}
    total = len(bearing_files)
    health_count = int(total * ratio)
    
    for i, filepath in enumerate(bearing_files):
        labels[filepath] = HEALTH_LABEL if i < health_count else FAULT_LABEL
    
    return labels


def segment_signal(signal: np.ndarray, segment_length: int, 
                   overlap: float = 0.5) -> np.ndarray:
    """
    将长信号切分为固定长度的片段。
    
    Args:
        signal: 原始信号 (1D 或 2D: C,L)
        segment_length: 每个片段的长度
        overlap: 重叠比例 (0-1)
        
    Returns:
        切分后的信号数组
            - 单通道输入 (L,) -> (N, segment_length)
            - 多通道输入 (C, L) -> (N, C, segment_length)
    """
    if signal.ndim == 1:
        # 单通道
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
    
    else:
        # 多通道 (C, L)
        C, L = signal.shape
        step = int(segment_length * (1 - overlap))
        if step <= 0:
            step = 1
        
        segments = []
        for start in range(0, L - segment_length + 1, step):
            segments.append(signal[:, start:start + segment_length])
        
        if len(segments) == 0:
            # 信号太短，零填充
            padded = np.zeros((C, segment_length), dtype=signal.dtype)
            padded[:, :L] = signal
            segments.append(padded)
        
        return np.array(segments)


# =============================================================================
# 数据加载器类
# =============================================================================
class XJTUDataLoader:
    """
    XJTU-SY 数据集加载器，用于批量读取和预处理数据。
    
    使用示例:
        loader = XJTUDataLoader('xjtu_dataset/XJTU-SY_Bearing_Datasets')
        
        # 加载所有数据
        X, y, meta = loader.load_all()
        
        # 加载并分段
        X, y, meta = loader.load_all(segment_length=2048, overlap=0.5)
        
        # 多通道加载
        X, y, meta = loader.load_all(multi_channel=True)
    """
    
    def __init__(self, root_dir: str):
        """
        初始化加载器。
        
        Args:
            root_dir: XJTU-SY 数据集根目录
        """
        self.root_dir = os.path.abspath(root_dir)
        self.csv_files = self._scan_csv_files()
        logger.info(f"扫描到 {len(self.csv_files)} 个 CSV 文件")
    
    def _scan_csv_files(self) -> List[str]:
        """递归扫描所有 CSV 文件"""
        csv_files = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fn in filenames:
                if fn.lower().endswith('.csv'):
                    csv_files.append(os.path.join(dirpath, fn))
        csv_files.sort()
        return csv_files
    
    def _assign_labels(self, health_ratio: float = 0.3) -> Dict[str, int]:
        """为所有文件分配健康标签"""
        # 按轴承分组
        bearing_groups: Dict[str, List[str]] = {}
        for filepath in self.csv_files:
            meta = parse_filepath_metadata(self.root_dir, filepath)
            bearing_key = f"{meta['working_condition']}_{meta['bearing_name']}"
            if bearing_key not in bearing_groups:
                bearing_groups[bearing_key] = []
            bearing_groups[bearing_key].append(filepath)
        
        # 为每组分配标签
        all_labels = {}
        for bearing_key, files in bearing_groups.items():
            # 按文件编号排序
            files_sorted = sorted(files, key=lambda f: parse_filepath_metadata(self.root_dir, f)['file_number'] or 0)
            labels = assign_health_label(files_sorted, health_ratio)
            all_labels.update(labels)
        
        return all_labels
    
    def load_all(self, 
                 segment_length: Optional[int] = None,
                 overlap: float = 0.5,
                 normalize: bool = False,
                 multi_channel: bool = False,
                 health_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        批量加载所有数据。
        
        Args:
            segment_length: 分段长度，None 表示不分段
            overlap: 分段重叠比例
            normalize: 是否对信号进行标准化
            multi_channel: 是否加载多通道（水平+垂直）
            health_ratio: 健康样本比例（用于自动标注）
            
        Returns:
            X: 信号数组或列表
            y: 标签数组
            meta: 元数据列表
        """
        all_signals = []
        all_labels = []
        all_meta = []
        
        # 分配标签
        file_labels = self._assign_labels(health_ratio)
        
        for filepath in self.csv_files:
            signal, channels = load_csv_signal(filepath, multi_channel=multi_channel)
            if signal is None:
                continue
            
            meta = parse_filepath_metadata(self.root_dir, filepath)
            meta['health_label'] = file_labels.get(filepath, FAULT_LABEL)
            meta['channels'] = channels
            
            label = meta['health_label']
            
            # 标准化
            if normalize:
                if signal.ndim == 1:
                    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
                else:
                    # 多通道逐通道标准化
                    for c in range(signal.shape[0]):
                        signal[c] = (signal[c] - np.mean(signal[c])) / (np.std(signal[c]) + 1e-8)
            
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
            X = all_signals  # 保持列表（长度可能不同）
        y = np.array(all_labels, dtype=np.int64)
        
        info_str = ''
        if segment_length and isinstance(X, np.ndarray):
            info_str = f", shape={X.shape}"
        elif not segment_length:
            info_str = f", list mode"
        
        logger.info(f"加载完成: {len(all_labels)} 个样本{info_str}, "
                    f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, all_meta
    
    def get_bearing_distribution(self) -> Dict[str, int]:
        """获取各轴承的文件数量分布"""
        dist = {}
        for filepath in self.csv_files:
            meta = parse_filepath_metadata(self.root_dir, filepath)
            bearing_key = f"{meta['working_condition']}_{meta['bearing_name']}"
            dist[bearing_key] = dist.get(bearing_key, 0) + 1
        return dist


# =============================================================================
# PyTorch Dataset (可选)
# =============================================================================
try:
    import torch
    from torch.utils.data import Dataset
    
    class XJTUDataset(Dataset):
        """
        PyTorch Dataset 封装，可直接用于 DataLoader。
        """
        
        def __init__(self, root_dir: str, segment_length: int = 2048,
                     overlap: float = 0.5, normalize: bool = True,
                     multi_channel: bool = False, transform=None):
            self.loader = XJTUDataLoader(root_dir)
            self.X, self.y, self.meta = self.loader.load_all(
                segment_length=segment_length,
                overlap=overlap,
                normalize=normalize,
                multi_channel=multi_channel,
            )
            self.transform = transform
        
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, idx):
            x = self.X[idx]
            y = self.y[idx]
            
            if self.transform:
                x = self.transform(x)
            
            x = torch.from_numpy(x)
            if x.ndim == 1:
                x = x.unsqueeze(0)  # 添加通道维度
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
        description='XJTU-SY 轴承故障数据集批量加载工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('root', help='XJTU-SY 数据集根目录')
    parser.add_argument('out', nargs='?', default='./xjtu_processed',
                        help='输出目录 (默认: ./xjtu_processed)')
    parser.add_argument('--info', action='store_true',
                        help='只显示数据集信息，不导出')
    parser.add_argument('--segment-length', type=int, default=2048,
                        help='分段长度 (默认: 2048)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='分段重叠比例 (默认: 0.5)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='不进行标准化')
    parser.add_argument('--multi-channel', action='store_true',
                        help='启用多通道加载（水平+垂直）')
    parser.add_argument('--health-ratio', type=float, default=0.3,
                        help='健康样本比例 (默认: 0.3)')
    
    args = parser.parse_args()
    
    loader = XJTUDataLoader(args.root)
    
    # 显示数据集信息
    print("\n" + "="*50)
    print("XJTU-SY 数据集统计信息")
    print("="*50)
    print(f"数据集路径: {loader.root_dir}")
    print(f"文件总数: {len(loader.csv_files)}")
    print("\n轴承分布:")
    for bearing, count in sorted(loader.get_bearing_distribution().items()):
        print(f"  {bearing:20s}: {count} 个文件")
    
    if args.info:
        return
    
    # 导出数据
    print("\n" + "="*50)
    print("开始导出数据...")
    print("="*50)
    
    X, y, meta = loader.load_all(
        segment_length=args.segment_length,
        overlap=args.overlap,
        normalize=not args.no_normalize,
        multi_channel=args.multi_channel,
        health_ratio=args.health_ratio,
    )
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / 'signals.npy', X)
    np.save(out_dir / 'labels.npy', y)
    
    # 保存元数据
    meta_clean = []
    for m in meta:
        m_clean = {k: v for k, v in m.items() if isinstance(v, (str, int, float, list, type(None)))}
        meta_clean.append(m_clean)
    
    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta_clean, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据已导出到 {out_dir}")
    logger.info(f"  - signals.npy: shape={X.shape if isinstance(X, np.ndarray) else 'list'}")
    logger.info(f"  - labels.npy: shape={y.shape}")
    logger.info(f"  - metadata.json: {len(meta_clean)} 条记录")
    
    print("\n完成!")


if __name__ == '__main__':
    main()
