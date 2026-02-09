"""
声音能量曲线数据加载器 - 从API获取数据

支持从API实时获取声音能量和密度曲线数据，而不是从本地xlsx文件读取。
适用于需要实时处理音频文件的场景。
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

# 导入API转换工具
from tools.sound_api.convert_sound_api import (
    get_default_config,
    test_sound_api,
    parse_api_response
)


class SoundAPIDataLoader:
    """
    从API获取声音能量曲线数据的加载器
    
    功能：
    1. 调用API将音频文件转换为能量和密度曲线
    2. 缓存API响应结果（可选）
    3. 兼容 SoundDataLoader 的接口
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        headers: Optional[Dict] = None,
        form_data_params: Optional[Dict] = None,
        file_param_name: str = 'files',
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Args:
            api_url: API接口URL，如果为None则使用默认配置
            headers: API请求头（用于认证等）
            form_data_params: API表单参数
            file_param_name: 文件参数名
            cache_dir: 缓存目录，用于保存API响应结果
            use_cache: 是否使用缓存（如果缓存存在则直接读取，不调用API）
        """
        # 获取API配置
        if api_url is None:
            api_url, default_headers, default_form_params, default_file_param = get_default_config()
            if headers is None:
                headers = default_headers
            if form_data_params is None:
                form_data_params = default_form_params
            if file_param_name == 'files':
                file_param_name = default_file_param
        
        self.api_url = api_url
        self.headers = headers or {}
        self.form_data_params = form_data_params or {}
        self.file_param_name = file_param_name
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # 创建缓存目录
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, audio_file_path: str) -> Optional[str]:
        """获取缓存文件路径"""
        if not self.cache_dir:
            return None
        
        filename_base = os.path.splitext(os.path.basename(audio_file_path))[0]
        cache_file = os.path.join(self.cache_dir, f"{filename_base}.json")
        return cache_file
    
    def _load_from_cache(self, cache_path: str) -> Optional[Dict]:
        """从缓存加载数据"""
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 转换为numpy数组
                return {
                    'frequency': np.array(data['frequency'], dtype=np.float32),
                    'volume': np.array(data['volume'], dtype=np.float32),
                    'density': np.array(data['density'], dtype=np.float32)
                }
        except Exception as e:
            print(f"Warning: 无法从缓存加载 {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, cache_path: str, data: Dict):
        """保存数据到缓存"""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'frequency': data['frequency'].tolist(),
                    'volume': data['volume'].tolist(),
                    'density': data['density'].tolist()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: 无法保存缓存到 {cache_path}: {e}")
    
    def load_sound_curves(self, audio_file_path: str) -> Optional[Dict]:
        """
        从API加载声音能量曲线数据
        
        Args:
            audio_file_path: 音频文件路径（如 .wav, .mp3 等）
            
        Returns:
            dict: {
                'frequency': np.array,  # 频率 (Hz)
                'volume': np.array,      # 音量/能量
                'density': np.array     # 密度
            } 或 None
        """
        # 检查缓存
        if self.use_cache:
            cache_path = self._get_cache_path(audio_file_path)
            if cache_path:
                cached_data = self._load_from_cache(cache_path)
                if cached_data is not None:
                    return cached_data
        
        # 调用API
        result = test_sound_api(
            audio_file_path,
            api_url=self.api_url,
            headers=self.headers,
            file_param_name=self.file_param_name,
            form_data_params=self.form_data_params
        )
        
        if result is None:
            return None
        
        # 解析API响应
        data = parse_api_response(result, verbose=False)
        
        if data is None:
            return None
        
        # 保存到缓存
        if self.use_cache and cache_path:
            self._save_to_cache(cache_path, data)
        
        return data
    
    def get_available_files(self, audio_dir: str) -> List[str]:
        """
        获取音频目录中所有可用的音频文件
        
        Args:
            audio_dir: 音频文件目录
            
        Returns:
            List[str]: 音频文件路径列表
        """
        audio_dir = Path(audio_dir)
        if not audio_dir.exists():
            return []
        
        # 支持的音频格式
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.pcm'}
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_dir.glob(f'*{ext}'))
            audio_files.extend(audio_dir.glob(f'*{ext.upper()}'))
        
        return [str(f) for f in sorted(audio_files)]


class SoundAPIDataset(Dataset):
    """
    声音能量曲线数据集 - 从API获取
    
    从API实时获取声音数据，转换为双通道格式 (volume, density)
    形状: (2, L)，其中 L 通常是 3000
    """
    
    def __init__(
        self,
        api_loader: SoundAPIDataLoader,
        audio_files: List[str],
        labels: np.ndarray,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
    ):
        """
        Args:
            api_loader: SoundAPIDataLoader 实例
            audio_files: 音频文件路径列表
            labels: 对应的标签数组 (N,)
            channel_mean: (2,) 通道均值，用于标准化
            channel_std: (2,) 通道标准差，用于标准化
        """
        self.api_loader = api_loader
        self.audio_files = audio_files
        self.labels = labels.astype(np.int64)
        self.channel_mean = channel_mean.reshape(2, 1)
        self.channel_std = channel_std.reshape(2, 1)
        
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        # 从API加载声音曲线
        curves = self.api_loader.load_sound_curves(audio_file)
        if curves is None:
            raise ValueError(f"无法从API加载样本: {audio_file}")
        
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
    audio_files: List[str],
    metadata_path: str = "cwru_processed/metadata.json",
) -> Optional[np.ndarray]:
    """
    从 metadata.json 中根据文件名匹配标签
    
    Args:
        audio_files: 音频文件路径列表
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
        filename = item['filename'].replace('.mat', '').replace('.wav', '')
        label = item.get('fault_label', -1)
        name_to_label[filename] = label
    
    # 匹配音频文件到标签
    labels = []
    unmatched = []
    for audio_file in audio_files:
        filename_base = os.path.splitext(os.path.basename(audio_file))[0]
        
        if filename_base in name_to_label:
            labels.append(name_to_label[filename_base])
        else:
            # 尝试去掉后缀匹配（如 '97_Normal_0' -> '97_Normal'）
            base_name = '_'.join(filename_base.split('_')[:-1])
            if base_name in name_to_label:
                labels.append(name_to_label[base_name])
            else:
                unmatched.append(filename_base)
                labels.append(-1)  # 未知标签
    
    if unmatched:
        print(f"Warning: {len(unmatched)} 个样本无法匹配标签: {unmatched[:5]}...")
    
    labels = np.array(labels)
    if (labels == -1).any():
        print(f"Warning: 存在 {np.sum(labels == -1)} 个未匹配的样本")
    
    return labels


def get_sound_api_dataloaders(
    audio_dir: str,
    batch_size: int = 64,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    metadata_path: str = "cwru_processed/metadata.json",
    api_url: Optional[str] = None,
    headers: Optional[Dict] = None,
    form_data_params: Optional[Dict] = None,
    cache_dir: Optional[str] = "sound_api_cache",
    use_cache: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建基于API声音数据的 train/val/test DataLoader
    
    Args:
        audio_dir: 音频文件目录
        batch_size: 批大小
        split_ratio: (train, val, test) 比例
        metadata_path: metadata.json 路径，用于获取标签
        api_url: API接口URL（可选，使用默认配置）
        headers: API请求头（可选）
        form_data_params: API表单参数（可选）
        cache_dir: 缓存目录，用于保存API响应
        use_cache: 是否使用缓存
        num_workers: DataLoader 工作进程数
        shuffle: 是否打乱
        pin_memory: 是否使用 pin_memory
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 初始化API数据加载器
    api_loader = SoundAPIDataLoader(
        api_url=api_url,
        headers=headers,
        form_data_params=form_data_params,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
    
    # 获取所有音频文件
    audio_files = api_loader.get_available_files(audio_dir)
    
    if len(audio_files) == 0:
        raise ValueError(f"未找到任何音频文件在目录: {audio_dir}")
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 从 metadata 加载标签
    labels = _load_labels_from_metadata(audio_files, metadata_path)
    if labels is None:
        raise ValueError("无法从 metadata.json 加载标签，请检查文件路径")
    
    # 过滤掉标签为 -1 的样本（未匹配的）
    valid_mask = labels != -1
    valid_files = [f for f, m in zip(audio_files, valid_mask) if m]
    valid_labels = labels[valid_mask]
    
    if len(valid_files) == 0:
        raise ValueError("没有有效的样本（所有样本都无法匹配标签）")
    
    print(f"有效样本数: {len(valid_files)} (已过滤 {len(audio_files) - len(valid_files)} 个未匹配样本)")
    
    n_samples = len(valid_files)
    
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
    
    # 在训练集上计算通道均值/方差（需要先调用API获取数据）
    print("计算训练集统计量（从API获取数据）...")
    train_volumes = []
    train_densities = []
    
    for idx in train_indices:
        audio_file = valid_files[idx]
        curves = api_loader.load_sound_curves(audio_file)
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
    train_files = [valid_files[i] for i in train_indices]
    val_files = [valid_files[i] for i in val_indices]
    test_files = [valid_files[i] for i in test_indices]
    
    train_labels = valid_labels[train_indices]
    val_labels = valid_labels[val_indices]
    test_labels = valid_labels[test_indices]
    
    train_dataset = SoundAPIDataset(api_loader, train_files, train_labels, channel_mean, channel_std)
    val_dataset = SoundAPIDataset(api_loader, val_files, val_labels, channel_mean, channel_std)
    test_dataset = SoundAPIDataset(api_loader, test_files, test_labels, channel_mean, channel_std)
    
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


__all__ = ["SoundAPIDataLoader", "SoundAPIDataset", "get_sound_api_dataloaders"]
