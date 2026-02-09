# -*- coding: utf-8 -*-
"""
CWRU 和 XJTU 数据集批量转换工具
=================================
支持将 CWRU 和 XJTU 轴承数据集批量转换为二进制 .f 文件，
并为每个 .f 文件生成对应的元数据 JSON 文件，保存所有原始信息。

使用示例:
    # 转换 CWRU 数据集
    python 数据库转换.py --dataset_type cwru --cwru_dir CWRU-dataset-main --output_dir output
    
    # 转换 XJTU 数据集
    python 数据库转换.py --dataset_type xjtu --xjtu_dir xjtu_dataset/XJTU-SY_Bearing_Datasets --output_dir output
    
    # 同时转换两个数据集
    python 数据库转换.py --dataset_type both --output_dir output_all
    
    # 多通道转换，分通道保存
    python 数据库转换.py --dataset_type both --multi-channel --split_channels --output_dir output_mc
"""
import struct
import numpy as np
import argparse
import os
import sys
import logging
import re
import json

# 修正导入路径：将项目根目录加入 sys.path，确保可以以包形式导入 tools.load_cwru
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from tools.load_cwru import CWRUDataLoader
    from tools.load_xjtu import XJTUDataLoader
except ModuleNotFoundError:
    # 兼容从 tools 目录直接运行的情况
    from load_cwru import CWRUDataLoader
    from load_xjtu import XJTUDataLoader
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_fft_transform(data_segment):
    """对数据段应用FFT转换，返回幅值谱和原始长度信息
    
    Args:
        data_segment: 时域信号数组，可以是1D或2D（多通道）
        
    Returns:
        tuple: (fft_amplitude, original_length)
            - fft_amplitude: FFT幅值谱（归一化）
            - original_length: 原始时域信号长度
    """
    if isinstance(data_segment, np.ndarray):
        if data_segment.ndim == 1:
            # 单通道
            L = len(data_segment)
            fft_vals = np.fft.rfft(data_segment, n=L)
            amplitude = np.abs(fft_vals) / L
            return amplitude.astype(np.float32), L
        elif data_segment.ndim == 2:
            # 多通道 (C, L)
            C, L = data_segment.shape
            fft_vals = np.fft.rfft(data_segment, n=L, axis=1)
            amplitude = np.abs(fft_vals) / L
            return amplitude.astype(np.float32), L
        else:
            raise ValueError(f"Unsupported data dimension: {data_segment.ndim}")
    else:
        # 如果不是数组，转换为数组
        data_array = np.array(data_segment)
        return apply_fft_transform(data_array)


def save_to_binary_file(data, output_path):
    """将 numpy 数组保存为 32 位浮点型二进制文件"""
    try:
        # 确保数据为 float32 类型
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # 写为二进制文件
        data.tofile(output_path)
        logger.debug(f"Saved binary file: {output_path}")
    except Exception as e:
        logger.error(f"Error saving binary file {output_path}: {str(e)}")
        raise


def save_metadata_file(meta: dict, output_path: str):
    """为 .f 文件生成对应的元数据 JSON 文件，保存所有原始信息"""
    try:
        # 清理不可序列化的内容，但保留所有原始信息
        meta_clean = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, list, bool, type(None))):
                meta_clean[k] = v
            elif isinstance(v, np.integer):
                meta_clean[k] = int(v)
            elif isinstance(v, np.floating):
                meta_clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                # 对于数组，保存形状和数据类型信息
                meta_clean[k] = {
                    'shape': list(v.shape),
                    'dtype': str(v.dtype),
                    'min': float(v.min()),
                    'max': float(v.max())
                }
            elif isinstance(v, dict):
                meta_clean[k] = {}
                for kk, vv in v.items():
                    if isinstance(vv, (str, int, float, list, bool, type(None))):
                        meta_clean[k][kk] = vv
                    elif isinstance(vv, np.integer):
                        meta_clean[k][kk] = int(vv)
                    elif isinstance(vv, np.floating):
                        meta_clean[k][kk] = float(vv)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(meta_clean, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved metadata file: {output_path}")
    except Exception as e:
        logger.error(f"Error saving metadata file {output_path}: {str(e)}")


def generate_readme_file(output_dir: str, dataset_name: str, stats: dict):
    """生成数据集说明文件 README.md"""
    readme_path = os.path.join(output_dir, 'README.md')
    
    content = f"""# {dataset_name} 数据集转换说明

## 数据格式

### 二进制文件 (.f)
- 格式：32位浮点数 (float32) 二进制文件
- 字节序：小端序 (Little Endian)
- 数据布局：
  - 单通道：连续存储的一维数组 `[x0, x1, x2, ..., xN-1]`
  - 多通道（未拆分）：按通道优先 (C-contiguous) 展平，`[ch0_0, ch0_1, ..., ch0_L-1, ch1_0, ch1_1, ..., ch1_L-1, ...]`
  - 多通道（拆分模式）：每个通道单独一个文件
- **时域数据**（默认）：存储原始时域信号
- **频域数据**（使用 `--fft` 选项）：存储FFT幅值谱（归一化），文件名包含 `_fft` 后缀
  - 单通道：时域长度 L → 频域长度 (L//2 + 1)
  - 多通道：时域形状 (C, L) → 频域形状 (C, L//2 + 1)

### 元数据文件 (.json)
每个 .f 文件都有对应的 .json 元数据文件，包含以下信息：

#### 通用字段
- `binary_file`: 对应的二进制文件名
- `label`: 标签值（整数）
- `data_shape`: 数据形状列表（时域或频域）
- `data_length`: 数据点总数
- `data_dtype`: 数据类型（固定为 'float32'）
- `segment_index`: 分段索引
- `dataset`: 数据集名称（CWRU 或 XJTU-SY）
- `is_fft_data`: 是否为FFT数据（布尔值）
  - `True`: 频域数据（FFT幅值谱）
  - `False`: 时域数据（原始信号）
- **FFT数据额外字段**（当 `is_fft_data=True` 时）：
  - `fft_type`: FFT类型（固定为 'amplitude_spectrum'）
  - `original_time_length`: 原始时域信号长度
  - `original_time_shape`: 原始时域信号形状（多通道时）
  - `frequency_resolution`: 频率分辨率（Hz）= 采样率 / 原始时域长度

"""
    
    if dataset_name.upper() == 'CWRU':
        content += """#### CWRU 特定字段
- `filepath`: 原始 .mat 文件完整路径
- `filename`: 原始文件名
- `fault_type`: 故障类型（Normal, B, IR, OR）
- `fault_label`: 故障标签（0=正常, 1=滚动体, 2=内圈, 3=外圈）
- `fault_description`: 故障描述（中文）
- `fault_diameter`: 故障直径（mm）
- `sensor_location`: 传感器位置（DE, FE, BA）
- `sensor_location_description`: 传感器位置描述
- `sampling_rate`: 采样率（Hz）
- `load_hp`: 负载（马力）
- `rpm`: 转速（转/分）
- `or_position`: 外圈故障位置（@3, @6, @12，仅OR故障）
- `channels`: 通道列表（多通道模式）

### 标签说明
- `0`: Normal（正常）
- `1`: Ball（滚动体故障）
- `2`: Inner Race（内圈故障）
- `3`: Outer Race（外圈故障）

"""
    elif 'XJTU' in dataset_name.upper():
        content += """#### XJTU-SY 特定字段
- `filepath`: 原始 CSV 文件完整路径
- `filename`: 原始文件名
- `working_condition`: 工作条件（35Hz12kN, 37.5Hz11kN, 40Hz10kN）
- `speed_hz`: 转速（Hz）
- `load_kn`: 负载（kN）
- `bearing_name`: 轴承名称
- `file_number`: 文件编号
- `health_label`: 健康标签（0=健康, 1=故障）
- `sampling_rate`: 采样率（25600 Hz）
- `channels`: 通道列表（Horizontal, Vertical）

### 标签说明
- `0`: 健康状态（前30%数据）
- `1`: 故障状态（后70%数据）

"""
    
    content += f"""## 数据统计

- 总样本数：{stats.get('total_samples', 0)}
- 标签分布：{json.dumps(stats.get('label_distribution', {}), ensure_ascii=False, indent=2)}

## 如何读取 .f 文件

### Python 示例

```python
import numpy as np
import json

# 读取二进制文件
with open('example.f', 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.float32)

# 如果需要恢复形状（从元数据文件获取）
with open('example.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)
    
shape = tuple(meta['data_shape'])
if len(shape) > 1:
    # 多通道数据需要reshape
    data = data.reshape(shape)
else:
    # 单通道数据
    data = data.reshape(-1)

# 读取标签和元数据
label = meta['label']
fault_type = meta.get('fault_type', 'Unknown')
print(f"Label: {{label}}, Fault Type: {{fault_type}}")
```

### MATLAB 示例

```matlab
% 读取二进制文件
fid = fopen('example.f', 'r');
data = fread(fid, 'float32');
fclose(fid);

% 读取元数据（需要MATLAB的jsondecode，R2016b+）
meta = jsondecode(fileread('example.json'));
shape = meta.data_shape;
data = reshape(data, shape);
```

## 注意事项

1. 所有二进制文件使用 float32 格式，读取时请指定正确的数据类型
2. 多通道数据未拆分时，按通道优先顺序展平存储
3. 元数据文件使用 UTF-8 编码，包含完整的原始文件路径信息
4. 数据已进行标准化处理（除非使用了 --no_normalize 参数）
5. **FFT数据说明**：
   - 使用 `--fft` 选项时，保存的是FFT幅值谱（归一化），而非原始时域信号
   - FFT数据文件名包含 `_fft` 后缀（如 `example_fft.f`）
   - FFT幅值谱长度 = 原始时域长度 // 2 + 1（使用 `np.fft.rfft`，只保留正频率部分）
   - FFT数据可直接用于特征提取，无需再次计算FFT
   - 通过 `is_fft_data` 字段可判断数据是时域还是频域

---
生成时间：{stats.get('generation_time', 'Unknown')}
"""
    
    try:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Generated README file: {readme_path}")
    except Exception as e:
        logger.error(f"Error generating README file: {str(e)}")


def convert_to_binary_files(X_raw, y, meta, output_dir="output", split_channels: bool = False, save_metadata: bool = True, apply_fft: bool = False):
    """将数据转换为二进制文件并生成元数据
    
    Args:
        X_raw: 原始数据数组
        y: 标签数组
        meta: 元数据列表
        output_dir: 输出目录
        split_channels: 是否分通道保存
        save_metadata: 是否保存元数据
        apply_fft: 是否应用FFT转换（保存频域幅值谱而非时域信号）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    total_samples = len(X_raw)
    logger.info(f"Converting {total_samples} segments to binary files...")
    
    # 检查数据长度是否一致
    assert len(X_raw) == len(y), f"Data length mismatch: X_raw has {len(X_raw)} items, y has {len(y)} items"
    if meta is not None:
        assert len(X_raw) == len(meta), f"Data length mismatch: X_raw has {len(X_raw)} items, meta has {len(meta)} items"
    
    # 统计信息
    label_distribution = {}
    for label in y:
        label_distribution[int(label)] = label_distribution.get(int(label), 0) + 1
    
    # 使用进度显示（如果数据量较大）
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(zip(X_raw, y)), total=total_samples, desc="Converting")
    except ImportError:
        iterator = enumerate(zip(X_raw, y))
    
    for i, (data_segment, label) in iterator:
        try:
            # 构造输出文件名，使用原文件的标识信息
            # 使用 meta 信息构建文件名
            if meta is not None and len(meta) > i:
                # 提取原始文件相关信息并使用它
                meta_item = meta[i] if isinstance(meta[i], dict) else {}
                channels = meta_item.get('channels', None)
                dataset = meta_item.get('dataset', 'unknown')
                
                if 'filename' in meta_item:
                    base_name = os.path.splitext(os.path.basename(meta_item['filename']))[0]
                    output_filename = f"{dataset}_{base_name}_{i}.f"
                elif 'bearing_name' in meta_item:
                    # XJTU 数据集
                    bearing = meta_item.get('bearing_name', 'bearing')
                    file_num = meta_item.get('file_number', i)
                    output_filename = f"{dataset}_{bearing}_{file_num}_{i}.f"
                elif 'id' in meta_item:
                    output_filename = f"{dataset}_data_{meta_item['id']}_{i}.f"
                elif 'rpm' in meta_item:
                    output_filename = f"{dataset}_rpm_{int(meta_item['rpm'])}_{i}.f"
                else:
                    # 如果没有特定标识符，使用序号和标签
                    output_filename = f"{dataset}_segment_{i}_label_{int(label)}.f"
            else:
                # 默认命名方式
                output_filename = f"segment_{i}_label_{int(label)}.f"
            
            # 仅清理文件名中的非法字符，避免破坏目录分隔符
            safe_filename = re.sub(r'[<>:\"/\\|?*@]', '_', output_filename)

            # 处理多通道：按需拆分每个通道或将多通道样本展平后写出
            if isinstance(data_segment, np.ndarray) and data_segment.ndim == 2:
                C, L = data_segment.shape
                if split_channels:
                    # 每个通道单独写文件，文件名带通道后缀
                    ch_names = channels if (isinstance(channels, list) and len(channels) == C) else [f"ch{idx}" for idx in range(C)]
                    
                    # 保存元数据（拆分模式下为每个通道生成一个元数据文件）
                    if save_metadata and meta is not None and len(meta) > i:
                        meta_item = meta[i].copy() if isinstance(meta[i], dict) else {}
                        
                    for ci in range(C):
                        ch_filename = os.path.splitext(safe_filename)[0] + f"_{ch_names[ci]}.f"
                        ch_path = os.path.join(output_dir, ch_filename)
                            
                            # 应用FFT转换（如果需要）
                            ch_data = data_segment[ci]
                            original_time_length = L
                            if apply_fft:
                                ch_data, original_time_length = apply_fft_transform(ch_data)
                                ch_filename = ch_filename.replace('.f', '_fft.f')
                                ch_path = os.path.join(output_dir, ch_filename)
                            
                            save_to_binary_file(ch_data, ch_path)
                            
                            # 为每个通道生成元数据
                            ch_meta = meta_item.copy()
                            ch_meta['binary_file'] = ch_filename
                            ch_meta['label'] = int(label)
                            ch_meta['data_shape'] = list(ch_data.shape)  # FFT后的形状
                            ch_meta['data_dtype'] = 'float32'
                            ch_meta['data_length'] = int(np.prod(ch_data.shape))
                            ch_meta['segment_index'] = i
                            ch_meta['channel_index'] = ci
                            ch_meta['channel_name'] = ch_names[ci]
                            ch_meta['total_channels'] = C
                            
                            # 添加FFT信息
                            if apply_fft:
                                ch_meta['is_fft_data'] = True
                                ch_meta['fft_type'] = 'amplitude_spectrum'
                                ch_meta['original_time_length'] = original_time_length
                                sampling_rate = meta_item.get('sampling_rate', None)
                                if sampling_rate:
                                    ch_meta['frequency_resolution'] = float(sampling_rate / original_time_length)
                            else:
                                ch_meta['is_fft_data'] = False
                            
                            # 添加故障描述信息
                            if 'fault_type' in ch_meta:
                                fault_type = ch_meta['fault_type']
                                fault_desc = {
                                    'Normal': '正常',
                                    'B': '滚动体故障 (Ball Fault)',
                                    'IR': '内圈故障 (Inner Race Fault)',
                                    'OR': '外圈故障 (Outer Race Fault)'
                                }.get(fault_type, fault_type)
                                ch_meta['fault_description'] = fault_desc
                            
                            # 添加传感器位置描述
                            if 'sensor_location' in ch_meta:
                                sensor_desc = {
                                    'DE': '驱动端 (Drive End)',
                                    'FE': '风扇端 (Fan End)',
                                    'BA': '基座 (Base)'
                                }.get(ch_meta['sensor_location'], ch_meta['sensor_location'])
                                ch_meta['sensor_location_description'] = sensor_desc
                            
                            # 生成元数据文件路径
                            ch_meta_filename = os.path.splitext(ch_filename)[0] + '.json'
                            ch_meta_path = os.path.join(output_dir, ch_meta_filename)
                            save_metadata_file(ch_meta, ch_meta_path)
                    else:
                        # 不保存元数据时，只保存二进制文件
                        for ci in range(C):
                            ch_filename = os.path.splitext(safe_filename)[0] + f"_{ch_names[ci]}.f"
                            ch_data = data_segment[ci]
                            if apply_fft:
                                ch_data, _ = apply_fft_transform(ch_data)
                                ch_filename = ch_filename.replace('.f', '_fft.f')
                            ch_path = os.path.join(output_dir, ch_filename)
                            save_to_binary_file(ch_data, ch_path)
                else:
                    # 将 (C, L) 展平为一维，按通道优先（C 连续）写出
                    data_to_save = data_segment
                    original_shape = data_segment.shape
                    original_time_length = data_segment.shape[1] if data_segment.ndim == 2 else data_segment.shape[0]
                    
                    # 应用FFT转换（如果需要）
                    if apply_fft:
                        fft_data, original_time_length = apply_fft_transform(data_segment)
                        # FFT后形状变为 (C, L_fft)，需要展平
                        data_to_save = fft_data.reshape(-1)
                        safe_filename = safe_filename.replace('.f', '_fft.f')
                    else:
                        data_to_save = data_segment.reshape(-1)
                    
                    output_path = os.path.join(output_dir, safe_filename)
                    save_to_binary_file(data_to_save, output_path)
                    
                    # 保存对应的元数据文件（包含所有原始信息）
                    if save_metadata and meta is not None and len(meta) > i:
                        meta_item = meta[i].copy() if isinstance(meta[i], dict) else {}
                        
                        # 添加二进制文件信息（不覆盖原始元数据）
                        meta_item['binary_file'] = safe_filename
                        meta_item['label'] = int(label)
                        meta_item['data_shape'] = list(data_to_save.shape) if not apply_fft else list(fft_data.shape)  # FFT后的形状或多通道形状
                        meta_item['data_dtype'] = 'float32'
                        meta_item['data_length'] = int(np.prod(data_to_save.shape))
                        meta_item['segment_index'] = i
                        
                        # 添加FFT信息
                        if apply_fft:
                            meta_item['is_fft_data'] = True
                            meta_item['fft_type'] = 'amplitude_spectrum'
                            meta_item['original_time_shape'] = list(original_shape)
                            meta_item['original_time_length'] = original_time_length
                            sampling_rate = meta_item.get('sampling_rate', None)
                            if sampling_rate:
                                meta_item['frequency_resolution'] = float(sampling_rate / original_time_length)
                        else:
                            meta_item['is_fft_data'] = False
                        
                        # 添加故障描述信息
                        if 'fault_type' in meta_item:
                            fault_type = meta_item['fault_type']
                            fault_desc = {
                                'Normal': '正常',
                                'B': '滚动体故障 (Ball Fault)',
                                'IR': '内圈故障 (Inner Race Fault)',
                                'OR': '外圈故障 (Outer Race Fault)'
                            }.get(fault_type, fault_type)
                            meta_item['fault_description'] = fault_desc
                        
                        # 添加传感器位置描述
                        if 'sensor_location' in meta_item:
                            sensor_desc = {
                                'DE': '驱动端 (Drive End)',
                                'FE': '风扇端 (Fan End)',
                                'BA': '基座 (Base)'
                            }.get(meta_item['sensor_location'], meta_item['sensor_location'])
                            meta_item['sensor_location_description'] = sensor_desc
                        
                        # 生成元数据文件路径
                        meta_filename = os.path.splitext(safe_filename)[0] + '.json'
                        meta_path = os.path.join(output_dir, meta_filename)
                        save_metadata_file(meta_item, meta_path)
            else:
                # 单通道或一维样本直接写出
                data_to_save = data_segment
                original_length = len(data_segment) if hasattr(data_segment, '__len__') else np.prod(np.array(data_segment).shape)
                
                # 应用FFT转换（如果需要）
                if apply_fft:
                    data_to_save, original_length = apply_fft_transform(data_segment)
                    safe_filename = safe_filename.replace('.f', '_fft.f')
                
                output_path = os.path.join(output_dir, safe_filename)
                save_to_binary_file(data_to_save, output_path)
            
                # 保存对应的元数据文件（包含所有原始信息）
            if save_metadata and meta is not None and len(meta) > i:
                meta_item = meta[i].copy() if isinstance(meta[i], dict) else {}
                    
                    # 添加二进制文件信息（不覆盖原始元数据）
                meta_item['binary_file'] = safe_filename
                meta_item['label'] = int(label)
                    
                    # 保存数据形状信息（用于后续读取）
                    if hasattr(data_to_save, 'shape'):
                        meta_item['data_shape'] = list(data_to_save.shape)
                        meta_item['data_dtype'] = 'float32'
                        meta_item['data_length'] = int(np.prod(data_to_save.shape))
                    else:
                        meta_item['data_shape'] = [len(data_to_save)]
                        meta_item['data_dtype'] = 'float32'
                        meta_item['data_length'] = len(data_to_save)
                    
                meta_item['segment_index'] = i
                
                    # 添加FFT信息
                    if apply_fft:
                        meta_item['is_fft_data'] = True
                        meta_item['fft_type'] = 'amplitude_spectrum'
                        meta_item['original_time_length'] = int(original_length)
                        sampling_rate = meta_item.get('sampling_rate', None)
                        if sampling_rate:
                            meta_item['frequency_resolution'] = float(sampling_rate / original_length)
                    else:
                        meta_item['is_fft_data'] = False
                    
                    # 添加故障描述信息（方便理解）
                    if 'fault_type' in meta_item:
                        fault_type = meta_item['fault_type']
                        fault_desc = {
                            'Normal': '正常',
                            'B': '滚动体故障 (Ball Fault)',
                            'IR': '内圈故障 (Inner Race Fault)',
                            'OR': '外圈故障 (Outer Race Fault)'
                        }.get(fault_type, fault_type)
                        meta_item['fault_description'] = fault_desc
                    
                    # 添加传感器位置描述
                    if 'sensor_location' in meta_item:
                        sensor_desc = {
                            'DE': '驱动端 (Drive End)',
                            'FE': '风扇端 (Fan End)',
                            'BA': '基座 (Base)'
                        }.get(meta_item['sensor_location'], meta_item['sensor_location'])
                        meta_item['sensor_location_description'] = sensor_desc
                    
                # 生成元数据文件路径
                meta_filename = os.path.splitext(safe_filename)[0] + '.json'
                meta_path = os.path.join(output_dir, meta_filename)
                save_metadata_file(meta_item, meta_path)
            
        except Exception as e:
            logger.error(f"Error processing segment {i}: {str(e)}")
            # 出现错误继续下一个，避免中断整个过程
            continue
    
    # 返回统计信息
    stats = {
        'total_samples': len(X_raw),
        'label_distribution': label_distribution
    }
    return stats


# 示例使用
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='CWRU & XJTU Dataset Batch Conversion Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本转换（分段，默认2048长度，50%重叠）
  python 数据库转换.py --dataset_type both --output_dir output

  # 不分段，导出完整信号
  python 数据库转换.py --dataset_type cwru --no_seg --output_dir output_full

  # 自定义分段长度和重叠比例
  python 数据库转换.py --dataset_type cwru --segment_length 1024 --overlap 0.3 --output_dir output_1024

  # 按故障类型过滤（仅CWRU）
  python 数据库转换.py --dataset_type cwru --fault_types IR OR --output_dir output_ir_or

  # 按采样率过滤（仅CWRU）
  python 数据库转换.py --dataset_type cwru --sampling_rates 12000 --output_dir output_12k

  # 多通道转换
  python 数据库转换.py --dataset_type both --multi-channel --sensor_locations DE FE --output_dir output_mc

  # FFT预计算（保存频域幅值谱）
  python 数据库转换.py --dataset_type both --fft --output_dir output_fft

  # FFT预计算 + 多通道
  python 数据库转换.py --dataset_type both --fft --multi-channel --output_dir output_fft_mc
        """
    )
    
    # 数据集选择
    parser.add_argument('--dataset_type', type=str, default='cwru', 
                        choices=['cwru', 'xjtu', 'both'],
                        help='Dataset type: cwru, xjtu, or both (default: cwru)')
    parser.add_argument('--cwru_dir', type=str, default='datasets/CWRU-dataset-main', 
                        help='Path to CWRU dataset')
    parser.add_argument('--xjtu_dir', type=str, default='datasets/xjtu_dataset/XJTU-SY_Bearing_Datasets',
                        help='Path to XJTU dataset')
    
    # 数据处理参数
    parser.add_argument('--segment_length', type=int, default=2048, 
                        help='Signal segment length (default: 2048). Use --no_seg/--no-seg to disable segmentation')
    parser.add_argument('--no_seg', action='store_true', help='Disable segmentation and export full-length signals')
    parser.add_argument('--no-seg', dest='no_seg', action='store_true', help='Alias of --no_seg')
    parser.add_argument('--overlap', type=float, default=0.5, help='Segmentation overlap ratio (default: 0.5)')
    parser.add_argument('--multi_channel', action='store_true', help='Enable multi-channel loading (DE/FE/BA)')
    parser.add_argument('--multi-channel', dest='multi_channel', action='store_true', help='Alias of --multi_channel')
    parser.add_argument('--sensor_locations', nargs='+', default=None, help='Select sensor locations, e.g., DE FE BA')
    parser.add_argument('--sensor-locations', dest='sensor_locations', nargs='+', default=None, help='Alias of --sensor_locations')
    parser.add_argument('--sampling_rates', nargs='+', type=int, default=None, help='Filter by sampling rates, e.g., 12000 48000 (CWRU only)')
    parser.add_argument('--fault_types', nargs='+', default=None, 
                        choices=['Normal', 'B', 'IR', 'OR'],
                        help='Filter by fault types: Normal B IR OR (CWRU only)')
    parser.add_argument('--fault-types', dest='fault_types', nargs='+', default=None,
                        choices=['Normal', 'B', 'IR', 'OR'],
                        help='Alias of --fault_types')
    parser.add_argument('--no_normalize', action='store_true', help='Disable normalization (default: normalize on)')
    parser.add_argument('--split_channels', action='store_true', help='Write each channel as a separate .f file')
    parser.add_argument('--health_ratio', type=float, default=0.3, 
                        help='Health sample ratio for XJTU dataset (default: 0.3)')
    parser.add_argument('--fft', action='store_true', 
                        help='Apply FFT transformation and save frequency domain (amplitude spectrum) instead of time domain signals')
    
    # 输出选项
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for binary files')
    parser.add_argument('--no_metadata', action='store_true', 
                        help='Do not generate metadata JSON files for each .f file')
    
    # 解析参数
    args = parser.parse_args()
    
    try:
        segment_length = None if args.no_seg else args.segment_length
        datasets_to_process = []
        
        # 根据 dataset_type 确定要处理的数据集
        if args.dataset_type in ['cwru', 'both']:
            datasets_to_process.append(('cwru', args.cwru_dir))
        if args.dataset_type in ['xjtu', 'both']:
            datasets_to_process.append(('xjtu', args.xjtu_dir))
        
        all_converted_count = 0
        
        for dataset_name, dataset_dir in datasets_to_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {dataset_name.upper()} dataset from {dataset_dir}...")
            logger.info(f"{'='*60}")
            
            # 创建数据集特定的输出目录
            output_dir = os.path.join(args.output_dir, dataset_name)
            
            if dataset_name == 'cwru':
                # 加载 CWRU 数据
                loader = CWRUDataLoader(dataset_dir)
                X_raw, y, meta = loader.load_all(
                    segment_length=segment_length,
                    overlap=args.overlap,
                    fault_types=args.fault_types,
                    sampling_rates=args.sampling_rates,
                    normalize=not args.no_normalize,
                    multi_channel=args.multi_channel,
                    sensor_locations=args.sensor_locations,
                )
                # 为元数据添加数据集标识
                for m in meta:
                    m['dataset'] = 'CWRU'
                
            elif dataset_name == 'xjtu':
                # 加载 XJTU 数据
                loader = XJTUDataLoader(dataset_dir)
                X_raw, y, meta = loader.load_all(
                    segment_length=segment_length,
                    overlap=args.overlap,
                    normalize=not args.no_normalize,
                    multi_channel=args.multi_channel,
                    health_ratio=args.health_ratio,
                )
                # XJTU 元数据已经包含 'dataset' 字段
            
            # 输出数据统计
            if isinstance(X_raw, list):
                shapes = []
                for item in X_raw[:5]:
                    try:
                        shapes.append(getattr(item, 'shape', None))
                    except Exception:
                        shapes.append(None)
                logger.info(f"X_raw is a list with {len(X_raw)} items, sample shapes (first 5): {shapes}...")
            else:
                logger.info(f"X_raw shape: {X_raw.shape}")
            
            logger.info(f"Loaded {len(X_raw)} data segments, labels: {len(y)}, meta: {len(meta) if meta is not None else 'None'}")
            
            # 转换为二进制文件
            stats = convert_to_binary_files(
                X_raw, y, meta, output_dir, 
                split_channels=args.split_channels,
                save_metadata=not args.no_metadata,
                apply_fft=args.fft
            )
            
            # 添加生成时间
            from datetime import datetime
            stats['generation_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 生成说明文件
            if not args.no_metadata:
                generate_readme_file(output_dir, dataset_name, stats)
            
            all_converted_count += len(X_raw)
            logger.info(f"{dataset_name.upper()} dataset conversion completed. Files saved to {output_dir}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"All datasets conversion completed!")
        logger.info(f"Total samples converted: {all_converted_count}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
