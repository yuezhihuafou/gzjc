# -*- coding: utf-8 -*-
"""
XJTU MC双通道文件转音频文件批量转换脚本
将.f格式的双通道数据转换为.wav音频文件，用于后续API处理
"""
import os
import json
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from tqdm import tqdm


def load_binary_signal(binary_file, json_file):
    """
    从.f二进制文件和对应的.json元数据文件中加载信号
    
    Args:
        binary_file: .f二进制文件路径
        json_file: .json元数据文件路径
    
    Returns:
        tuple: (data, metadata) 或 (None, None)
               data: numpy数组，形状为 (channels, length)
               metadata: 字典，包含元数据信息
    """
    try:
        # 读取元数据
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 获取数据参数
        data_shape = tuple(metadata['data_shape'])  # (channels, length)
        data_dtype = metadata['data_dtype']
        
        # 读取二进制数据
        data = np.fromfile(binary_file, dtype=data_dtype)
        data = data.reshape(data_shape)
        
        return data, metadata
    
    except Exception as e:
        print(f"错误: 加载文件失败 {binary_file}: {e}")
        return None, None


def normalize_signal(signal, method='minmax'):
    """
    归一化信号到[-1, 1]范围（音频标准范围）
    
    Args:
        signal: numpy数组
        method: 归一化方法 'minmax' 或 'zscore'
    
    Returns:
        numpy数组，归一化后的信号
    """
    if method == 'minmax':
        # Min-Max归一化到[-1, 1]
        min_val = signal.min()
        max_val = signal.max()
        if max_val > min_val:
            signal_norm = 2 * (signal - min_val) / (max_val - min_val) - 1
        else:
            signal_norm = np.zeros_like(signal)
    elif method == 'zscore':
        # Z-Score归一化后裁剪到[-1, 1]
        mean = signal.mean()
        std = signal.std()
        if std > 0:
            signal_norm = (signal - mean) / std
            signal_norm = np.clip(signal_norm / 3, -1, 1)  # 3-sigma规则
        else:
            signal_norm = np.zeros_like(signal)
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    return signal_norm


def convert_to_wav(data, output_file, sampling_rate=25600, channel_mode='horizontal',
                   normalize_method='minmax'):
    """
    将双通道振动信号转换为WAV音频文件
    
    Args:
        data: numpy数组，形状为 (2, length)，两个通道分别是水平和垂直方向
        output_file: 输出WAV文件路径
        sampling_rate: 采样率
        channel_mode: 通道选择模式
                     'horizontal' - 只使用水平通道（通道0）
                     'vertical' - 只使用垂直通道（通道1）
                     'stereo' - 双通道立体声
                     'mix' - 混合两个通道（单声道）
        normalize_method: 归一化方法 'minmax' 或 'zscore'
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', 
                   exist_ok=True)
        
        # 根据通道模式选择或处理数据
        if channel_mode == 'horizontal':
            signal = data[0]  # 水平通道
        elif channel_mode == 'vertical':
            signal = data[1]  # 垂直通道
        elif channel_mode == 'mix':
            signal = (data[0] + data[1]) / 2  # 混合
        elif channel_mode == 'stereo':
            signal = data.T  # 转置为 (length, 2) 用于立体声
        else:
            raise ValueError(f"不支持的通道模式: {channel_mode}")
        
        # 归一化
        if channel_mode == 'stereo':
            # 对每个通道分别归一化
            signal_norm = np.zeros_like(signal)
            for i in range(signal.shape[1]):
                signal_norm[:, i] = normalize_signal(signal[:, i], normalize_method)
        else:
            signal_norm = normalize_signal(signal, normalize_method)
        
        # 转换为16位整数（WAV标准格式）
        signal_int16 = (signal_norm * 32767).astype(np.int16)
        
        # 保存为WAV文件
        wavfile.write(output_file, sampling_rate, signal_int16)
        
        return True
    
    except Exception as e:
        print(f"错误: 转换WAV失败 {output_file}: {e}")
        return False


def batch_convert_mc_to_wav(input_dir, output_dir, channel_mode='horizontal',
                            normalize_method='minmax', file_pattern='XJTU-SY_*.f'):
    """
    批量转换MC双通道文件为WAV音频文件
    
    Args:
        input_dir: 输入目录，包含.f和.json文件
        output_dir: 输出目录，保存WAV文件
        channel_mode: 通道模式（见convert_to_wav函数说明）
        normalize_method: 归一化方法
        file_pattern: 文件匹配模式
    
    Returns:
        dict: 转换结果统计
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有.f文件
    input_path = Path(input_dir)
    binary_files = list(input_path.glob(file_pattern))
    
    if not binary_files:
        print(f"错误: 在 {input_dir} 中未找到匹配 {file_pattern} 的文件")
        return {'success': 0, 'failed': 0, 'files': []}
    
    results = {
        'success': 0,
        'failed': 0,
        'files': []
    }
    
    print(f"\n开始批量转换 {len(binary_files)} 个文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"通道模式: {channel_mode}")
    print(f"归一化方法: {normalize_method}\n")
    
    for binary_file in tqdm(binary_files, desc="转换中"):
        # 构建对应的JSON文件路径
        json_file = binary_file.with_suffix('.json')
        
        if not json_file.exists():
            print(f"警告: 未找到元数据文件 {json_file}")
            results['failed'] += 1
            results['files'].append({
                'file': str(binary_file),
                'status': 'failed',
                'reason': 'missing_json'
            })
            continue
        
        # 加载信号数据
        data, metadata = load_binary_signal(str(binary_file), str(json_file))
        
        if data is None:
            results['failed'] += 1
            results['files'].append({
                'file': str(binary_file),
                'status': 'failed',
                'reason': 'load_error'
            })
            continue
        
        # 构建输出WAV文件路径
        output_file = os.path.join(output_dir, binary_file.stem + '.wav')
        
        # 获取采样率
        sampling_rate = metadata.get('sampling_rate', 25600)
        
        # 转换为WAV
        success = convert_to_wav(data, output_file, sampling_rate, 
                                channel_mode, normalize_method)
        
        if success:
            results['success'] += 1
            results['files'].append({
                'file': str(binary_file),
                'output': output_file,
                'status': 'success',
                'metadata': {
                    'working_condition': metadata.get('working_condition', 'unknown'),
                    'bearing_name': metadata.get('bearing_name', 'unknown'),
                    'health_label': metadata.get('health_label', -1),
                    'sampling_rate': sampling_rate
                }
            })
        else:
            results['failed'] += 1
            results['files'].append({
                'file': str(binary_file),
                'status': 'failed',
                'reason': 'convert_error'
            })
    
    print(f"\n转换完成: 成功 {results['success']}, 失败 {results['failed']}")
    
    # 保存转换报告
    report_file = os.path.join(output_dir, 'conversion_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"转换报告已保存: {report_file}")
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("XJTU MC双通道文件转WAV音频批量转换工具")
    print("=" * 60)
    
    # 默认配置
    default_input_dir = r'D:\guzhangjiance\datasets\output_xjtu_mc\xjtu'
    default_output_dir = r'D:\guzhangjiance\datasets\output_xjtu_mc\wav_files'
    
    print(f"\n默认输入目录: {default_input_dir}")
    print(f"默认输出目录: {default_output_dir}")
    
    # 选择输入目录
    input_dir = input("\n请输入MC文件目录（回车使用默认）: ").strip().strip('"').strip("'")
    if not input_dir:
        input_dir = default_input_dir
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 {input_dir}")
        return
    
    # 选择输出目录
    output_dir = input(f"请输入输出目录（回车使用默认）: ").strip().strip('"').strip("'")
    if not output_dir:
        output_dir = default_output_dir
    
    # 选择通道模式
    print("\n通道模式选择:")
    print("1. horizontal - 只使用水平通道（推荐，用于API处理）")
    print("2. vertical - 只使用垂直通道")
    print("3. mix - 混合两个通道")
    print("4. stereo - 双通道立体声")
    
    channel_choice = input("请选择通道模式 (1/2/3/4，默认1): ").strip() or '1'
    channel_modes = {
        '1': 'horizontal',
        '2': 'vertical',
        '3': 'mix',
        '4': 'stereo'
    }
    channel_mode = channel_modes.get(channel_choice, 'horizontal')
    
    # 选择归一化方法
    print("\n归一化方法选择:")
    print("1. minmax - Min-Max归一化（推荐）")
    print("2. zscore - Z-Score归一化")
    
    norm_choice = input("请选择归一化方法 (1/2，默认1): ").strip() or '1'
    normalize_methods = {
        '1': 'minmax',
        '2': 'zscore'
    }
    normalize_method = normalize_methods.get(norm_choice, 'minmax')
    
    # 批量转换
    results = batch_convert_mc_to_wav(
        input_dir=input_dir,
        output_dir=output_dir,
        channel_mode=channel_mode,
        normalize_method=normalize_method
    )
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"成功: {results['success']} 个文件")
    print(f"失败: {results['failed']} 个文件")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
