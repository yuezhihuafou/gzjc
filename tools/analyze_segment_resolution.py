#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析不同分段长度对频率分辨率的影响
"""
import os
import sys
import numpy as np
from scipy import signal

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def analyze_segment_resolutions():
    """分析不同分段长度的频率分辨率和时间特性"""
    
    # XJTU采样率
    sampling_rate = 25600  # Hz
    
    # 常见的分段长度选项
    segment_lengths = [1024, 2048, 4096, 8192, 16384]
    
    print("="*70)
    print("分段长度 vs 频率分辨率分析")
    print("="*70)
    print(f"采样率: {sampling_rate} Hz")
    print()
    
    print(f"{'分段长度':<12} {'频率分辨率':<15} {'时间长度':<15} {'200Hz以下点数':<15} {'可分辨最小频率':<15}")
    print("-"*70)
    
    for seg_len in segment_lengths:
        # 频率分辨率 = 采样率 / 分段长度
        freq_resolution = sampling_rate / seg_len
        
        # 时间长度 = 分段长度 / 采样率
        time_length = seg_len / sampling_rate
        
        # 200Hz以下的频率点数
        points_below_200hz = int(200 / freq_resolution)
        
        # 可分辨的最小频率（理论上为频率分辨率）
        min_freq = freq_resolution
        
        print(f"{seg_len:<12} {freq_resolution:<15.2f} {time_length:<15.4f} {points_below_200hz:<15} {min_freq:<15.2f}")
    
    print()
    print("="*70)
    print("分析说明")
    print("="*70)
    print()
    print("1. 频率分辨率：")
    print("   - 频率分辨率 = 采样率 / 分段长度")
    print("   - 决定了能区分的最小频率间隔")
    print("   - 分辨率越小（越细），频率分析越精确")
    print()
    
    print("2. 时间长度：")
    print("   - 每个分段的时间长度")
    print("   - 越长的分段包含更多信息，但计算量也更大")
    print()
    
    print("3. 200Hz以下点数：")
    print("   - 在200Hz以下有多少个频率采样点")
    print("   - 点数越多，低频分析越详细")
    print()
    
    print("4. 推荐选择：")
    print("   - 1024点：频率分辨率25Hz，适合快速处理，但分辨率较粗")
    print("   - 2048点：频率分辨率12.5Hz，平衡选择（当前使用）")
    print("   - 4096点：频率分辨率6.25Hz，更精细，计算量增加")
    print("   - 8192点：频率分辨率3.125Hz，非常精细，但内存和计算量大")
    print()
    
    # 实际数据对比
    print("="*70)
    print("实际数据对比（使用XJTU第一个文件）")
    print("="*70)
    
    output_dir = 'output_xjtu_first'
    original_file = os.path.join(output_dir, 'original_signal.f')
    
    if os.path.exists(original_file):
        original = np.fromfile(original_file, dtype=np.float32)
        print(f"原始信号长度: {len(original)} 点 ({len(original)/sampling_rate:.3f} 秒)")
        print()
        
        print(f"{'分段长度':<12} {'分段数量':<12} {'频率分辨率':<15} {'计算时间增加':<15}")
        print("-"*70)
        
        for seg_len in segment_lengths:
            # 计算能生成多少个分段（50%重叠）
            step = int(seg_len * 0.5)  # 50%重叠
            num_segments = (len(original) - seg_len) // step + 1 if len(original) >= seg_len else 0
            freq_res = sampling_rate / seg_len
            
            # 相对计算量（以2048为基准）
            compute_factor = (seg_len / 2048) ** 2 if seg_len >= 2048 else (2048 / seg_len) ** 2
            
            print(f"{seg_len:<12} {num_segments:<12} {freq_res:<15.2f} {compute_factor:<15.2f}x")
        
        print()
        print("注意：计算时间增加是FFT计算复杂度的近似估计")
        print("     实际训练时间还会受模型结构、批大小等影响")
    
    print()
    print("="*70)
    print("建议")
    print("="*70)
    print("当前使用2048点分段：")
    print("  [OK] 频率分辨率12.5Hz，对于轴承故障频率（通常>50Hz）足够精确")
    print("  [OK] 计算效率高，适合大批量数据处理")
    print("  [OK] 是深度学习中的常见选择")
    print()
    print("如果需要更精细的频率分析，可以考虑：")
    print("  - 4096点：分辨率提升到6.25Hz，计算量增加约4倍")
    print("  - 8192点：分辨率提升到3.125Hz，计算量增加约16倍")
    print()
    print("权衡建议：")
    print("  - 对于故障诊断：2048点通常足够（故障频率特征足够明显）")
    print("  - 对于精细频率分析：4096点是不错的折中")
    print("  - 对于研究低频特性：8192点可以更详细")

if __name__ == '__main__':
    analyze_segment_resolutions()

