#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查输出文件的频率内容，查看是否缺少低频信息
"""
import os
import sys
import numpy as np
from scipy import signal

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def analyze_file_frequency(filepath, sampling_rate=25600, title=""):
    """分析文件的频率内容"""
    print(f"\n{'='*60}")
    print(f"分析: {title}")
    print(f"文件: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    # 读取二进制数据
    data = np.fromfile(filepath, dtype=np.float32)
    print(f"数据点数: {len(data)}")
    print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"数据均值: {data.mean():.4f}")
    print(f"数据标准差: {data.std():.4f}")
    
    # 计算FFT
    # 对于分段数据，需要判断是单个分段还是多个分段
    if len(data) > 0:
        # 判断是否是多个分段（2048的倍数）
        segment_length = 2048
        if len(data) % segment_length == 0 and len(data) > segment_length:
            # 多个分段，取第一个分段分析
            print(f"检测到多个分段，分析第一个分段（前{segment_length}点）")
            segment_data = data[:segment_length]
        else:
            segment_data = data
        
        # 使用welch方法计算功率谱密度
        nperseg = min(segment_length, len(segment_data))
        freqs, psd = signal.welch(segment_data, fs=sampling_rate, nperseg=nperseg)
        
        # 查找200Hz以下的能量
        low_freq_mask = freqs < 200
        low_freq_energy = np.sum(psd[low_freq_mask])
        total_energy = np.sum(psd)
        low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        
        print(f"\n频率分析 (采样率: {sampling_rate} Hz):")
        print(f"  频率范围: {freqs.min():.2f} - {freqs.max():.2f} Hz")
        print(f"  200Hz以下能量占比: {low_freq_ratio*100:.2f}%")
        print(f"  200Hz以下能量: {low_freq_energy:.6f}")
        print(f"  总能量: {total_energy:.6f}")
        
        # 显示主要频率成分
        peak_indices = signal.find_peaks(psd, height=np.max(psd)*0.1)[0]
        if len(peak_indices) > 0:
            top_5 = peak_indices[np.argsort(psd[peak_indices])[-5:]][::-1]
            print(f"\n主要频率成分 (前5个):")
            for idx in top_5:
                print(f"  {freqs[idx]:.2f} Hz: 功率 = {psd[idx]:.6f}")
        
        # 检查200Hz以下是否有显著成分
        if low_freq_ratio < 0.01:
            print(f"\n[WARNING] 200Hz以下能量占比极低 ({low_freq_ratio*100:.2f}%)")
        else:
            print(f"\n[OK] 200Hz以下有能量 ({low_freq_ratio*100:.2f}%)")
        
        # 如果是多个分段，也分析整体
        if len(data) > 0 and len(data) % 2048 == 0 and len(data) > 2048:
            print(f"\n--- 整体分析（所有分段平均）---")
            num_segments = len(data) // 2048
            all_psds = []
            for i in range(num_segments):
                seg = data[i*2048:(i+1)*2048]
                _, psd_seg = signal.welch(seg, fs=sampling_rate, nperseg=2048)
                all_psds.append(psd_seg)
            avg_psd = np.mean(all_psds, axis=0)
            
            low_freq_mask_avg = freqs < 200
            low_freq_energy_avg = np.sum(avg_psd[low_freq_mask_avg])
            total_energy_avg = np.sum(avg_psd)
            low_freq_ratio_avg = low_freq_energy_avg / total_energy_avg if total_energy_avg > 0 else 0
            print(f"平均200Hz以下能量占比: {low_freq_ratio_avg*100:.2f}%")

def main():
    output_dir = 'output_xjtu_first'
    sampling_rate = 25600  # XJTU采样率
    
    files_to_check = [
        ('original_signal.f', '原始完整信号'),
        ('first_segment.f', '第一个分段'),
        ('sample_segments_10.f', '前10个分段（展平）'),
    ]
    
    print("="*60)
    print("检查XJTU输出文件的频率内容")
    print("="*60)
    
    for filename, title in files_to_check:
        filepath = os.path.join(output_dir, filename)
        analyze_file_frequency(filepath, sampling_rate, title)
    
    print("\n" + "="*60)
    print("分析完成")
    print("="*60)
    print("\n说明:")
    print("1. 原始信号 (original_signal.f): 完整的32768点信号")
    print("2. 第一个分段 (first_segment.f): 2048点分段（前2048个点）")
    print("3. 前10个分段 (sample_segments_10.f): 10个2048点分段展平后的数据")
    print("\n如果分段数据缺少200Hz以下信息，可能的原因：")
    print("- 分段长度(2048点)对低频分辨率有限")
    print("- 分段可能从信号的某个位置开始，恰好跳过了低频成分")
    print("- FFT分析时的频率分辨率限制")

if __name__ == '__main__':
    main()

