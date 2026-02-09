#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比原始信号和分段信号的频谱，特别关注200Hz以下
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def compare_spectra():
    output_dir = 'output_xjtu_first'
    sampling_rate = 25600
    
    # 读取数据
    original = np.fromfile(os.path.join(output_dir, 'original_signal.f'), dtype=np.float32)
    first_seg = np.fromfile(os.path.join(output_dir, 'first_segment.f'), dtype=np.float32)
    
    # 计算频谱
    # 原始信号（使用完整长度）
    freqs_orig, psd_orig = signal.welch(original, fs=sampling_rate, nperseg=2048, average='mean')
    
    # 第一个分段
    freqs_seg, psd_seg = signal.welch(first_seg, fs=sampling_rate, nperseg=2048)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 全频段对比（线性坐标）
    ax1 = axes[0, 0]
    ax1.plot(freqs_orig, psd_orig, label='原始信号', alpha=0.7, linewidth=1.5)
    ax1.plot(freqs_seg, psd_seg, label='第一个分段', alpha=0.7, linewidth=1.5, linestyle='--')
    ax1.set_xlabel('频率 (Hz)')
    ax1.set_ylabel('功率谱密度')
    ax1.set_title('全频段频谱对比（线性）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(200, color='r', linestyle=':', alpha=0.5, label='200Hz')
    ax1.legend()
    
    # 2. 200Hz以下放大
    ax2 = axes[0, 1]
    mask_low = freqs_orig < 200
    ax2.plot(freqs_orig[mask_low], psd_orig[mask_low], label='原始信号', marker='o', markersize=3)
    ax2.plot(freqs_seg[freqs_seg < 200], psd_seg[freqs_seg < 200], 
             label='第一个分段', marker='s', markersize=3, linestyle='--')
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('功率谱密度')
    ax2.set_title('200Hz以下频谱对比（放大）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 计算200Hz以下的能量占比
    low_freq_orig = np.sum(psd_orig[mask_low])
    total_orig = np.sum(psd_orig)
    ratio_orig = low_freq_orig / total_orig * 100
    
    low_freq_seg = np.sum(psd_seg[freqs_seg < 200])
    total_seg = np.sum(psd_seg)
    ratio_seg = low_freq_seg / total_seg * 100
    
    ax2.text(0.5, 0.95, f'原始信号: {ratio_orig:.2f}%\n分段信号: {ratio_seg:.2f}%',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. 全频段对比（对数坐标）
    ax3 = axes[1, 0]
    ax3.semilogy(freqs_orig, psd_orig, label='原始信号', alpha=0.7, linewidth=1.5)
    ax3.semilogy(freqs_seg, psd_seg, label='第一个分段', alpha=0.7, linewidth=1.5, linestyle='--')
    ax3.set_xlabel('频率 (Hz)')
    ax3.set_ylabel('功率谱密度 (对数)')
    ax3.set_title('全频段频谱对比（对数坐标）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(200, color='r', linestyle=':', alpha=0.5)
    
    # 4. 0-1000Hz放大
    ax4 = axes[1, 1]
    mask_1k = freqs_orig < 1000
    ax4.plot(freqs_orig[mask_1k], psd_orig[mask_1k], label='原始信号', linewidth=1.5)
    ax4.plot(freqs_seg[freqs_seg < 1000], psd_seg[freqs_seg < 1000], 
             label='第一个分段', linewidth=1.5, linestyle='--')
    ax4.set_xlabel('频率 (Hz)')
    ax4.set_ylabel('功率谱密度')
    ax4.set_title('0-1000Hz频谱对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvline(200, color='r', linestyle=':', alpha=0.5, label='200Hz')
    ax4.legend()
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'frequency_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"频谱对比图已保存到: {output_file}")
    plt.close()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("频谱分析结果")
    print("="*60)
    print(f"原始信号:")
    print(f"  总能量: {total_orig:.6f}")
    print(f"  200Hz以下能量: {low_freq_orig:.6f}")
    print(f"  200Hz以下占比: {ratio_orig:.2f}%")
    print(f"\n第一个分段:")
    print(f"  总能量: {total_seg:.6f}")
    print(f"  200Hz以下能量: {low_freq_seg:.6f}")
    print(f"  200Hz以下占比: {ratio_seg:.2f}%")
    print(f"\n差异: {abs(ratio_orig - ratio_seg):.2f}%")
    
    if abs(ratio_orig - ratio_seg) < 0.5:
        print("\n结论: 分段信号和原始信号的200Hz以下能量占比基本相同")
        print("      分段过程没有丢失低频信息")
    else:
        print("\n注意: 分段信号和原始信号的200Hz以下能量占比有差异")
        print("      这可能是由于频率分辨率或窗口效应造成的")

if __name__ == '__main__':
    compare_spectra()

