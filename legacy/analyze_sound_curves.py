"""
声音能量曲线数据分析
理解李群算法转换的物理意义
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_sound_curve(xlsx_path, mat_name):
    """分析单个声音曲线文件"""
    try:
        df = pd.read_excel(xlsx_path, header=None, skiprows=2)
        
        freq = df.iloc[:, 0].values
        volume = df.iloc[:, 1].values  
        density = df.iloc[:, 2].values
        
        return {
            'file': Path(xlsx_path).name,
            'mat_name': mat_name,
            'n_points': len(freq),
            'freq_range': (freq.min(), freq.max()),
            'volume_range': (volume.min(), volume.max()),
            'volume_mean': np.mean(volume),
            'volume_std': np.std(volume),
            'volume_energy': np.sum(volume**2),
            'density_range': (density.min(), density.max()),
            'density_mean': np.mean(density),
            'density_std': np.std(density),
            'density_energy': np.sum(density**2),
            'freq': freq,
            'volume': volume,
            'density': density
        }
    except Exception as e:
        print(f"Error processing {xlsx_path}: {e}")
        return None

def main():
    print("\n" + "="*80)
    print("声音能量曲线数据分析 - 李群算法转换的物理意义")
    print("="*80)
    
    sound_dir = Path('声音能量曲线数据')
    results = []
    
    # 分析所有11个文件
    for xlsx_file in sorted(sound_dir.glob('*.xlsx')):
        # 获取原始文件名
        df_header = pd.read_excel(xlsx_file, header=None, nrows=1)
        wav_name = df_header.iloc[0, 0]
        mat_name = wav_name.replace('.wav', '')
        
        result = analyze_sound_curve(xlsx_file, mat_name)
        if result:
            results.append(result)
    
    # 1. 汇总统计信息
    print("\n【1】数据基本统计")
    print("-" * 80)
    print(f"{'样本':<20} {'频率范围(Hz)':<20} {'音量范围':<20} {'密度范围':<20}")
    print("-" * 80)
    
    for res in results:
        freq_range = f"{res['freq_range'][0]:.1f}-{res['freq_range'][1]:.1f}"
        vol_range = f"{res['volume_range'][0]:.2f}-{res['volume_range'][1]:.2f}"
        dens_range = f"{res['density_range'][0]:.2f}-{res['density_range'][1]:.2f}"
        print(f"{res['mat_name']:<20} {freq_range:<20} {vol_range:<20} {dens_range:<20}")
    
    # 2. 能量分析（理解李群变换的意义）
    print("\n【2】能量特征分析（李群变换反映的物理量）")
    print("-" * 80)
    print("李群变换的本质：")
    print("  - 音量：振动信号经傅里叶变换后的幅度谱")
    print("  - 密度：信号能量在频域的分布浓度")
    print("  - 李群表示：SE(3)群作用在特征空间上的几何变换")
    print("-" * 80)
    
    vol_energies = [r['volume_energy'] for r in results]
    dens_energies = [r['density_energy'] for r in results]
    
    print(f"\n音量能量统计:")
    print(f"  平均值: {np.mean(vol_energies):.2e}")
    print(f"  最大值: {np.max(vol_energies):.2e} ({results[np.argmax(vol_energies)]['mat_name']})")
    print(f"  最小值: {np.min(vol_energies):.2e} ({results[np.argmin(vol_energies)]['mat_name']})")
    print(f"  标准差: {np.std(vol_energies):.2e}")
    
    print(f"\n密度能量统计:")
    print(f"  平均值: {np.mean(dens_energies):.2e}")
    print(f"  最大值: {np.max(dens_energies):.2e} ({results[np.argmax(dens_energies)]['mat_name']})")
    print(f"  最小值: {np.min(dens_energies):.2e} ({results[np.argmin(dens_energies)]['mat_name']})")
    print(f"  标准差: {np.std(dens_energies):.2e}")
    
    # 3. 频域特征
    print("\n【3】频域特征分析")
    print("-" * 80)
    
    for res in results:
        freq = res['freq']
        volume = res['volume']
        density = res['density']
        
        # 找最强频率
        peak_freq_idx = np.argmax(volume)
        peak_freq = freq[peak_freq_idx]
        peak_vol = volume[peak_freq_idx]
        
        # 找密度最高的频率
        peak_dens_idx = np.argmax(density)
        peak_dens_freq = freq[peak_dens_idx]
        peak_dens = density[peak_dens_idx]
        
        # 计算频带内的能量分布
        low_freq_energy = np.sum(volume[(freq >= 0) & (freq < 1000)]**2)      # 0-1kHz
        mid_freq_energy = np.sum(volume[(freq >= 1000) & (freq < 10000)]**2)  # 1-10kHz
        high_freq_energy = np.sum(volume[(freq >= 10000)]**2)                 # >10kHz
        
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
        
        print(f"\n{res['mat_name']}:")
        print(f"  最强频率: {peak_freq:.2f} Hz (幅度: {peak_vol:.2f})")
        print(f"  密度峰值频率: {peak_dens_freq:.2f} Hz (密度: {peak_dens:.2f})")
        print(f"  频带能量分布:")
        print(f"    - 低频(0-1kHz):   {100*low_freq_energy/total_energy:6.2f}%")
        print(f"    - 中频(1-10kHz):  {100*mid_freq_energy/total_energy:6.2f}%")
        print(f"    - 高频(>10kHz):   {100*high_freq_energy/total_energy:6.2f}%")
    
    # 4. 样本分类特性
    print("\n【4】样本类型与物理意义")
    print("-" * 80)
    
    normal_sample = [r for r in results if 'Normal' in r['mat_name']]
    fault_samples = [r for r in results if 'Normal' not in r['mat_name']]
    
    print(f"\n正常样本 (Normal):")
    if normal_sample:
        s = normal_sample[0]
        print(f"  {s['mat_name']}: 音量均值={s['volume_mean']:.4f}, 密度均值={s['density_mean']:.4f}")
        print(f"  物理意义：无故障时的基准频谱响应")
    
    print(f"\n故障样本 ({len(fault_samples)}个):")
    for s in fault_samples:
        print(f"  {s['mat_name']}: 音量均值={s['volume_mean']:.4f}, 密度均值={s['density_mean']:.4f}")
    
    # 5. 李群变换的数学含义
    print("\n【5】李群(SE(3))变换的数学含义")
    print("-" * 80)
    print("李群是具有群结构的光滑流形，SE(3)特殊欧几里得群表示：")
    print("  - 旋转矩阵 R 属于 SO(3)：特征旋转变换")
    print("  - 平移向量 t 属于 R^3：特征平移变换")
    print("")
    print("在声音能量曲线中的应用：")
    print("  1. 频谱旋转 (Spectral Rotation):")
    print("     - 通过李群作用重新排列频率分量")
    print("     - 保持能量的不变性")
    print("")
    print("  2. 能量平移 (Energy Translation):")
    print("     - 沿频轴的移动：f -> f + Δf")
    print("     - 特别适合处理转速变化引起的频率漂移")
    print("")
    print("  3. 几何不变性 (Geometric Invariance):")
    print("     - 特征对旋转/平移保持不变")
    print("     - 提高了跨工况/跨转速的泛化能力")
    print("")
    print("在本项目中的意义：")
    print("  - 音量曲线: 基本频谱信息，受转速影响")
    print("  - 密度曲线: 能量集中度，对故障类型敏感（诊断关键特征）")
    
    # 6. 可视化建议
    print("\n【6】建议的进一步分析")
    print("-" * 80)
    print("1. 绘制频谱对比图：")
    print("   - 正常vs故障样本的频谱特性")
    print("   - 不同故障类型的频域差异")
    print("")
    print("2. 能量聚集分析：")
    print("   - 低频泄漏 (低频噪声)")
    print("   - 中频峰值 (故障特征频率及谐波)")
    print("   - 高频衰减 (衰减特性)")
    print("")
    print("3. 李群不变特征提取：")
    print("   - 基于李群变换的特征对齐")
    print("   - 跨转速的特征归一化")
    print("   - 故障诊断的鲁棒性测试")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
