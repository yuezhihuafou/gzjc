"""
李群特征提取结果分析 - 可视化 & 统计诊断
对标准的PHM故障诊断框架进行严谨的数据分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据映射：样本名称 -> (类型, 标签)
SAMPLE_MAP = {
    '97_Normal_0': ('Normal', 0, '正常状态'),
    '108_3': ('Inner Race', 1, '内圈故障'),
    '187_2': ('Inner Race', 1, '内圈故障'),
    '200@6_3': ('Inner Race', 1, '内圈故障-6点位'),
    '234_0': ('Ball', 2, '滚动体故障'),
    '247_1': ('Ball', 2, '滚动体故障'),
    '301_3': ('Outer Race', 3, '外圈故障'),
    '156_0': ('Outer Race', 3, '外圈故障'),
    '169_0': ('Outer Race', 3, '外圈故障'),
    '202@6_1': ('Outer Race', 3, '外圈故障-6点位'),
    '190_1': ('Outer Race', 3, '外圈故障'),
}

def load_sound_data(sample_name):
    """加载单个样本的声音能量曲线数据"""
    sound_dir = Path('声音能量曲线数据')
    
    for xlsx_file in sound_dir.glob('*.xlsx'):
        df_header = pd.read_excel(xlsx_file, header=None, nrows=1)
        wav_name = df_header.iloc[0, 0]
        
        if sample_name in wav_name:
            df = pd.read_excel(xlsx_file, header=None, skiprows=2)
            freq = df.iloc[:, 0].values
            volume = df.iloc[:, 1].values  # 能量曲线
            density = df.iloc[:, 2].values  # 密度曲线
            
            return {
                'freq': freq,
                'energy': volume,
                'density': density
            }
    
    return None

def main():
    print("\n" + "="*100)
    print("李群特征提取结果分析 - Lie Group Decomposition Analysis")
    print("="*100)
    
    # ============================================================================
    # 第1部分：数据加载与基本统计
    # ============================================================================
    print("\n【第1部分】数据加载与基本统计")
    print("-" * 100)
    
    all_data = {}
    for sample_name in SAMPLE_MAP.keys():
        data = load_sound_data(sample_name)
        if data:
            all_data[sample_name] = data
            fault_type, label, description = SAMPLE_MAP[sample_name]
            print(f"[OK] {sample_name:12s} ({description:15s}) - E: {data['energy'].min():.2f}~{data['energy'].max():.2f}, "
                  f"D: {data['density'].min():.2f}~{data['density'].max():.2f}")
    
    if len(all_data) < len(SAMPLE_MAP):
        print(f"[WARNING] Only loaded {len(all_data)}/{len(SAMPLE_MAP)} samples")
    
    # ============================================================================
    # 第2部分：对比分析 - 正常 vs 内圈故障
    # ============================================================================
    print("\n【第2部分】对比分析：正常 vs 内圈故障")
    print("-" * 100)
    
    # 获取对比样本
    normal_data = all_data.get('97_Normal_0')
    inner_race_samples = ['108_3', '187_2', '200@6_3']
    inner_race_data = [all_data[s] for s in inner_race_samples if s in all_data]
    
    if normal_data and inner_race_data:
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('李群分解结果对比：正常状态 vs 内圈故障', fontsize=16, fontweight='bold')
        
        freq = normal_data['freq']
        
        # 图1：能量曲线对比
        ax = axes[0, 0]
        ax.plot(freq, normal_data['energy'], 'g-', linewidth=2.5, label='正常状态', alpha=0.8)
        for i, sample_name in enumerate(inner_race_samples):
            if sample_name in all_data:
                ax.plot(freq, all_data[sample_name]['energy'], '--', linewidth=1.5, 
                       label=f'内圈故障-{sample_name}', alpha=0.7)
        ax.set_xlabel('频率 (Hz)', fontsize=12)
        ax.set_ylabel('能量幅度 (Energy)', fontsize=12)
        ax.set_title('能量曲线对比 (Volume Spectrum)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 图2：密度曲线对比
        ax = axes[0, 1]
        ax.plot(freq, normal_data['density'], 'g-', linewidth=2.5, label='正常状态', alpha=0.8)
        for i, sample_name in enumerate(inner_race_samples):
            if sample_name in all_data:
                ax.plot(freq, all_data[sample_name]['density'], '--', linewidth=1.5,
                       label=f'内圈故障-{sample_name}', alpha=0.7)
        ax.set_xlabel('频率 (Hz)', fontsize=12)
        ax.set_ylabel('密度 (Density)', fontsize=12)
        ax.set_title('密度曲线对比 (Density Distribution)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 图3：能量曲线 - 低频段放大（0-2kHz）
        ax = axes[1, 0]
        mask = freq <= 2000
        ax.plot(freq[mask], normal_data['energy'][mask], 'g-', linewidth=2.5, label='正常状态', marker='o', markersize=3)
        for sample_name in inner_race_samples:
            if sample_name in all_data:
                ax.plot(freq[mask], all_data[sample_name]['energy'][mask], '--', linewidth=1.5,
                       label=f'{sample_name}', marker='s', markersize=2, alpha=0.7)
        ax.set_xlabel('频率 (Hz)', fontsize=12)
        ax.set_ylabel('能量幅度', fontsize=12)
        ax.set_title('低频段放大 (0-2kHz)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 图4：密度曲线 - 中频段放大（1-10kHz）
        ax = axes[1, 1]
        mask = (freq >= 1000) & (freq <= 10000)
        ax.plot(freq[mask], normal_data['density'][mask], 'g-', linewidth=2.5, label='正常状态', marker='o', markersize=3)
        for sample_name in inner_race_samples:
            if sample_name in all_data:
                ax.plot(freq[mask], all_data[sample_name]['density'][mask], '--', linewidth=1.5,
                       label=f'{sample_name}', marker='s', markersize=2, alpha=0.7)
        ax.set_xlabel('频率 (Hz)', fontsize=12)
        ax.set_ylabel('密度', fontsize=12)
        ax.set_title('中频段放大 (1-10kHz) - 故障特征区域', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sound_curves_comparison_normal_vs_inner_race.png', dpi=150, bbox_inches='tight')
        print("[SAVE] Comparison chart: sound_curves_comparison_normal_vs_inner_race.png")
        plt.close()
    
    # ============================================================================
    # 第3部分：统计分布分析
    # ============================================================================
    print("\n【第3部分】统计分布分析 - 检查归一化需求")
    print("-" * 100)
    
    # 收集所有能量和密度值
    all_energies = []
    all_densities = []
    
    for sample_name, data in all_data.items():
        all_energies.extend(data['energy'])
        all_densities.extend(data['density'])
    
    all_energies = np.array(all_energies)
    all_densities = np.array(all_densities)
    
    # 计算统计信息
    energy_stats = {
        'mean': np.mean(all_energies),
        'std': np.std(all_energies),
        'min': np.min(all_energies),
        'max': np.max(all_energies),
        'median': np.median(all_energies),
        'range': np.max(all_energies) - np.min(all_energies),
    }
    
    density_stats = {
        'mean': np.mean(all_densities),
        'std': np.std(all_densities),
        'min': np.min(all_densities),
        'max': np.max(all_densities),
        'median': np.median(all_densities),
        'range': np.max(all_densities) - np.min(all_densities),
    }
    
    print("\n能量曲线 (Energy) 统计：")
    print(f"  均值: {energy_stats['mean']:.4f}")
    print(f"  标差: {energy_stats['std']:.4f}")
    print(f"  最小值: {energy_stats['min']:.4f}")
    print(f"  最大值: {energy_stats['max']:.4f}")
    print(f"  中位数: {energy_stats['median']:.4f}")
    print(f"  动态范围: {energy_stats['range']:.4f}")
    print(f"  变异系数 (CV): {energy_stats['std']/energy_stats['mean']:.4f}")
    
    print("\n密度曲线 (Density) 统计：")
    print(f"  均值: {density_stats['mean']:.4f}")
    print(f"  标差: {density_stats['std']:.4f}")
    print(f"  最小值: {density_stats['min']:.4f}")
    print(f"  最大值: {density_stats['max']:.4f}")
    print(f"  中位数: {density_stats['median']:.4f}")
    print(f"  动态范围: {density_stats['range']:.4f}")
    print(f"  变异系数 (CV): {density_stats['std']/density_stats['mean']:.4f}")
    
    # 创建直方图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('数据分布与归一化需求分析', fontsize=14, fontweight='bold')
    
    # 能量直方图
    ax = axes[0]
    ax.hist(all_energies, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(energy_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"均值={energy_stats['mean']:.2f}")
    ax.axvline(energy_stats['median'], color='green', linestyle='--', linewidth=2, label=f"中位数={energy_stats['median']:.2f}")
    ax.set_xlabel('能量幅度', fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.set_title(f'能量分布直方图\n(CV={energy_stats["std"]/energy_stats["mean"]:.3f}, 需要归一化)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 密度直方图
    ax = axes[1]
    ax.hist(all_densities, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.axvline(density_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"均值={density_stats['mean']:.2f}")
    ax.axvline(density_stats['median'], color='green', linestyle='--', linewidth=2, label=f"中位数={density_stats['median']:.2f}")
    ax.set_xlabel('密度', fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.set_title(f'密度分布直方图\n(CV={density_stats["std"]/density_stats["mean"]:.3f}, 需要归一化)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('energy_density_distribution_histograms.png', dpi=150, bbox_inches='tight')
    print("[SAVE] Histogram chart: energy_density_distribution_histograms.png")
    plt.close()
    
    # ============================================================================
    # 第4部分：相关性分析
    # ============================================================================
    print("\n【第4部分】相关性分析 - 信息冗余度评估")
    print("-" * 100)
    
    # 对每个样本计算皮尔逊相关系数
    correlations = {}
    
    for sample_name, data in all_data.items():
        corr, p_value = pearsonr(data['energy'], data['density'])
        fault_type, label, description = SAMPLE_MAP[sample_name]
        correlations[sample_name] = {'corr': corr, 'p_value': p_value}
        
        print(f"{sample_name:12s} ({description:15s}): r = {corr:7.4f}, p-value = {p_value:.2e}")
    
    # 总体相关系数
    overall_corr, overall_p = pearsonr(all_energies, all_densities)
    print(f"\n{'整体相关系数':12s}: r = {overall_corr:7.4f}, p-value = {overall_p:.2e}")
    
    # 按故障类型分组计算相关系数
    print("\n按故障类型分组的相关系数：")
    
    fault_groups = {
        '正常': {'Normal': []},
        '内圈': {'Inner Race': []},
        '外圈': {'Outer Race': []},
        '滚动体': {'Ball': []},
    }
    
    for sample_name, (fault_type, label, desc) in SAMPLE_MAP.items():
        if sample_name in all_data:
            data = all_data[sample_name]
            corr = correlations[sample_name]['corr']
            
            if fault_type == 'Normal':
                fault_groups['正常']['Normal'].append(corr)
            elif fault_type == 'Inner Race':
                fault_groups['内圈']['Inner Race'].append(corr)
            elif fault_type == 'Outer Race':
                fault_groups['外圈']['Outer Race'].append(corr)
            elif fault_type == 'Ball':
                fault_groups['滚动体']['Ball'].append(corr)
    
    for group_name, group_data in fault_groups.items():
        for fault_type, corrs in group_data.items():
            if corrs:
                mean_corr = np.mean(corrs)
                std_corr = np.std(corrs) if len(corrs) > 1 else 0
                print(f"  {group_name:8s}: r_mean = {mean_corr:.4f} (±{std_corr:.4f})")
    
    # 创建相关性分析图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('能量与密度的相关性分析', fontsize=14, fontweight='bold')
    
    # 图1：相关性条形图
    ax = axes[0]
    sample_names = list(correlations.keys())
    corr_values = [correlations[s]['corr'] for s in sample_names]
    colors = ['green' if SAMPLE_MAP[s][0] == 'Normal' else 'red' for s in sample_names]
    
    ax.barh(sample_names, corr_values, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(overall_corr, color='blue', linestyle='--', linewidth=2, label=f'整体相关系数={overall_corr:.4f}')
    ax.set_xlabel('皮尔逊相关系数 (r)', fontsize=11)
    ax.set_title('样本间能量-密度相关性', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 图2：散点图 - 能量vs密度
    ax = axes[1]
    
    # 绘制所有点
    scatter = ax.scatter(all_energies, all_densities, alpha=0.3, s=10, c='gray', label='所有点')
    
    # 按样本类型着色
    colors_map = {'Normal': 'green', 'Inner Race': 'red', 'Outer Race': 'blue', 'Ball': 'orange'}
    for sample_name, data in all_data.items():
        fault_type = SAMPLE_MAP[sample_name][0]
        color = colors_map.get(fault_type, 'gray')
        ax.scatter(data['energy'], data['density'], alpha=0.6, s=30, c=color, 
                  label=fault_type if fault_type not in [SAMPLE_MAP[sn][0] for sn in list(all_data.keys())[:sample_names.index(sample_name)]] else '')
    
    ax.set_xlabel('能量幅度', fontsize=11)
    ax.set_ylabel('密度', fontsize=11)
    ax.set_title(f'能量-密度散点图 (r={overall_corr:.4f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加拟合线
    z = np.polyfit(all_energies, all_densities, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_energies.min(), all_energies.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label=f'拟合线: y={z[0]:.4f}x+{z[1]:.4f}')
    
    ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    plt.savefig('correlation_analysis_energy_density.png', dpi=150, bbox_inches='tight')
    print("[SAVE] Correlation chart: correlation_analysis_energy_density.png")
    plt.close()
    
    # ============================================================================
    # 第5部分：结论与建议
    # ============================================================================
    print("\n[SECTION 5] Analysis Conclusions & Recommendations")
    print("-" * 100)
    
    print("\n1. Normalization Requirements:")
    if energy_stats['std'] / energy_stats['mean'] > 0.5:
        print(f"   [OK] Energy CV = {energy_stats['std']/energy_stats['mean']:.3f} > 0.5")
        print("        Recommend: Z-score or Min-Max normalization")
    
    if density_stats['std'] / density_stats['mean'] > 0.5:
        print(f"   [OK] Density CV = {density_stats['std']/density_stats['mean']:.3f} > 0.5")
        print("        Recommend: Z-score or Min-Max normalization")
    
    print("\n2. Information Redundancy:")
    if abs(overall_corr) < 0.3:
        print(f"   [OK] Pearson r = {overall_corr:.4f} < 0.3 (Weak correlation)")
        print("        Conclusion: Energy & Density contain different information, low redundancy")
    elif 0.3 <= abs(overall_corr) < 0.7:
        print(f"   [OK] Pearson r = {overall_corr:.4f} in [0.3, 0.7) (Moderate correlation)")
        print("        Conclusion: Curves have partial correlation but complementary information")
    else:
        print(f"   [OK] Pearson r = {overall_corr:.4f} > 0.7 (Strong correlation)")
        print("        Conclusion: High information redundancy, consider single-channel approach")
    
    print("\n3. Model Design Recommendations:")
    print("   [OK] Input format: Dual-channel (Energy, Density) - Frequency dimension x 2")
    print("   [OK] Preprocessing: Z-score normalization before deep learning")
    print("   [OK] Architecture: Multi-channel CNN or Transformer for spectral processing")
    print("   [OK] Fusion strategy: Feature fusion at intermediate layers, not direct concatenation")
    
    print("\n" + "="*100 + "\n")

if __name__ == '__main__':
    main()
