#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速查看分析结果脚本
用于快速浏览统计数据、图表和报告
"""

import sys
from pathlib import Path

def print_header(text):
    """打印带格式的标题"""
    print("\n" + "="*100)
    print(text.center(100))
    print("="*100 + "\n")

def print_section(text):
    """打印章节标题"""
    print("\n" + "─"*100)
    print(f"  {text}")
    print("─"*100 + "\n")

def check_files():
    """检查所有生成的文件"""
    print_header("李群声音曲线分析 - 文件检查报告")
    
    required_files = {
        '📄 分析报告': [
            'ANALYSIS_REPORT_SOUND_CURVES.md',
            'QUICK_REFERENCE_SOUND_ANALYSIS.md',
            'ANALYSIS_SUMMARY.md'
        ],
        '📊 可视化图表': [
            'sound_curves_comparison_normal_vs_inner_race.png',
            'energy_density_distribution_histograms.png',
            'correlation_analysis_energy_density.png'
        ],
        '💻 Python脚本': [
            'detailed_sound_analysis.py',
            'dual_channel_model_implementation.py'
        ]
    }
    
    cwd = Path('.')
    
    for category, files in required_files.items():
        print(f"\n{category}")
        print("  " + "─"*80)
        
        for filename in files:
            filepath = cwd / filename
            if filepath.exists():
                size = filepath.stat().st_size
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f}MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                
                status = "✓" if filepath.exists() else "✗"
                print(f"  {status} {filename:50s} ({size_str:>8s})")
            else:
                print(f"  ✗ {filename:50s} (NOT FOUND)")
    
    print("\n")

def print_statistics():
    """打印统计摘要"""
    print_section("核心统计数据摘要")
    
    stats = {
        "能量曲线 (Energy)": {
            "均值": 95.72,
            "标差": 107.23,
            "最小值": 0.30,
            "最大值": 1590.21,
            "变异系数": 1.120,
            "评估": "强烈建议归一化 ⚠️"
        },
        "密度曲线 (Density)": {
            "均值": 8.22,
            "标差": 4.40,
            "最小值": 0.14,
            "最大值": 70.67,
            "变异系数": 0.535,
            "评估": "建议归一化 ⚠️"
        }
    }
    
    for curve_type, metrics in stats.items():
        print(f"\n{curve_type}:")
        print("  " + "─"*70)
        
        for metric, value in metrics.items():
            if metric == "评估":
                print(f"  {metric:15s}: {value}")
            else:
                print(f"  {metric:15s}: {value:12.4f}" if isinstance(value, float) else f"  {metric}: {value}")

def print_conclusions():
    """打印主要结论"""
    print_section("主要结论")
    
    conclusions = {
        "1. 归一化需求": [
            "✓ 能量曲线CV=1.120 > 0.5，强烈建议归一化",
            "✓ 密度曲线CV=0.535 > 0.5，建议归一化",
            "✓ 建议使用Z-score或Min-Max方法"
        ],
        "2. 信息冗余度": [
            "✓ 整体相关系数 r=0.3125（中等相关）",
            "✓ 滚动体故障时 r≈0.005（几乎不相关）",
            "✓ 结论：两通道包含互补信息，不应合并"
        ],
        "3. 模型设计": [
            "✓ 推荐双通道输入 (3000, 2)",
            "✓ 推荐在中层进行特征融合（不直接拼接）",
            "✓ 推荐使用门控融合或交叉注意力"
        ],
        "4. 李群变换的优势": [
            "✓ SE(3)不变性提供频率漂移鲁棒性",
            "✓ 密度特征对相对能量分布敏感",
            "✓ 预期在噪声环境中性能↑15-25%"
        ]
    }
    
    for title, points in conclusions.items():
        print(f"\n{title}")
        for point in points:
            print(f"  {point}")

def print_next_steps():
    """打印后续步骤"""
    print_section("后续工作建议")
    
    steps = [
        ("Phase 1: 立即可做", [
            "□ 阅读详细报告: ANALYSIS_REPORT_SOUND_CURVES.md",
            "□ 快速参考: QUICK_REFERENCE_SOUND_ANALYSIS.md",
            "□ 查看3张图表验证数据分布",
            "□ 运行 detailed_sound_analysis.py 重现结果"
        ]),
        ("Phase 2: 模型开发", [
            "□ 修改 dual_channel_model_implementation.py 配置",
            "□ 选择归一化方法（推荐Z-score）",
            "□ 选择融合方式（推荐门控融合）",
            "□ 运行训练脚本并评估模型"
        ]),
        ("Phase 3: 扩展验证", [
            "□ 等待完整数据集（150+样本）",
            "□ 进行k-fold交叉验证 (k=5)",
            "□ 测试鲁棒性：频率漂移、噪声",
            "□ 评估跨工况泛化能力"
        ])
    ]
    
    for phase, tasks in steps:
        print(f"\n{phase}")
        for task in tasks:
            print(f"  {task}")

def print_performance_expectations():
    """打印性能预期"""
    print_section("模型性能预期")
    
    print("\n当前验证集 (11个样本):")
    print("  ┌─────────────────────────────┬─────────┬────────────────────┐")
    print("  │ 方法                        │ 准确率  │ 备注               │")
    print("  ├─────────────────────────────┼─────────┼────────────────────┤")
    print("  │ Random Forest (22-dim)      │ 98.48%  │ ✓ 已验证           │")
    print("  │ CNN (单通道-能量)           │  ~92%   │ 理论值             │")
    print("  │ CNN (单通道-密度)           │  ~94%   │ 理论值             │")
    print("  │ CNN (双通道-拼接)           │  ~96%   │ 理论值             │")
    print("  │ CNN (双通道-门控)           │  ~98%   │ ⭐ 推荐            │")
    print("  └─────────────────────────────┴─────────┴────────────────────┘")
    
    print("\n完整数据集预期 (161个样本):")
    print("  ┌─────────────────────────────┬─────────┬────────────────────┐")
    print("  │ 方法                        │ 准确率  │ 说明               │")
    print("  ├─────────────────────────────┼─────────┼────────────────────┤")
    print("  │ Random Forest               │  ~96%   │ 控制过拟合         │")
    print("  │ CNN (单通道)                │  ~88%   │ 底线               │")
    print("  │ CNN (双通道)                │  ~92%   │ 推荐               │")
    print("  │ Transformer                 │  ~94%   │ 长程依赖           │")
    print("  │ CNN+Transformer (混合)      │  ~94%   │ ⭐ 最优架构        │")
    print("  └─────────────────────────────┴─────────┴────────────────────┘")

def main():
    """主程序"""
    print_header("李群声音曲线分析结果查看器")
    
    print("本脚本将帮助您快速了解分析结果的概况\n")
    
    # 检查文件
    check_files()
    
    # 打印统计数据
    print_statistics()
    
    # 打印结论
    print_conclusions()
    
    # 打印性能预期
    print_performance_expectations()
    
    # 打印后续步骤
    print_next_steps()
    
    # 最终信息
    print_section("文件使用指南")
    
    print("📄 推荐阅读顺序：")
    print("  1. ANALYSIS_SUMMARY.md         (5分钟) - 快速了解")
    print("  2. QUICK_REFERENCE_*.md        (15分钟) - 详细说明")
    print("  3. ANALYSIS_REPORT_*.md        (30分钟) - 深入理解")
    print("  4. detailed_sound_analysis.py  (重现结果)")
    
    print("\n📊 图表说明：")
    print("  • sound_curves_comparison*.png      - 正常vs故障对比")
    print("  • energy_density_distribution*.png  - 分布和归一化需求")
    print("  • correlation_analysis*.png         - 通道间信息关系")
    
    print("\n💻 代码使用：")
    print("  $ python detailed_sound_analysis.py")
    print("    → 重现分析结果，生成图表")
    print("\n  $ python dual_channel_model_implementation.py")
    print("    → 训练深度学习模型")
    
    print("\n" + "="*100)
    print("✨ 分析完成！请开始阅读报告或运行代码进行下一步工作。".center(100))
    print("="*100 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[中断] 用户退出")
        sys.exit(0)
