#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试原始 t 质量告警功能

模拟不同场景：
1. 正常场景：原始 t = [0,1,2,3,4]，无问题
2. 乱序场景：原始 t = [0,3,1,4,2]，50% 乱序
3. 重复场景：原始 t = [0,1,1,2,3]，20% 重复
4. 跳号场景：原始 t = [0,1,5,6,7]，从2跳到5
5. 混合场景：原始 t = [5,3,3,7,1]，多种问题
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.build_sound_api_cache import check_original_t_quality


def create_mock_tasks(bearing_id: str, orig_t_list):
    """创建模拟任务列表"""
    tasks = []
    for idx, orig_t in enumerate(orig_t_list):
        tasks.append({
            'bearing_id': bearing_id,
            't': idx,  # renumber 后的 t（总是 0..N-1）
            'original_t': orig_t,  # 原始的 t（可能有问题）
            'source_path': f'/mock/{bearing_id}/file_{idx}.json'
        })
    return tasks


def test_scenario(name: str, bearing_id: str, orig_t_list, threshold: float = 0.3):
    """测试一个场景"""
    print(f"\n{'='*80}")
    print(f"场景: {name}")
    print(f"Bearing: {bearing_id}")
    print(f"原始 t: {orig_t_list}")
    print(f"告警阈值: {threshold*100:.0f}%")
    print('-' * 80)
    
    tasks = create_mock_tasks(bearing_id, orig_t_list)
    warnings = check_original_t_quality(tasks, threshold=threshold)
    
    if bearing_id in warnings:
        info = warnings[bearing_id]
        print(f"[WARNING] 检测到问题!")
        print(f"  总文件数: {info['total_files']}")
        print(f"  有显式 t 的文件数: {info['files_with_orig_t']}")
        print(f"  问题比例: {info['problem_ratio']*100:.1f}%")
        print(f"  问题类型:")
        print(f"    - 乱序（不严格递增）: {info['issues']['disorder']} 处")
        print(f"    - 重复值: {info['issues']['duplicate']} 个")
        print(f"    - 跳号/不连续: {info['issues']['gaps']} 处")
    else:
        print(f"[OK] 未达到告警阈值（质量良好或问题比例低于 {threshold*100:.0f}%）")


def main():
    print("=" * 80)
    print("原始 t 质量告警功能测试")
    print("=" * 80)
    
    # 场景1：正常情况
    test_scenario(
        name="正常场景（无问题）",
        bearing_id="bearing_normal",
        orig_t_list=[0, 1, 2, 3, 4]
    )
    
    # 场景2：乱序
    test_scenario(
        name="乱序场景（50% 乱序，应告警）",
        bearing_id="bearing_disorder",
        orig_t_list=[0, 3, 1, 4, 2]
    )
    
    # 场景3：重复
    test_scenario(
        name="重复场景（20% 重复，低于30%阈值）",
        bearing_id="bearing_duplicate",
        orig_t_list=[0, 1, 1, 2, 3]
    )
    
    # 场景4：跳号
    test_scenario(
        name="跳号场景（从2跳到5，应告警）",
        bearing_id="bearing_gaps",
        orig_t_list=[0, 1, 5, 6, 7]
    )
    
    # 场景5：混合问题
    test_scenario(
        name="混合问题场景（乱序+重复+跳号，应告警）",
        bearing_id="bearing_mixed",
        orig_t_list=[5, 3, 3, 7, 1]
    )
    
    # 场景6：部分文件有 original_t
    print(f"\n{'='*80}")
    print("场景: 部分文件缺失原始 t（测试容错性）")
    print('-' * 80)
    tasks = [
        {'bearing_id': 'bearing_partial', 't': 0, 'original_t': 0, 'source_path': '/mock/file0.json'},
        {'bearing_id': 'bearing_partial', 't': 1, 'original_t': None, 'source_path': '/mock/file1.json'},
        {'bearing_id': 'bearing_partial', 't': 2, 'original_t': 5, 'source_path': '/mock/file2.json'},
        {'bearing_id': 'bearing_partial', 't': 3, 'original_t': None, 'source_path': '/mock/file3.json'},
    ]
    warnings = check_original_t_quality(tasks, threshold=0.3)
    
    if 'bearing_partial' in warnings:
        info = warnings['bearing_partial']
        print(f"[WARNING] 检测到问题!")
        print(f"  总文件数: {info['total_files']}")
        print(f"  有显式 t 的文件数: {info['files_with_orig_t']}")
        print(f"  问题比例: {info['problem_ratio']*100:.1f}%")
    else:
        print("[OK] 未达到告警阈值（只有2个文件有 original_t，跳号1处，比例50%但样本少）")
    
    print(f"\n{'='*80}")
    print("测试完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
