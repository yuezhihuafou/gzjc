#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证声音数据加载器是否正确识别所有samples
"""
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.load_sound import SoundDataLoader

def verify_sound_data():
    print("=" * 80)
    print("验证声音数据加载")
    print("=" * 80)
    
    # 初始化加载器
    loader = SoundDataLoader()
    
    # 获取统计信息
    stats = loader.get_statistics()
    available = loader.get_available_files()
    
    print(f"\n总xlsx文件数: {len(loader.file_mapping)}")
    print(f"总Sheet/样本数: {len(available)}")
    
    # 显示样本分布
    print(f"\n前20个样本:")
    for i, sample in enumerate(sorted(available)[:20]):
        print(f"  {i+1:2d}. {sample}")
    
    # 测试加载前5个样本
    print(f"\n测试加载前5个样本:")
    success_count = 0
    for sample in sorted(available)[:5]:
        curves = loader.load_sound_curves(sample)
        if curves is not None:
            freq_len = len(curves['frequency'])
            vol_min = curves['volume'].min()
            vol_max = curves['volume'].max()
            den_min = curves['density'].min()
            den_max = curves['density'].max()
            
            print(f"  ✓ {sample:20s}: {freq_len} pts, "
                  f"vol=[{vol_min:.2f}, {vol_max:.2f}], "
                  f"den=[{den_min:.2f}, {den_max:.2f}]")
            success_count += 1
        else:
            print(f"  ✗ {sample:20s}: 加载失败")
    
    print(f"\n加载成功率: {success_count}/5")
    
    # 检查覆盖的原始.mat文件
    print(f"\n=== 与CWRU数据集对比 ===")
    import json
    metadata_path = Path(__file__).parent.parent / 'cwru_processed' / 'metadata.json'
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        original_files = set()
        for item in metadata:
            mat_file = item['filename'].replace('.mat', '')
            original_files.add(mat_file)
        
        available_set = set(available)
        coverage = len(available_set & original_files) / len(original_files) * 100
        
        print(f"CWRU原始.mat文件数: {len(original_files)}")
        print(f"声音数据覆盖的样本: {len(available_set & original_files)}")
        print(f"覆盖率: {coverage:.2f}%")
        
        uncovered = original_files - available_set
        print(f"未覆盖的样本数: {len(uncovered)}")
    
    print("\n" + "=" * 80)
    print("验证完成")
    print("=" * 80)

if __name__ == '__main__':
    verify_sound_data()
