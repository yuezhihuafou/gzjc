#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 NPZ 文件中的 orig_t 字段

验证：
1. NPZ 文件中包含 t 和 orig_t 两个字段
2. t 为重编号（0..T-1）
3. orig_t 保留原始值（可能乱序/跳号）
"""
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_npz_structure(npz_path: str):
    """测试单个 NPZ 文件的结构"""
    print(f"\n检查文件: {npz_path}")
    print("-" * 80)
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        print("字段列表:")
        for key in data.keys():
            value = data[key]
            if hasattr(value, 'shape'):
                print(f"  - {key}: {value.dtype}, shape={value.shape}")
            else:
                print(f"  - {key}: {type(value).__name__}, value={value}")
        
        print("\n关键字段:")
        if 't' in data:
            print(f"  t (训练用): {data['t']}")
        else:
            print("  t: [缺失]")
        
        if 'orig_t' in data:
            print(f"  orig_t (追溯用): {data['orig_t']}")
        else:
            print("  orig_t: [缺失]")
        
        if 'bearing_id' in data:
            print(f"  bearing_id: {data['bearing_id']}")
        
        if 'source_path' in data:
            print(f"  source_path: {data['source_path']}")
        
        # 验收标准
        print("\n验收结果:")
        checks = []
        
        # 检查1：必须有 x 字段
        if 'x' in data and data['x'].shape == (2, 3000):
            checks.append(("x 字段", True, f"shape={data['x'].shape}"))
        else:
            checks.append(("x 字段", False, "缺失或形状不对"))
        
        # 检查2：必须有 t 字段
        if 't' in data:
            checks.append(("t 字段（训练用）", True, f"value={data['t']}"))
        else:
            checks.append(("t 字段（训练用）", False, "缺失"))
        
        # 检查3：orig_t 字段（可选，但如果有原始 t 应该存在）
        if 'orig_t' in data:
            checks.append(("orig_t 字段（追溯用）", True, f"value={data['orig_t']}"))
        else:
            checks.append(("orig_t 字段（追溯用）", False, "缺失（可能无原始 t）"))
        
        for name, passed, detail in checks:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} {name}: {detail}")
        
        return all(check[1] for check in checks[:2])  # x 和 t 必须通过
        
    except Exception as e:
        print(f"[ERROR] 读取失败: {str(e)}")
        return False


def test_bearing_directory(cache_dir: str, bearing_id: str):
    """测试某个 bearing 目录下的所有 NPZ 文件"""
    print("\n" + "=" * 80)
    print(f"测试 bearing: {bearing_id}")
    print("=" * 80)
    
    bearing_path = Path(cache_dir) / bearing_id
    if not bearing_path.exists():
        print(f"[ERROR] 目录不存在: {bearing_path}")
        return
    
    npz_files = sorted(bearing_path.glob('*.npz'))
    if not npz_files:
        print(f"[WARNING] 目录为空: {bearing_path}")
        return
    
    print(f"找到 {len(npz_files)} 个 NPZ 文件")
    
    # 测试前3个文件
    for i, npz_file in enumerate(npz_files[:3]):
        test_npz_structure(str(npz_file))
    
    if len(npz_files) > 3:
        print(f"\n... 省略其余 {len(npz_files) - 3} 个文件")
    
    # 汇总统计
    print("\n" + "-" * 80)
    print("汇总统计:")
    
    t_values = []
    orig_t_values = []
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            if 't' in data:
                t_values.append(int(data['t']))
            if 'orig_t' in data:
                orig_t_values.append(int(data['orig_t']))
        except:
            pass
    
    print(f"  有 t 的文件数: {len(t_values)}")
    print(f"  t 范围: {min(t_values) if t_values else 'N/A'} ~ {max(t_values) if t_values else 'N/A'}")
    print(f"  t 连续性: {'连续' if t_values == list(range(len(t_values))) else '不连续'}")
    
    print(f"  有 orig_t 的文件数: {len(orig_t_values)}")
    if orig_t_values:
        print(f"  orig_t 范围: {min(orig_t_values)} ~ {max(orig_t_values)}")
        print(f"  orig_t 样本（前10个）: {orig_t_values[:10]}")


def main():
    print("=" * 80)
    print("NPZ 文件 orig_t 字段测试")
    print("=" * 80)
    
    # 默认缓存目录
    cache_dir = Path('datasets/sound_api/cache_npz')
    
    if not cache_dir.exists():
        print(f"\n[ERROR] 缓存目录不存在: {cache_dir}")
        print("请先运行 build_sound_api_cache.py 生成缓存")
        return
    
    # 列出所有 bearing
    bearings = [d.name for d in cache_dir.iterdir() if d.is_dir()]
    
    if not bearings:
        print(f"\n[WARNING] 缓存目录为空: {cache_dir}")
        return
    
    print(f"\n找到 {len(bearings)} 个 bearing")
    print("Bearings:", bearings[:5], "..." if len(bearings) > 5 else "")
    
    # 测试第一个 bearing
    if bearings:
        test_bearing_directory(str(cache_dir), bearings[0])
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    print("\n提示：")
    print("- t 字段为重编号（0..T-1），用于训练")
    print("- orig_t 字段保留原始采集顺序，用于追溯")
    print("- 如果文件名中无显式 t，orig_t 可能缺失")


if __name__ == '__main__':
    main()
