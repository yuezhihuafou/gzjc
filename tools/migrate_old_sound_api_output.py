#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
迁移历史产物：tools/sound_api/ 下的输出目录 → datasets/sound_api/

目的：避免 IDE 索引 tools/ 下的数据导致卡顿
"""
import os
import shutil
from pathlib import Path

# 源目录（旧产物位置）
OLD_OUTPUTS = [
    'tools/sound_api/sound_api_output',
    'tools/sound_api/sound_api_output_test',
]

# 目标根目录
NEW_ROOT = 'datasets/sound_api/output_json'


def migrate_directory(old_dir: str, new_root: str, bearing_id_hint: str = 'migrated'):
    """
    将旧输出目录迁移到新规范目录
    
    Args:
        old_dir: 旧目录路径
        new_root: 新根目录
        bearing_id_hint: bearing_id 提示（用于分桶）
    """
    old_path = Path(old_dir)
    if not old_path.exists():
        print(f"跳过（不存在）: {old_dir}")
        return
    
    # 目标目录（按 bearing_id 分桶，这里用目录名作为默认 bearing_id）
    bearing_id = old_path.name  # 如 sound_api_output
    target_dir = Path(new_root) / bearing_id_hint / old_path.name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计
    files_moved = 0
    files_skipped = 0
    
    # 遍历所有文件
    for item in old_path.iterdir():
        if item.is_file():
            target_file = target_dir / item.name
            
            # 检查是否已存在
            if target_file.exists():
                print(f"  跳过（已存在）: {target_file.name}")
                files_skipped += 1
            else:
                shutil.move(str(item), str(target_file))
                print(f"  迁移: {item.name} → {target_file}")
                files_moved += 1
    
    print(f"完成: 迁移 {files_moved} 个文件，跳过 {files_skipped} 个")
    
    # 如果源目录为空，删除它
    if not any(old_path.iterdir()):
        old_path.rmdir()
        print(f"  删除空目录: {old_dir}")


def main():
    print("=" * 80)
    print("迁移 tools/sound_api/ 下的历史产物到 datasets/sound_api/")
    print("=" * 80)
    print()
    
    for old_dir in OLD_OUTPUTS:
        print(f"\n迁移: {old_dir}")
        print("-" * 80)
        migrate_directory(old_dir, NEW_ROOT)
    
    print("\n" + "=" * 80)
    print("迁移完成")
    print("=" * 80)
    print()
    print("提示：")
    print("1. 迁移后的文件位于 datasets/sound_api/output_json/migrated/")
    print("2. 如需重新按 bearing_id 分桶，请手动整理或重新运行 API 转换")
    print("3. 建议清理后，在 .gitignore 中添加 tools/sound_api/*_output/")


if __name__ == '__main__':
    main()
