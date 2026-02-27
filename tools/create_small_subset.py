"""
快速生成小数据集索引（用于 Cloud Studio 快速验证）
用法：python tools/create_small_subset.py --n_bearings 5 --output index_small.jsonl
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import tarfile
import os

def main():
    parser = argparse.ArgumentParser(description="从完整 index.jsonl 生成小数据集索引")
    parser.add_argument(
        "--index_path",
        type=str,
        default="datasets/sound_api/cache_npz/index.jsonl",
        help="完整索引文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/sound_api/cache_npz/index_small.jsonl",
        help="输出的小索引文件路径"
    )
    parser.add_argument(
        "--n_bearings",
        type=int,
        default=10,
        help="选择前 N 个 bearing（默认 10）"
    )
    parser.add_argument(
        "--tar",
        action="store_true",
        help="是否打包所选 bearing 目录为tar，便于上传"
    )
    args = parser.parse_args()

    index_path = Path(args.index_path)
    output_path = Path(args.output)
    
    if not index_path.exists():
        raise FileNotFoundError(f"索引文件不存在: {index_path}")

    # 按 bearing_id 分组
    bearing_groups = defaultdict(list)
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bearing_groups[rec["bearing_id"]].append(rec)

    # 选择前 N 个 bearing
    selected_bearings = sorted(bearing_groups.keys())[:args.n_bearings]
    print(f"选择 {len(selected_bearings)} 个 bearing: {selected_bearings}")

    # 统计样本数
    total_samples = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for bid in selected_bearings:
            for rec in bearing_groups[bid]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_samples += 1

    print(f"小数据集索引已写入: {output_path}")
    print(f"共 {total_samples} 个样本")
    print(f"\n下一步：")
    print(f"1. 上传 {output_path} 到 Cloud Studio")
    print(f"2. 上传对应的 NPZ 文件（只上传这些 bearing 的目录）")
    print(f"   例如：datasets/sound_api/cache_npz/{selected_bearings[0]}/")
    print(f"        datasets/sound_api/cache_npz/{selected_bearings[1]}/")
    print(f"        ...")

    # 新增：打包成 tar 包
    if args.tar:
        cache_dir = output_path.parent  # cache_npz
        tar_output_dir = cache_dir.parent  # cache_npz的上级
        tar_name = f"{output_path.stem}_npz_subset.tar"
        tar_path = tar_output_dir / tar_name
        print(f"\n正在将以下目录打包为 {tar_path}:")
        with tarfile.open(tar_path, "w") as tar:
            for bid in selected_bearings:
                bearing_dir = cache_dir / bid
                if not bearing_dir.exists():
                    print(f"警告: 目录不存在，跳过: {bearing_dir}")
                    continue
                print(f"  - {bearing_dir}")
                # 把 bearing 目录整体打包
                tar.add(bearing_dir, arcname=bearing_dir.name)
        print(f"打包完成，tar 文件路径: {tar_path}")

if __name__ == "__main__":
    main()