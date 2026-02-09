"""
将 sound_api_output 的 Excel 文件转换为 NPZ 缓存

输入：datasets/sound_api_output/index.csv
输出：datasets/sound_api_cache/{bearing_id}/{t:06d}.npz

处理流程：
1. 读取 index.csv
2. 对每个 xlsx 文件：
   - 跳过前两行
   - 读取 3000 行三列（frequency, volume, density）
   - 变换：x[0]=log1p(volume), x[1]=density
   - 保存为 (2, 3000) 的 float32 数组
3. 可选保存元数据：frequency, bearing_id, t, source_path
"""
import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd


def process_single_xlsx(
    xlsx_path: str,
    bearing_id: str,
    t: int,
    output_dir: str,
    max_rows: int = 3000,
    save_metadata: bool = True
) -> Tuple[bool, str]:
    """
    处理单个 xlsx 文件，转换为 NPZ
    
    Args:
        xlsx_path: xlsx 文件路径
        bearing_id: bearing ID
        t: 时间序号
        output_dir: 输出目录
        max_rows: 最大读取行数（默认 3000）
        save_metadata: 是否保存元数据
    
    Returns:
        (success, message): 成功标志和消息
    """
    try:
        # 读取 xlsx（跳过前两行）
        df = pd.read_excel(xlsx_path, header=None, skiprows=2, nrows=max_rows)
        
        if df.shape[0] < max_rows:
            return False, f"行数不足 {max_rows}，实际 {df.shape[0]}"
        
        if df.shape[1] < 3:
            return False, f"列数不足 3，实际 {df.shape[1]}"
        
        # 提取三列
        frequency = df.iloc[:, 0].values.astype(np.float32)
        volume = df.iloc[:, 1].values.astype(np.float32)
        density = df.iloc[:, 2].values.astype(np.float32)
        
        # 变换：x[0]=log1p(volume), x[1]=density
        x = np.zeros((2, max_rows), dtype=np.float32)
        x[0] = np.log1p(volume)  # log1p(volume)
        x[1] = density
        
        # 创建输出目录
        bearing_dir = Path(output_dir) / bearing_id
        bearing_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 NPZ
        npz_path = bearing_dir / f"{t:06d}.npz"
        
        save_dict = {
            'x': x,  # (2, 3000) float32
        }
        
        if save_metadata:
            save_dict.update({
                'frequency': frequency,
                'bearing_id': bearing_id,
                't': t,
                'source_path': xlsx_path
            })
        
        np.savez_compressed(npz_path, **save_dict)
        
        return True, f"成功: {npz_path}"
    
    except Exception as e:
        return False, f"错误: {str(e)}"


def process_batch(args_list: List[Tuple]) -> List[Tuple[bool, str, str]]:
    """
    批量处理（用于多进程）
    
    Args:
        args_list: [(xlsx_path, bearing_id, t, output_dir, max_rows, save_metadata), ...]
    
    Returns:
        List[(success, xlsx_path, message)]
    """
    results = []
    for args in args_list:
        xlsx_path = args[0]  # 第一个参数是 xlsx_path
        success, message = process_single_xlsx(*args)
        results.append((success, xlsx_path, message))
    return results


def load_index(index_path: str) -> List[Dict]:
    """加载 index.csv"""
    records = []
    with open(index_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'path': row['path'],
                'bearing_id': row['bearing_id'],
                't': int(row['t'])
            })
    return records


def main():
    parser = argparse.ArgumentParser(description='将 Excel 转换为 NPZ 缓存')
    parser.add_argument(
        '--index',
        type=str,
        default='datasets/sound_api_output/index.csv',
        help='索引文件路径（默认: datasets/sound_api_output/index.csv）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='datasets/sound_api_cache',
        help='输出目录（默认: datasets/sound_api_cache）'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=3000,
        help='最大读取行数（默认: 3000）'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='并行工作进程数（默认: CPU 核心数）'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='不保存元数据（仅保存 x 数组）'
    )
    parser.add_argument(
        '--bad_files',
        type=str,
        default='datasets/sound_api_output/bad_files_cache.txt',
        help='问题文件列表路径（默认: datasets/sound_api_output/bad_files_cache.txt）'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Excel → NPZ 缓存转换")
    print("=" * 80)
    
    # 加载索引
    print(f"\n加载索引: {args.index}")
    if not os.path.exists(args.index):
        print(f"错误: 索引文件不存在: {args.index}")
        print("请先运行 tools/build_sound_api_index.py 生成索引")
        return
    
    records = load_index(args.index)
    print(f"找到 {len(records)} 个文件需要处理")
    
    # 准备参数
    save_metadata = not args.no_metadata
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定工作进程数
    if args.workers is None:
        workers = multiprocessing.cpu_count()
    else:
        workers = args.workers
    
    print(f"使用 {workers} 个并行进程")
    
    # 准备任务列表
    tasks = [
        (
            record['path'],
            record['bearing_id'],
            record['t'],
            str(output_dir),
            args.max_rows,
            save_metadata
        )
        for record in records
    ]
    
    # 分批处理（每批包含多个任务，减少进程间通信开销）
    batch_size = max(1, len(tasks) // (workers * 4))
    batches = []
    for i in range(0, len(tasks), batch_size):
        batches.append(tasks[i:i + batch_size])
    
    print(f"分为 {len(batches)} 个批次处理")
    
    # 多进程处理
    success_count = 0
    fail_count = 0
    bad_files = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        completed = 0
        for future in as_completed(futures):
            batch_results = future.result()
            for success, xlsx_path, message in batch_results:
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    bad_files.append({
                        'path': xlsx_path,
                        'reason': message
                    })
                    print(f"失败: {xlsx_path} - {message}")
            
            completed += 1
            if completed % max(1, len(batches) // 10) == 0:
                print(f"进度: {completed}/{len(batches)} 批次完成")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("处理完成")
    print("=" * 80)
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    
    # 写入问题文件列表
    if bad_files:
        bad_files_path = Path(args.bad_files)
        bad_files_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bad_files_path, 'w', encoding='utf-8') as f:
            f.write("缓存转换问题文件列表:\n")
            f.write("=" * 80 + "\n")
            for item in bad_files:
                f.write(f"文件: {item['path']}\n")
                f.write(f"原因: {item['reason']}\n")
                f.write("-" * 80 + "\n")
        print(f"\n问题文件列表已写入: {bad_files_path}")
    
    # 检查输出目录结构
    bearing_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    print(f"\n输出目录结构:")
    print(f"  bearing 目录数: {len(bearing_dirs)}")
    total_npz = sum(len(list(d.glob('*.npz'))) for d in bearing_dirs)
    print(f"  总 NPZ 文件数: {total_npz}")


if __name__ == '__main__':
    # Windows 多进程需要这个保护
    multiprocessing.freeze_support()
    main()
