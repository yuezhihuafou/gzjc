"""
验证 cache_npz 中的 fault_label 是否与源数据 sidecar 中的标签一一对应，
以及 NPZ -> JSON -> .f -> sidecar 的还原链路是否信息无损（在标签这一维）。

用法示例：
  python tools/verify_cache_labels.py --cache_dir datasets/sound_api/cache_npz
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

# 确保脚本运行时可以找到项目内模块（如 dl.sound_api_cache_dataset）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dl.sound_api_cache_dataset import resolve_fault_label_from_f_source


def check_one_npz(npz_path: Path) -> Tuple[str, int, int]:
    """
    检查单个 NPZ 的标签是否与 sidecar 一致。

    返回 (status, label_npz, label_src)，其中：
    - status:
        - "ok":    两边都有标签，且数值相同
        - "miss_npz": sidecar 有标签，但 NPZ 中没有或为负
        - "miss_src": NPZ 中有非负标签，但 sidecar 还原不到标签
        - "mismatch": 两边都有非负标签，但数值不一致
        - "both_missing": 两边都没有有效标签
        - "error":  读取 NPZ 或还原时出错
    """
    try:
        with np.load(npz_path) as data:
            if "fault_label" in data:
                try:
                    label_npz = int(data["fault_label"])
                except Exception:
                    label_npz = -1
            else:
                label_npz = -1
    except Exception:
        return "error", -1, -1

    # 通过 NPZ -> JSON -> .f -> sidecar 还原标签
    label_src = resolve_fault_label_from_f_source(npz_path)

    has_npz = label_npz is not None and label_npz >= 0
    has_src = label_src is not None and label_src >= 0

    if has_npz and has_src:
        if int(label_npz) == int(label_src):
            return "ok", int(label_npz), int(label_src)
        return "mismatch", int(label_npz), int(label_src)
    if has_src and not has_npz:
        return "miss_npz", -1, int(label_src)
    if has_npz and not has_src:
        return "miss_src", int(label_npz), -1
    if not has_npz and not has_src:
        return "both_missing", -1, -1
    return "error", label_npz, label_src


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证 cache_npz 中 fault_label 与 sidecar 标签的一致性（NPZ↔源标签 信息无损）"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="datasets/sound_api/cache_npz",
        help="NPZ 缓存根目录（build_sound_api_cache 输出）",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="verify_cache_labels.log",
        help="不一致/缺失标签的明细输出路径",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行 worker 数（默认 8）",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        raise SystemExit(f"cache_dir 不存在: {cache_path}")

    npz_files = [p for p in cache_path.rglob("*.npz") if not p.name.endswith(".tmp.npz")]
    if not npz_files:
        raise SystemExit(f"在 {cache_path} 下未找到 npz 文件")

    print(f"将在 {cache_path} 下检查 {len(npz_files)} 个 NPZ 标签一致性...")

    stats = {
        "ok": 0,
        "miss_npz": 0,
        "miss_src": 0,
        "mismatch": 0,
        "both_missing": 0,
        "error": 0,
    }
    log_lines = []

    # 多 worker 并行检查
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = ex.map(check_one_npz, npz_files, chunksize=128)
        for npz_path, (status, label_npz, label_src) in tqdm(
            zip(npz_files, results),
            total=len(npz_files),
            desc="verify_labels",
            unit="file",
            ncols=100,
        ):
            stats[status] = stats.get(status, 0) + 1
            if status != "ok":
                log_lines.append(
                    f"{npz_path}\tstatus={status}\tlabel_npz={label_npz}\tlabel_src={label_src}"
                )

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# NPZ vs sidecar 标签一致性检查结果\n")
        for line in log_lines:
            f.write(line + os.linesep)

    print("\n检查完成：")
    for k in ["ok", "mismatch", "miss_npz", "miss_src", "both_missing", "error"]:
        print(f"  {k}: {stats.get(k, 0)}")
    print(f"明细已写入: {log_path}")


if __name__ == "__main__":
    main()

