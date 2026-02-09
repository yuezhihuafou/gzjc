"""
从 .f 的 sidecar JSON 给已生成的 output_json 和 cache_npz 补写标签（无损还原）。

用法：
  # 只补 output_json 的 metadata（fault_label / health_label）
  python tools/backfill_labels_from_sidecar.py --mode output_json --output_json_dir datasets/sound_api/output_json

  # 只补 NPZ 缓存的 fault_label（多 worker + 错误落 log）
  python tools/backfill_labels_from_sidecar.py --mode cache_npz --cache_dir datasets/sound_api/cache_npz --workers 8 --log backfill_errors.log

  # 两者都补
  python tools/backfill_labels_from_sidecar.py --mode both --output_json_dir datasets/sound_api/output_json --cache_dir datasets/sound_api/cache_npz --workers 8
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.json_io import loadFirstJson
from tqdm import tqdm


def read_label_from_sidecar(f_path: Path) -> int:
    """从 .f 对应的 sidecar .json 读取 health_label 或 fault_label，无则返回 -1。"""
    sidecar = f_path.with_suffix(".json")
    if not sidecar.exists():
        return -1
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            s = json.load(f)
        label = s.get("fault_label", s.get("health_label", -1))
        if label is None:
            return -1
        val = int(label)
        return val if val >= 0 else -1
    except Exception:
        return -1


def _process_one_output_json(jpath: Path, root: Path, dry_run: bool) -> Tuple[str, Optional[str]]:
    """处理单个 output_json。返回 ('ok'|'skip'|'fail', 错误信息或 None)。"""
    try:
        obj = loadFirstJson(jpath)
    except Exception as e:
        return "fail", f"读 JSON 失败: {e}"
    meta = obj.get("metadata")
    if not meta:
        return "skip", None
    f_path_str = meta.get("source_path")
    if not f_path_str:
        return "skip", None
    f_path = Path(f_path_str)
    if not f_path.exists():
        f_path_alt = (root / f_path_str).resolve()
        if f_path_alt.exists():
            f_path = f_path_alt
        else:
            return "fail", "source_path 不存在"
    label = read_label_from_sidecar(f_path)
    if label < 0:
        return "skip", None
    meta["fault_label"] = label
    meta["health_label"] = label
    if not dry_run:
        try:
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return "fail", f"写 JSON 失败: {e}"
    return "ok", None


def backfill_output_json(
    output_json_dir: str,
    dry_run: bool,
    workers: int = 1,
    log_errors: Optional[List[str]] = None,
) -> Tuple[int, int, int]:
    """
    遍历 output_json 下所有 .json，用 metadata.source_path（.f 路径）找到 sidecar，补写 metadata.fault_label。
    返回 (成功数, 跳过数, 失败数)。失败详情追加到 log_errors（若提供）。
    """
    root = Path(output_json_dir)
    if not root.exists():
        print(f"目录不存在: {root}")
        return 0, 0, 0
    print("正在扫描 output_json 目录...", flush=True)
    files = list(root.rglob("*.json"))
    if not files:
        return 0, 0, 0
    total = len(files)
    print(f"共 {total} 个 .json，开始补写（分批处理）...", flush=True)
    err_list = log_errors if log_errors is not None else []
    ok, skip, fail = 0, 0, 0
    batch_size = 20000
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in range(0, total, batch_size):
            batch = files[start : start + batch_size]
            batch_num = start // batch_size + 1
            n_batches = (total + batch_size - 1) // batch_size
            print(f"  批次 {batch_num}/{n_batches} ({len(batch)} 个文件)...", flush=True)
            futures = {ex.submit(_process_one_output_json, jpath, root, dry_run): jpath for jpath in batch}
            with tqdm(
                total=len(batch),
                desc="output_json",
                unit="file",
                ncols=100,
                leave=False,
                file=sys.stderr,
                disable=False,
                mininterval=0.5,
            ) as pbar:
                for fut in as_completed(futures):
                    jpath = futures[fut]
                    try:
                        status, err_msg = fut.result()
                        if status == "ok":
                            ok += 1
                        elif status == "skip":
                            skip += 1
                        else:
                            fail += 1
                            if err_msg:
                                err_list.append(f"{jpath}\t{err_msg}")
                    except Exception as e:
                        fail += 1
                        err_list.append(f"{jpath}\t{e}")
                    done += 1
                    pbar.update(1)
                    if done % 5000 == 0:
                        print(f"  已处理 {done}/{total}", flush=True)
    return ok, skip, fail


def _process_one_cache_npz(
    npz_path: Path,
    cache_path: Path,
    dry_run: bool,
) -> Tuple[str, Optional[str]]:
    """处理单个 cache npz。返回 ('ok'|'skip'|'fail', 错误信息或 None)。"""
    import numpy as np
    if npz_path.name.endswith(".tmp.npz"):
        return "skip", None
    try:
        with np.load(npz_path) as data:
            if "source_path" not in data:
                return "skip", None
            sp = data["source_path"]
            json_path = Path(sp.item() if hasattr(sp, "item") else str(sp))
    except Exception as e:
        return "fail", f"读 NPZ 失败: {e}"
    if not json_path.is_absolute():
        json_path = (cache_path / json_path).resolve() if (cache_path / json_path).exists() else Path(os.path.abspath(str(json_path)))
    if not json_path.exists():
        return "skip", None
    try:
        obj = loadFirstJson(json_path)
        meta = obj.get("metadata") or {}
        f_path_str = meta.get("source_path")
        if not f_path_str:
            return "skip", None
        f_path = Path(f_path_str)
        if not f_path.exists():
            return "skip", None
        label = read_label_from_sidecar(f_path)
        if label < 0:
            return "skip", None
    except Exception as e:
        return "fail", f"解析 JSON/sidecar 失败: {e}"
    if not dry_run:
        try:
            with np.load(npz_path) as data:
                d = {}
                for k in data.files:
                    arr = data[k]
                    d[k] = arr.copy() if isinstance(arr, np.ndarray) else np.array(arr)
                d["fault_label"] = np.int64(label)
            tmp = npz_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **d)
            tmp.replace(npz_path)
        except Exception as e:
            return "fail", f"写 NPZ 失败: {e}"
    return "ok", None


def backfill_cache_npz(
    cache_dir: str,
    dry_run: bool,
    workers: int = 1,
    log_errors: Optional[List[str]] = None,
) -> Tuple[int, int, int]:
    """
    遍历 cache_npz 下所有 .npz，通过 NPZ.source_path → JSON → .f → sidecar 取标签，写回 NPZ.fault_label。
    返回 (成功数, 跳过数, 失败数)。失败详情追加到 log_errors（若提供）。
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"目录不存在: {cache_path}")
        return 0, 0, 0
    print("正在扫描 cache_npz 目录...", flush=True)
    files = [p for p in cache_path.rglob("*.npz") if not p.name.endswith(".tmp.npz")]
    if not files:
        return 0, 0, 0
    total = len(files)
    print(f"共 {total} 个 .npz，开始补写（分批处理）...", flush=True)
    err_list = log_errors if log_errors is not None else []
    ok, skip, fail = 0, 0, 0
    batch_size = 20000
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in range(0, total, batch_size):
            batch = files[start : start + batch_size]
            batch_num = start // batch_size + 1
            n_batches = (total + batch_size - 1) // batch_size
            print(f"  批次 {batch_num}/{n_batches} ({len(batch)} 个文件)...", flush=True)
            futures = {ex.submit(_process_one_cache_npz, npz_path, cache_path, dry_run): npz_path for npz_path in batch}
            with tqdm(
                total=len(batch),
                desc="cache_npz",
                unit="file",
                ncols=100,
                leave=False,
                file=sys.stderr,
                disable=False,
                mininterval=0.5,
            ) as pbar:
                for fut in as_completed(futures):
                    npz_path = futures[fut]
                    try:
                        status, err_msg = fut.result()
                        if status == "ok":
                            ok += 1
                        elif status == "skip":
                            skip += 1
                        else:
                            fail += 1
                            if err_msg:
                                err_list.append(f"{npz_path}\t{err_msg}")
                    except Exception as e:
                        fail += 1
                        err_list.append(f"{npz_path}\t{e}")
                    done += 1
                    pbar.update(1)
                    if done % 5000 == 0:
                        print(f"  已处理 {done}/{total}", flush=True)
    return ok, skip, fail


def main():
    parser = argparse.ArgumentParser(
        description="从 .f 的 sidecar 给已生成的 output_json 和 cache_npz 补写标签"
    )
    parser.add_argument(
        "--mode",
        choices=["output_json", "cache_npz", "both"],
        default="both",
        help="补写目标：output_json / cache_npz / both",
    )
    parser.add_argument(
        "--output_json_dir",
        type=str,
        default="datasets/sound_api/output_json",
        help="output_json 根目录（convert_mc_to_api_json 输出）",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="datasets/sound_api/cache_npz",
        help="NPZ 缓存根目录（build_sound_api_cache 输出）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只统计可补写数量，不写入文件",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行 worker 数（默认 8）",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="backfill_errors.log",
        help="失败明细输出到该 log 文件（默认 backfill_errors.log）",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("（dry_run：不写入文件）")
    print(f"workers={args.workers}, 错误输出: {args.log}")

    log_errors: List[str] = []
    if args.mode in ("output_json", "both"):
        print("\n[output_json] 从 sidecar 补写 metadata.fault_label / health_label ...")
        ok, skip, fail = backfill_output_json(
            args.output_json_dir, args.dry_run, workers=args.workers, log_errors=log_errors
        )
        print(f"  成功: {ok}, 跳过: {skip}, 失败: {fail}")

    if args.mode in ("cache_npz", "both"):
        print("\n[cache_npz] 从 sidecar 补写 NPZ.fault_label ...")
        ok, skip, fail = backfill_cache_npz(
            args.cache_dir, args.dry_run, workers=args.workers, log_errors=log_errors
        )
        print(f"  成功: {ok}, 跳过: {skip}, 失败: {fail}")

    if log_errors:
        log_path = Path(args.log)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# backfill 失败明细\t原因\n")
            for line in log_errors:
                f.write(line.strip() + "\n")
        print(f"\n失败明细已写入: {log_path}（共 {len(log_errors)} 条）")
    print("\n完成。")


if __name__ == "__main__":
    main()
