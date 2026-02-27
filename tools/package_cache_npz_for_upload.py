#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 sound_api/cache_npz 按体积分卷打包，便于上传到云端。"""

import argparse
import json
import tarfile
from pathlib import Path
from typing import List, Dict, Tuple


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def collect_bearing_dirs(cache_dir: Path) -> List[Tuple[str, Path, int]]:
    items: List[Tuple[str, Path, int]] = []
    for d in sorted(cache_dir.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.isdigit():
            continue
        size_b = dir_size_bytes(d)
        items.append((d.name, d, size_b))
    return items


def split_chunks(items: List[Tuple[str, Path, int]], max_bytes: int) -> List[List[Tuple[str, Path, int]]]:
    chunks: List[List[Tuple[str, Path, int]]] = []
    current: List[Tuple[str, Path, int]] = []
    current_size = 0

    for item in items:
        _, _, sz = item
        if current and current_size + sz > max_bytes:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(item)
        current_size += sz

    if current:
        chunks.append(current)

    return chunks


def human_gb(b: int) -> float:
    return b / (1024 ** 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="分卷打包 cache_npz 供云端上传")
    parser.add_argument("--cache_dir", type=str, default="datasets/sound_api/cache_npz", help="NPZ 缓存目录")
    parser.add_argument("--output_dir", type=str, default="datasets/sound_api/upload_packages", help="输出目录")
    parser.add_argument("--max_chunk_gb", type=float, default=4.0, help="每卷最大体积（GB）")
    parser.add_argument("--prefix", type=str, default="cache_npz_chunk", help="分卷文件名前缀")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir.exists():
        raise FileNotFoundError(f"缓存目录不存在: {cache_dir}")

    items = collect_bearing_dirs(cache_dir)
    if not items:
        raise ValueError(f"未找到 bearing 目录: {cache_dir}")

    max_bytes = int(args.max_chunk_gb * (1024 ** 3))
    chunks = split_chunks(items, max_bytes=max_bytes)

    manifest: Dict = {
        "cache_dir": str(cache_dir),
        "max_chunk_gb": args.max_chunk_gb,
        "num_bearings": len(items),
        "num_chunks": len(chunks),
        "chunks": [],
        "index_files": [],
    }

    for idx, chunk in enumerate(chunks, start=1):
        tar_name = f"{args.prefix}_{idx:03d}.tar"
        tar_path = output_dir / tar_name

        chunk_bearings = [bid for bid, _, _ in chunk]
        chunk_size = sum(sz for _, _, sz in chunk)

        with tarfile.open(tar_path, "w") as tf:
            for bid, dpath, _ in chunk:
                tf.add(dpath, arcname=f"cache_npz/{bid}")

        manifest["chunks"].append({
            "chunk": idx,
            "file": tar_name,
            "num_bearings": len(chunk_bearings),
            "size_bytes": chunk_size,
            "size_gb": round(human_gb(chunk_size), 4),
            "bearing_ids": chunk_bearings,
        })
        print(f"[{idx}/{len(chunks)}] {tar_name}: {len(chunk_bearings)} bearings, {human_gb(chunk_size):.3f} GB")

    # 同步索引文件到上传目录（如果存在）
    for name in ("index.jsonl", "index_small.jsonl", "index_small.csv"):
        src = cache_dir / name
        if src.exists():
            dst = output_dir / name
            dst.write_bytes(src.read_bytes())
            manifest["index_files"].append(name)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n打包完成: {output_dir}")
    print(f"清单文件: {manifest_path}")


if __name__ == "__main__":
    main()
