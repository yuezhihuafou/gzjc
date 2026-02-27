#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建统一标签表（master_labels.csv）。

当前支持从 CWRU 处理目录生成标签表：
- 读取 labels.npy
- 若存在 metadata.json，则补充 load/rpm/fault_type/trace_id 等字段
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


FAULT_TYPE_MAP = {
    0: "Normal",
    1: "B",
    2: "IR",
    3: "OR",
}


def _to_float_or_empty(v: Any):
    if v is None:
        return ""
    try:
        return float(v)
    except Exception:
        return ""


def _build_condition_id(dataset: str, load_hp: Any, rpm: Any) -> str:
    if load_hp in (None, "") or rpm in (None, ""):
        return f"{dataset}_unknown"
    return f"{dataset}_load{int(load_hp)}_rpm{int(rpm)}"


def build_master_labels_from_cwru(
    base_dir: Path,
    output_csv: Path,
    label_source: str,
    label_version: str,
    device_id: str,
) -> None:
    labels_path = base_dir / "labels.npy"
    metadata_path = base_dir / "metadata.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.npy 不存在: {labels_path}")

    labels = np.load(labels_path)
    if labels.ndim != 1:
        raise ValueError(f"labels.npy 期望 1 维，实际 {labels.shape}")

    metadata: List[Dict[str, Any]] = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, list):
            raise ValueError("metadata.json 格式错误，应为 list")
        if len(metadata) != len(labels):
            raise ValueError(
                f"metadata.json 与 labels.npy 长度不一致: {len(metadata)} vs {len(labels)}"
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "dataset",
        "device_id",
        "condition_id",
        "speed_rpm",
        "load_value",
        "load_unit",
        "source_label",
        "fault_binary",
        "fault_type",
        "label_source",
        "label_version",
        "event_time",
        "window_start",
        "window_end",
        "trace_id",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, lb in enumerate(labels.tolist()):
            m = metadata[i] if metadata else {}
            load_hp = m.get("load_hp")
            rpm = m.get("rpm")
            fault_type = m.get("fault_type") or FAULT_TYPE_MAP.get(int(lb), "Unknown")
            trace_id = (
                m.get("rel_path")
                or m.get("file")
                or m.get("filepath")
                or m.get("source")
                or f"sample_{i}"
            )

            row = {
                "sample_id": i,
                "dataset": "cwru",
                "device_id": device_id,
                "condition_id": _build_condition_id("cwru", load_hp, rpm),
                "speed_rpm": _to_float_or_empty(rpm),
                "load_value": _to_float_or_empty(load_hp),
                "load_unit": "hp" if load_hp not in (None, "") else "",
                "source_label": int(lb),
                "fault_binary": 0 if int(lb) == 0 else 1,
                "fault_type": fault_type,
                "label_source": label_source,
                "label_version": label_version,
                "event_time": "",
                "window_start": "",
                "window_end": "",
                "trace_id": trace_id,
            }
            writer.writerow(row)

    print(f"master_labels 已生成: {output_csv}")
    print(f"样本数: {len(labels)}")
    if metadata:
        known_conditions = set()
        for m in metadata:
            known_conditions.add(_build_condition_id("cwru", m.get("load_hp"), m.get("rpm")))
        print(f"工况数: {len(known_conditions)}")
    else:
        print("未检测到 metadata.json，condition_id 将统一为 cwru_unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建统一标签表 master_labels.csv")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cwru"],
        default="cwru",
        help="当前支持数据集类型",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="cwru_processed",
        help="处理后数据目录（包含 labels.npy，建议同时包含 metadata.json）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="datasets/labels/master_labels.csv",
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--label_source",
        type=str,
        default="dataset_rule",
        help="标签来源标识：dataset_rule/alarm/manual",
    )
    parser.add_argument(
        "--label_version",
        type=str,
        default="v1",
        help="标签版本号",
    )
    parser.add_argument(
        "--device_id",
        type=str,
        default="cwru_rig_1",
        help="设备 ID（用于跨设备扩展）",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_csv = Path(args.output_csv)

    if args.dataset == "cwru":
        build_master_labels_from_cwru(
            base_dir=base_dir,
            output_csv=output_csv,
            label_source=args.label_source,
            label_version=args.label_version,
            device_id=args.device_id,
        )


if __name__ == "__main__":
    main()
