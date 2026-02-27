#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端完整性校验（MC .f/.json -> output_json -> cache_npz）

检查层级（可按需启用/跳过）：
1) .f 是否都有 sidecar .json
2) 每个 .f 是否产出 output_json（API 输出 JSON-first）
3) output_json 是否可用（schema/长度 3000/finite）
4) cache_npz 是否齐全（通过 index.jsonl 对账）以及 NPZ 字段一致性

输出：
- datasets/sound_api/logs/integrity_report.json（默认）
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 确保可直接以脚本方式运行（把项目根目录加入 sys.path）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.sound_api.convert_sound_api import (
    parse_bearing_id_from_filename,
    parse_t_from_filename,
)


def safe_read_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def validate_output_json_schema(obj: Dict) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_a_dict"
    if "data" not in obj or "metadata" not in obj:
        return False, "missing_data_or_metadata"
    data = obj.get("data", {})
    if not all(k in data for k in ("frequency", "volume", "density")):
        return False, "missing_frequency_volume_density"
    try:
        f = np.asarray(data["frequency"], dtype=np.float32)
        v = np.asarray(data["volume"], dtype=np.float32)
        d = np.asarray(data["density"], dtype=np.float32)
    except Exception:
        return False, "data_not_numeric"
    if not (len(f) == len(v) == len(d) == 3000):
        return False, f"length_not_3000: f={len(f)},v={len(v)},d={len(d)}"
    if not (np.isfinite(f).all() and np.isfinite(v).all() and np.isfinite(d).all()):
        return False, "non_finite_values"
    return True, "ok"


def read_index_jsonl(index_path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not index_path.exists():
        return records
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def validate_npz_against_record(npz_path: Path, record: Dict) -> Tuple[bool, str]:
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return False, f"npz_load_error: {e}"

    # 必须字段
    for key in ("x", "bearing_id", "t", "source_path"):
        if key not in data:
            return False, f"missing_npz_key: {key}"

    x = data["x"]
    if not (hasattr(x, "shape") and tuple(x.shape) == (2, 3000)):
        return False, f"bad_x_shape: {getattr(x,'shape',None)}"

    try:
        t_npz = int(data["t"])
    except Exception:
        return False, "bad_t_type"

    if "t" in record and record["t"] is not None and int(record["t"]) != t_npz:
        return False, f"t_mismatch: record={record['t']} npz={t_npz}"

    bearing_npz = str(data["bearing_id"])
    if "bearing_id" in record and str(record["bearing_id"]) != bearing_npz:
        return False, f"bearing_id_mismatch: record={record['bearing_id']} npz={bearing_npz}"

    # orig_t（可选）
    if record.get("orig_t") is not None:
        if "orig_t" not in data:
            return False, "missing_orig_t_in_npz"
        try:
            orig_npz = int(data["orig_t"])
        except Exception:
            return False, "bad_orig_t_type"
        if int(record["orig_t"]) != orig_npz:
            return False, f"orig_t_mismatch: record={record['orig_t']} npz={orig_npz}"

    # source_path 对账：只比对文件名
    if record.get("path"):
        try:
            src_name = Path(str(data["source_path"])).name
            if src_name != str(record["path"]):
                return False, f"source_path_name_mismatch: record={record['path']} npz={src_name}"
        except Exception:
            return False, "bad_source_path"

    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 MC->API->NPZ 全链路完整性")
    parser.add_argument("--mc_dir", type=str, required=True, help="包含 .f 与 sidecar .json 的目录（递归扫描）")
    parser.add_argument("--output_root", type=str, default="datasets/sound_api", help="datasets/sound_api 根目录")
    parser.add_argument("--cache_dir", type=str, default="datasets/sound_api/cache_npz", help="NPZ cache 目录")
    parser.add_argument("--index_file", type=str, default=None, help="cache index.jsonl（默认：{cache_dir}/index.jsonl）")
    parser.add_argument("--report", type=str, default="datasets/sound_api/logs/integrity_report.json", help="输出报告路径")
    parser.add_argument("--sample_npz_check", type=int, default=50, help="最多抽查多少条 index 记录对应的 NPZ（默认 50）")
    args = parser.parse_args()

    mc_dir = Path(args.mc_dir)
    output_root = Path(args.output_root)
    cache_dir = Path(args.cache_dir)
    index_path = Path(args.index_file) if args.index_file else (cache_dir / "index.jsonl")

    f_files = sorted(mc_dir.rglob("*.f"))
    if not f_files:
        raise SystemExit(f"错误: 在 {mc_dir} 中未找到 .f 文件")

    missing_sidecar: List[str] = []
    missing_output_json: List[str] = []
    invalid_output_json: List[Dict] = []

    # 1) .f/.json 配对 + 2) output_json 存在性 + 3) output_json 结构校验
    for f_path in f_files:
        sidecar = f_path.with_suffix(".json")
        if not sidecar.exists():
            missing_sidecar.append(str(f_path))
            continue

        bearing_id = parse_bearing_id_from_filename(f_path.name) or f_path.parent.name
        stem = f_path.stem
        out_json = output_root / "output_json" / str(bearing_id) / f"{stem}.json"
        if not out_json.exists():
            missing_output_json.append(str(f_path))
            continue

        obj = safe_read_json(out_json)
        if obj is None:
            invalid_output_json.append({"path": str(out_json), "reason": "json_load_error"})
            continue

        ok, reason = validate_output_json_schema(obj)
        if not ok:
            invalid_output_json.append({"path": str(out_json), "reason": reason})

    # 4) cache_npz 对账（依赖 index.jsonl）
    index_records = read_index_jsonl(index_path)
    missing_npz: List[Dict] = []
    bad_npz: List[Dict] = []

    if index_records:
        # 抽样校验（避免读太多 npz）
        to_check = index_records[: max(0, int(args.sample_npz_check))]
        for rec in to_check:
            try:
                bearing_id = str(rec["bearing_id"])
                t = int(rec["t"])
            except Exception:
                continue
            npz_path = cache_dir / bearing_id / f"{t:06d}.npz"
            if not npz_path.exists():
                missing_npz.append({"record": rec, "npz_path": str(npz_path)})
                continue
            ok, reason = validate_npz_against_record(npz_path, rec)
            if not ok:
                bad_npz.append({"record": rec, "npz_path": str(npz_path), "reason": reason})

    report_obj = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mc_dir": str(mc_dir),
        "output_root": str(output_root),
        "cache_dir": str(cache_dir),
        "index_file": str(index_path),
        "counts": {
            "mc_files": len(f_files),
            "missing_sidecar": len(missing_sidecar),
            "missing_output_json": len(missing_output_json),
            "invalid_output_json": len(invalid_output_json),
            "index_records": len(index_records),
            "missing_npz_checked": len(missing_npz),
            "bad_npz_checked": len(bad_npz),
        },
        "missing_sidecar": missing_sidecar[:200],
        "missing_output_json": missing_output_json[:200],
        "invalid_output_json": invalid_output_json[:200],
        "missing_npz_checked": missing_npz[:200],
        "bad_npz_checked": bad_npz[:200],
        "notes": {
            "t_vs_orig_t": "训练用 t 为稳定重编号，仅用于序列建模；orig_t 保留原始采集顺序，用于追溯与人工校验。",
            "index_required_for_full_cache_check": "如需完整 cache 对账，请先运行 build_sound_api_cache.py 生成 cache_npz/index.jsonl。",
        },
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("MC->API->NPZ 完整性校验完成")
    print("=" * 80)
    for k, v in report_obj["counts"].items():
        print(f"{k}: {v}")
    print(f"报告: {report_path}")
    if not index_records:
        print("提示：未找到 index.jsonl，NPZ 对账已跳过。")


if __name__ == "__main__":
    main()

