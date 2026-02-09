#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 MC->API 转换完成度和数据质量
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

MC_DIR = Path("D:/guzhangjiance/datasets/xjtu/output_xjtu_mc/xjtu")
JSON_DIR = Path("D:/guzhangjiance/datasets/sound_api/output_json")

def main():
    print("=" * 70)
    print("Conversion Status Report")
    print("=" * 70)
    
    # 1. Count .f files
    print("\n[1] Source File Count")
    print("-" * 70)
    f_files = list(MC_DIR.rglob("*.f"))
    f_stems = set(f.stem for f in f_files)
    print(f"Total .f files: {len(f_files):,}")
    print(f"Unique stems: {len(f_stems):,}")
    
    # 2. Count JSON files
    print("\n[2] Output JSON Count")
    print("-" * 70)
    json_files = list(JSON_DIR.rglob("*.json"))
    json_stems = set(j.stem for j in json_files)
    print(f"Total JSON files: {len(json_files):,}")
    print(f"Unique stems: {len(json_stems):,}")
    
    # 3. Completion rate
    print("\n[3] Completion Rate")
    print("-" * 70)
    completion = len(json_files) / len(f_files) * 100 if f_files else 0
    print(f"Completion: {len(json_files):,} / {len(f_files):,} = {completion:.2f}%")
    
    # 4. Check duplicates
    print("\n[4] Duplicate Check")
    print("-" * 70)
    json_stem_counts = Counter(j.stem for j in json_files)
    duplicates = [(stem, cnt) for stem, cnt in json_stem_counts.items() if cnt > 1]
    if duplicates:
        print(f"[WARNING] Duplicate JSON count: {len(duplicates)}")
        print(f"  Examples: {duplicates[:5]}")
    else:
        print("[OK] No duplicate JSON files")
    
    # 5. Check missing
    print("\n[5] Missing Check")
    print("-" * 70)
    missing = f_stems - json_stems
    if missing:
        print(f"[PENDING] Missing .f count: {len(missing):,}")
        print(f"  Examples (first 10): {list(missing)[:10]}")
    else:
        print("[OK] No missing files - all .f converted")
    
    # 6. Check extra
    print("\n[6] Extra Check")
    print("-" * 70)
    extra = json_stems - f_stems
    if extra:
        print(f"[INFO] Extra JSON count: {len(extra)}")
        print(f"  Examples: {list(extra)[:5]}")
    else:
        print("[OK] No extra JSON files")
    
    # 7. Data Quality Check (sample)
    print("\n[7] Data Quality Check (sample 10 files)")
    print("-" * 70)
    sample_files = json_files[:10]
    quality_ok = 0
    quality_fail = 0
    issues = []
    
    for jf in sample_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check structure
            has_data = "data" in data and isinstance(data["data"], dict)
            has_meta = "metadata" in data and isinstance(data["metadata"], dict)
            
            if not has_data or not has_meta:
                quality_fail += 1
                issues.append(f"{jf.name}: missing data/metadata")
                continue
            
            # Check data lengths
            freq = data["data"].get("frequency", [])
            vol = data["data"].get("volume", [])
            den = data["data"].get("density", [])
            
            if len(freq) != 3000 or len(vol) != 3000 or len(den) != 3000:
                quality_fail += 1
                issues.append(f"{jf.name}: wrong length freq={len(freq)} vol={len(vol)} den={len(den)}")
                continue
            
            # Check numeric values
            vol_arr = np.array(vol, dtype=float)
            den_arr = np.array(den, dtype=float)
            freq_arr = np.array(freq, dtype=float)
            
            if not np.all(np.isfinite(vol_arr)):
                quality_fail += 1
                issues.append(f"{jf.name}: volume has non-finite values")
                continue
            if not np.all(vol_arr >= 0):
                quality_fail += 1
                issues.append(f"{jf.name}: volume has negative values")
                continue
            if not np.all(np.isfinite(den_arr)):
                quality_fail += 1
                issues.append(f"{jf.name}: density has non-finite values")
                continue
            if not np.all(np.isfinite(freq_arr)):
                quality_fail += 1
                issues.append(f"{jf.name}: frequency has non-finite values")
                continue
            
            # Check metadata
            meta = data["metadata"]
            required = ["bearing_id", "t", "orig_t", "source_path", "sampling_rate"]
            missing_meta = [k for k in required if k not in meta]
            if missing_meta:
                quality_fail += 1
                issues.append(f"{jf.name}: missing metadata {missing_meta}")
                continue
            
            quality_ok += 1
            
        except Exception as e:
            quality_fail += 1
            issues.append(f"{jf.name}: error {e}")
    
    if quality_ok == len(sample_files):
        print(f"[OK] All {quality_ok} samples passed quality check")
    else:
        print(f"[WARNING] {quality_ok}/{len(sample_files)} passed, {quality_fail} failed")
        for issue in issues[:5]:
            print(f"  - {issue}")
    
    # 8. Directory structure
    print("\n[8] Directory Structure")
    print("-" * 70)
    bearing_dirs = [d for d in JSON_DIR.iterdir() if d.is_dir()]
    print(f"bearing_id directories: {len(bearing_dirs):,}")
    if bearing_dirs:
        sample_dir = bearing_dirs[0]
        sample_count = len(list(sample_dir.glob("*.json")))
        print(f"  Example: {sample_dir.name}/ has {sample_count} JSON files")
    
    # 9. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if len(missing) == 0:
        print("[COMPLETE] All .f files have been converted to JSON")
    else:
        print(f"[INCOMPLETE] {len(missing):,} files pending conversion ({100-completion:.2f}%)")
    
    if duplicates:
        print(f"[WARNING] {len(duplicates)} duplicate JSON files detected")
    else:
        print("[OK] No duplicates")
    
    if quality_ok == len(sample_files):
        print("[OK] Sample data quality check passed")
        print("     -> Ready for training and review")
    else:
        print(f"[WARNING] {quality_fail} samples failed quality check")
    
    print()
    if len(missing) > 0:
        print("Next step: Run conversion with --resume to complete remaining files")
        print("  python tools/sound_api/convert_mc_to_api_json.py \\")
        print("    --mc_dir ... --output_root ... --auth-file auth_example.json \\")
        print("    --workers 32 --qps 15 --resume")

if __name__ == "__main__":
    main()
