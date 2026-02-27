import numpy as np
from collections import Counter
from pathlib import Path

def load_labels_from_npz(npz_dir):
    """从NPZ文件加载fault_label"""
    labels = []
    p = Path(npz_dir)
    npz_files = [f for f in sorted(p.rglob("*.npz")) if not f.name.endswith(".tmp.npz")]
    for npz_file in npz_files:
        try:
            with np.load(npz_file) as data:
                if 'fault_label' in data:
                    labels.append(int(data['fault_label']))
        except (EOFError, OSError, ValueError):
            pass
    return labels

def check_label_distribution(labels):
    """检查标签分布情况"""
    label_counts = Counter(labels)
    total = len(labels)

    print("="*60)
    print(f"样本总数: {total}")
    print("标签分布:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total) * 100
        print(f"  标签 {label}: {count} ({percentage:.1f}%)")
        if percentage > 90:
            print(f"    ⚠️  标签'{label}' 占比过高 (>90%)")
        if label == 1 and count < 50:
            print(f"    ⚠️  正例样本过少: {count}")

    if len(label_counts) == 2:
        values = sorted(label_counts.values())
        ratio = values[1] / values[0] if values[0] > 0 else float('inf')
        print(f"\n类别不平衡比: {ratio:.2f}")
    print("="*60)

if __name__ == '__main__':
    # 直接分析 index_small_npz_subset 目录
    npz_dir = r'D:\guzhangjiance\datasets\sound_api\index_small_npz_subset'
    labels = load_labels_from_npz(npz_dir)
    check_label_distribution(labels)