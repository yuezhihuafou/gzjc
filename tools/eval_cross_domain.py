#!/usr/bin/env python3
"""
Cross-domain evaluation: XJTU-trained risk model -> CWRU dataset.

Outputs:
- AUC / PR-AUC / Acc / confusion matrix
- JSON metrics file
- Optional prediction CSV
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from dl.model import build_backbone


class BinaryClassificationHead(nn.Module):
    def __init__(self, in_features: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


class NpyRiskDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
    ):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.channel_mean = channel_mean.reshape(2, 1).astype(np.float32)
        self.channel_std = channel_std.reshape(2, 1).astype(np.float32)

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.signals[idx]
        y = self.labels[idx]
        x = (x - self.channel_mean) / (self.channel_std + 1e-8)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def _load_cwru_arrays(base_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    signals_path = base_dir / "signals.npy"
    labels_path = base_dir / "labels.npy"
    if not signals_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"signals/labels 不存在: {signals_path}, {labels_path}")

    signals = np.load(signals_path)
    labels = np.load(labels_path).reshape(-1)

    if signals.ndim == 2:
        signals = np.stack([signals, signals], axis=1)
    elif signals.ndim == 3 and signals.shape[1] == 1:
        signals = np.concatenate([signals, signals], axis=1)

    if signals.ndim != 3 or signals.shape[1] != 2:
        raise ValueError(f"signals 期望形状 (N,2,L)，实际: {signals.shape}")
    if labels.shape[0] != signals.shape[0]:
        raise ValueError(f"signals/labels 数量不一致: {signals.shape[0]} vs {labels.shape[0]}")

    # Binary align: Normal=0, Fault(others)=1
    labels_bin = (labels.astype(np.int64) != 0).astype(np.float32)
    return signals, labels_bin


def _resolve_threshold(checkpoint_dir: Path, threshold_arg: float) -> float:
    if threshold_arg is not None:
        return float(threshold_arg)
    th_path = checkpoint_dir.parent / "outputs" / "risk_threshold.txt"
    if th_path.exists():
        try:
            return float(th_path.read_text(encoding="utf-8").strip())
        except Exception:
            return 0.5
    return 0.5


def _resolve_model_hparams(
    checkpoint_dir: Path,
    model_scale_arg: str,
    embedding_dim_arg: int,
    head_dropout_arg: float,
) -> Tuple[str, int, float]:
    if model_scale_arg and embedding_dim_arg:
        return model_scale_arg, int(embedding_dim_arg), float(head_dropout_arg or 0.0)

    cfg_path = checkpoint_dir.parent / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            model_scale = model_scale_arg or cfg.get("model_scale", "base")
            embedding_dim = int(embedding_dim_arg or cfg.get("embedding_dim") or 0)
            if embedding_dim <= 0:
                if model_scale == "xlarge":
                    embedding_dim = 1024
                elif model_scale == "large":
                    embedding_dim = 768
                else:
                    embedding_dim = 512
            head_dropout = float(
                head_dropout_arg if head_dropout_arg is not None else cfg.get("head_dropout", 0.0)
            )
            return model_scale, embedding_dim, head_dropout
        except Exception:
            pass

    model_scale = model_scale_arg or "base"
    if embedding_dim_arg:
        embedding_dim = int(embedding_dim_arg)
    elif model_scale == "xlarge":
        embedding_dim = 1024
    elif model_scale == "large":
        embedding_dim = 768
    else:
        embedding_dim = 512
    head_dropout = float(head_dropout_arg or 0.0)
    return model_scale, embedding_dim, head_dropout


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-domain eval: XJTU risk checkpoint on CWRU")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="目录下应有 backbone.pth 和 risk_head.pth")
    parser.add_argument("--cwru_dir", type=str, default="datasets/cwru/cwru_processed_risk", help="CWRU npy 目录")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"], help="默认自动选择")
    parser.add_argument("--threshold", type=float, default=None, help="风险阈值，不填则尝试读取 risk_threshold.txt")
    parser.add_argument("--max_samples", type=int, default=None, help="仅评估前 N 个样本")
    parser.add_argument("--model_scale", type=str, default=None, choices=[None, "base", "large", "xlarge"])
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--head_dropout", type=float, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--pred_csv", type=str, default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    backbone_path = ckpt_dir / "backbone.pth"
    head_path = ckpt_dir / "risk_head.pth"
    if not backbone_path.exists() or not head_path.exists():
        raise FileNotFoundError(f"缺少 checkpoint 文件: {backbone_path}, {head_path}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_scale, embedding_dim, head_dropout = _resolve_model_hparams(
        checkpoint_dir=ckpt_dir,
        model_scale_arg=args.model_scale,
        embedding_dim_arg=args.embedding_dim,
        head_dropout_arg=args.head_dropout,
    )
    threshold = _resolve_threshold(ckpt_dir, args.threshold)

    print(f"设备: {device}")
    print(f"模型: scale={model_scale}, embedding_dim={embedding_dim}, head_dropout={head_dropout}")
    print(f"阈值: {threshold:.4f}")

    signals, labels = _load_cwru_arrays(Path(args.cwru_dir))
    if args.max_samples and args.max_samples > 0:
        n = min(args.max_samples, signals.shape[0])
        signals = signals[:n]
        labels = labels[:n]
    print(f"CWRU 样本数: {signals.shape[0]}")

    channel_mean = signals.mean(axis=(0, 2))
    channel_std = signals.std(axis=(0, 2))
    dataset = NpyRiskDataset(signals, labels, channel_mean, channel_std)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    backbone = build_backbone(in_channels=2, embedding_dim=embedding_dim, model_scale=model_scale).to(device)
    head = BinaryClassificationHead(in_features=embedding_dim, dropout=head_dropout).to(device)
    backbone.load_state_dict(torch.load(backbone_path, map_location="cpu", weights_only=True))
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    backbone.eval()
    head.eval()

    probs_all = []
    targets_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats = backbone(x)
            logits = head(feats)
            probs = torch.sigmoid(logits)
            probs_all.append(probs.cpu().numpy())
            targets_all.append(y.cpu().numpy())

    probs = np.concatenate(probs_all).reshape(-1)
    targets = np.concatenate(targets_all).reshape(-1)
    preds = (probs > threshold).astype(np.float32)

    auc = float(roc_auc_score(targets, probs)) if np.unique(targets).size >= 2 else float("nan")
    pr_auc = float(average_precision_score(targets, probs)) if np.unique(targets).size >= 2 else float("nan")
    acc = float(accuracy_score(targets, preds))
    cm = confusion_matrix(targets.astype(np.int64), preds.astype(np.int64), labels=[0, 1]).tolist()

    metrics: Dict = {
        "cwru_dir": str(args.cwru_dir),
        "checkpoint_dir": str(ckpt_dir),
        "n_samples": int(targets.shape[0]),
        "threshold": float(threshold),
        "accuracy": acc,
        "auc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix_2x2": cm,
        "model_scale": model_scale,
        "embedding_dim": int(embedding_dim),
        "head_dropout": float(head_dropout),
    }

    print(
        f"Cross-CWRU | Acc: {acc:.4f} | AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | "
        f"CM: {cm}"
    )

    out_json = Path(args.output_json) if args.output_json else (ckpt_dir.parent / "outputs" / "cross_cwru_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标已写入: {out_json}")

    if args.pred_csv:
        pred_csv = Path(args.pred_csv)
        pred_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["target", "pred_prob", "pred_label"])
            for t, p, pl in zip(targets.tolist(), probs.tolist(), preds.tolist()):
                writer.writerow([int(t), float(p), int(pl)])
        print(f"预测已写入: {pred_csv}")


if __name__ == "__main__":
    main()
