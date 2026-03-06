"""
训练脚本 - 支持多种任务和数据源

支持的任务：
- hi: 健康指数回归
- risk: 风险预测二分类
- arcface: ArcFace 分类（需要标签）

支持的数据源：
- cwru: CWRU 处理后的 npy 数据
- sound: 本地 xlsx 文件
- sound_api: 从 API 获取
- sound_api_cache: 从 NPZ 缓存加载（新增）
"""
import os
import sys
import argparse
import csv
import json
import random
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dl.data_loader import (
    get_dataloaders,
    get_cwru_multi_dataloaders,
    get_sound_api_cache_dataloaders,
)
from dl.sound_data_loader import get_sound_dataloaders
from dl.sound_api_data_loader import get_sound_api_dataloaders
from dl.model import build_backbone
from dl.loss import ArcMarginProduct

# 类别 id → 可读名（与 CWRU/项目约定一致），评估/推理时输出用
CLASS_ID_TO_NAME = {0: "Normal", 1: "B", 2: "IR", 3: "OR"}


class RegressionHead(nn.Module):
    """回归头：输出单个标量"""
    def __init__(self, in_features: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)  # (B,)


class BinaryClassificationHead(nn.Module):
    """二分类头：输出单个 logit"""
    def __init__(self, in_features: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)  # (B,)


class MultiTaskHead(nn.Module):
    """共享骨干的多头输出：risk/fault/condition"""

    def __init__(
        self,
        in_features: int = 512,
        n_fault_classes: int = 4,
        n_condition_classes: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.risk_fc = nn.Linear(in_features, 1)
        self.fault_fc = nn.Linear(in_features, n_fault_classes)
        self.condition_fc = nn.Linear(in_features, n_condition_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.dropout(x)
        return {
            "risk": self.risk_fc(z).squeeze(-1),
            "fault": self.fault_fc(z),
            "condition": self.condition_fc(z),
        }


def train_one_epoch_arcface(
    backbone: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional["torch.amp.GradScaler"] = None,
) -> Tuple[float, float]:
    """ArcFace 训练（可选 AMP）"""
    backbone.train()
    head.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(dataloader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                features = backbone(x)
                logits = head(features, y)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            features = backbone(x)
            logits = head(features, y)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_one_epoch_regression(
    backbone: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional["torch.amp.GradScaler"] = None,
) -> Tuple[float, float]:
    """回归训练（HI，可选 AMP）"""
    backbone.train()
    head.train()

    running_loss = 0.0
    total = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                features = backbone(x)
                pred = head(features)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            features = backbone(x)
            pred = head(features)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        total += x.size(0)

    epoch_loss = running_loss / total
    return epoch_loss, 0.0  # 回归任务没有准确率


def train_one_epoch_binary(
    backbone: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional["torch.amp.GradScaler"] = None,
) -> Tuple[float, float]:
    """二分类训练（Risk，可选 AMP）"""
    backbone.train()
    head.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad()
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                features = backbone(x)
                logits = head(features)
                raw_loss = criterion(logits, y)
                if isinstance(raw_loss, torch.Tensor) and raw_loss.ndim > 0:
                    if meta:
                        sw = torch.tensor(
                            [float(m.get("sample_weight", 1.0)) for m in meta],
                            dtype=raw_loss.dtype,
                            device=raw_loss.device,
                        )
                        loss = (raw_loss * sw).mean()
                    else:
                        loss = raw_loss.mean()
                else:
                    loss = raw_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            features = backbone(x)
            logits = head(features)
            raw_loss = criterion(logits, y)
            if isinstance(raw_loss, torch.Tensor) and raw_loss.ndim > 0:
                if meta:
                    sw = torch.tensor(
                        [float(m.get("sample_weight", 1.0)) for m in meta],
                        dtype=raw_loss.dtype,
                        device=raw_loss.device,
                    )
                    loss = (raw_loss * sw).mean()
                else:
                    loss = raw_loss.mean()
            else:
                loss = raw_loss
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += x.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_one_epoch_multi(
    backbone: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fns: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional["torch.amp.GradScaler"] = None,
) -> Tuple[float, Dict[str, float]]:
    backbone.train()
    head.train()

    running_total = 0.0
    running_risk = 0.0
    running_fault = 0.0
    running_condition = 0.0
    n = 0

    risk_correct = 0
    fault_correct = 0
    condition_correct = 0

    for x, targets, meta in tqdm(dataloader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y_risk = targets["risk"].to(device, non_blocking=True).float()
        y_fault = targets["fault"].to(device, non_blocking=True).long()
        y_cond = targets["condition"].to(device, non_blocking=True).long()
        bs = x.size(0)

        optimizer.zero_grad()
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                feats = backbone(x)
                out = head(feats)
                raw_risk = loss_fns["risk"](out["risk"], y_risk)
                if isinstance(raw_risk, torch.Tensor) and raw_risk.ndim > 0:
                    if meta:
                        sw = torch.tensor(
                            [float(m.get("sample_weight", 1.0)) for m in meta],
                            dtype=raw_risk.dtype,
                            device=raw_risk.device,
                        )
                        loss_risk = (raw_risk * sw).mean()
                    else:
                        loss_risk = raw_risk.mean()
                else:
                    loss_risk = raw_risk
                loss_fault = loss_fns["fault"](out["fault"], y_fault)
                loss_cond = loss_fns["condition"](out["condition"], y_cond)
                loss = (
                    float(loss_weights["risk"]) * loss_risk
                    + float(loss_weights["fault"]) * loss_fault
                    + float(loss_weights["condition"]) * loss_cond
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            feats = backbone(x)
            out = head(feats)
            raw_risk = loss_fns["risk"](out["risk"], y_risk)
            if isinstance(raw_risk, torch.Tensor) and raw_risk.ndim > 0:
                if meta:
                    sw = torch.tensor(
                        [float(m.get("sample_weight", 1.0)) for m in meta],
                        dtype=raw_risk.dtype,
                        device=raw_risk.device,
                    )
                    loss_risk = (raw_risk * sw).mean()
                else:
                    loss_risk = raw_risk.mean()
            else:
                loss_risk = raw_risk
            loss_fault = loss_fns["fault"](out["fault"], y_fault)
            loss_cond = loss_fns["condition"](out["condition"], y_cond)
            loss = (
                float(loss_weights["risk"]) * loss_risk
                + float(loss_weights["fault"]) * loss_fault
                + float(loss_weights["condition"]) * loss_cond
            )
            loss.backward()
            optimizer.step()

        running_total += float(loss.item()) * bs
        running_risk += float(loss_risk.item()) * bs
        running_fault += float(loss_fault.item()) * bs
        running_condition += float(loss_cond.item()) * bs
        n += bs

        risk_pred = (torch.sigmoid(out["risk"]) > 0.5).long()
        risk_correct += int((risk_pred == y_risk.long()).sum().item())
        fault_pred = torch.argmax(out["fault"], dim=1)
        fault_correct += int((fault_pred == y_fault).sum().item())
        cond_pred = torch.argmax(out["condition"], dim=1)
        condition_correct += int((cond_pred == y_cond).sum().item())

    if n == 0:
        return 0.0, {"risk_acc": 0.0, "fault_acc": 0.0, "condition_acc": 0.0}
    metrics = {
        "risk_acc": risk_correct / n,
        "fault_acc": fault_correct / n,
        "condition_acc": condition_correct / n,
        "loss_risk": running_risk / n,
        "loss_fault": running_fault / n,
        "loss_condition": running_condition / n,
    }
    return running_total / n, metrics


def evaluate_arcface(
    backbone: nn.Module,
    head: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """ArcFace 评估"""
    backbone.eval()
    head.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    torch.set_grad_enabled(False)
    for x, y in tqdm(dataloader, desc="Val", leave=False):
        x = x.to(device)
        y = y.to(device)

        features = backbone(x)
        logits = head(features, y)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    torch.set_grad_enabled(True)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_regression(
    backbone: nn.Module,
    head: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, List]:
    """回归评估（HI）- 返回预测值和真实值用于绘图"""
    backbone.eval()
    head.eval()

    running_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []
    all_meta = []

    torch.set_grad_enabled(False)
    for batch in tqdm(dataloader, desc="Val", leave=False):
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        
        x = x.to(device)
        y = y.to(device)

        features = backbone(x)
        pred = head(features)
        loss = criterion(pred, y)

        running_loss += loss.item() * x.size(0)
        total += x.size(0)
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        if meta:
            all_meta.extend(meta)
    torch.set_grad_enabled(True)

    epoch_loss = running_loss / total
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # 计算 MAE 和 RMSE
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    return epoch_loss, mae, preds, targets, all_meta


def evaluate_binary(
    backbone: nn.Module,
    head: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, float, np.ndarray, np.ndarray, List]:
    """二分类评估（Risk）- 返回预测值和真实值用于计算 AUC"""
    backbone.eval()
    head.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_meta = []

    torch.set_grad_enabled(False)
    for batch in tqdm(dataloader, desc="Val", leave=False):
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        
        x = x.to(device)
        y = y.to(device).float()

        features = backbone(x)
        logits = head(features)
        raw_loss = criterion(logits, y)
        if isinstance(raw_loss, torch.Tensor) and raw_loss.ndim > 0:
            if meta:
                sw = torch.tensor(
                    [float(m.get("sample_weight", 1.0)) for m in meta],
                    dtype=raw_loss.dtype,
                    device=raw_loss.device,
                )
                loss = (raw_loss * sw).mean()
            else:
                loss = raw_loss.mean()
        else:
            loss = raw_loss

        running_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        correct += (preds == y).sum().item()
        total += x.size(0)
        
        all_preds.append(probs.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        if meta:
            all_meta.extend(meta)
    torch.set_grad_enabled(True)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # 计算 AUC（需要 sklearn）
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(targets, preds)
        pr_auc = average_precision_score(targets, preds)
    except ImportError:
        auc = 0.0
        pr_auc = 0.0
    
    return epoch_loss, epoch_acc, preds, targets, all_meta, auc, pr_auc


def evaluate_multi(
    backbone: nn.Module,
    head: nn.Module,
    loss_fns: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    dataloader: DataLoader,
    device: torch.device,
    risk_threshold: float = 0.5,
) -> Dict[str, Any]:
    backbone.eval()
    head.eval()

    running_total = 0.0
    running_risk = 0.0
    running_fault = 0.0
    running_condition = 0.0
    n = 0

    risk_probs_all: List[np.ndarray] = []
    risk_targets_all: List[np.ndarray] = []
    fault_logits_all: List[np.ndarray] = []
    fault_targets_all: List[np.ndarray] = []
    cond_logits_all: List[np.ndarray] = []
    cond_targets_all: List[np.ndarray] = []
    meta_all: List[Dict[str, Any]] = []

    with torch.no_grad():
        for x, targets, meta in tqdm(dataloader, desc="Val", leave=False):
            x = x.to(device)
            y_risk = targets["risk"].to(device).float()
            y_fault = targets["fault"].to(device).long()
            y_cond = targets["condition"].to(device).long()
            bs = x.size(0)

            feats = backbone(x)
            out = head(feats)

            raw_risk = loss_fns["risk"](out["risk"], y_risk)
            if isinstance(raw_risk, torch.Tensor) and raw_risk.ndim > 0:
                if meta:
                    sw = torch.tensor(
                        [float(m.get("sample_weight", 1.0)) for m in meta],
                        dtype=raw_risk.dtype,
                        device=raw_risk.device,
                    )
                    loss_risk = (raw_risk * sw).mean()
                else:
                    loss_risk = raw_risk.mean()
            else:
                loss_risk = raw_risk
            loss_fault = loss_fns["fault"](out["fault"], y_fault)
            loss_cond = loss_fns["condition"](out["condition"], y_cond)
            loss = (
                float(loss_weights["risk"]) * loss_risk
                + float(loss_weights["fault"]) * loss_fault
                + float(loss_weights["condition"]) * loss_cond
            )

            running_total += float(loss.item()) * bs
            running_risk += float(loss_risk.item()) * bs
            running_fault += float(loss_fault.item()) * bs
            running_condition += float(loss_cond.item()) * bs
            n += bs

            risk_probs_all.append(torch.sigmoid(out["risk"]).cpu().numpy())
            risk_targets_all.append(y_risk.cpu().numpy())
            fault_logits_all.append(out["fault"].cpu().numpy())
            fault_targets_all.append(y_fault.cpu().numpy())
            cond_logits_all.append(out["condition"].cpu().numpy())
            cond_targets_all.append(y_cond.cpu().numpy())
            meta_all.extend(list(meta))

    if n == 0:
        return {
            "loss": 0.0,
            "loss_risk": 0.0,
            "loss_fault": 0.0,
            "loss_condition": 0.0,
            "risk_probs": np.array([], dtype=np.float32),
            "risk_targets": np.array([], dtype=np.float32),
            "fault_logits": np.zeros((0, 4), dtype=np.float32),
            "fault_targets": np.array([], dtype=np.int64),
            "condition_logits": np.zeros((0, 1), dtype=np.float32),
            "condition_targets": np.array([], dtype=np.int64),
            "meta": [],
        }

    return {
        "loss": running_total / n,
        "loss_risk": running_risk / n,
        "loss_fault": running_fault / n,
        "loss_condition": running_condition / n,
        "risk_probs": np.concatenate(risk_probs_all),
        "risk_targets": np.concatenate(risk_targets_all),
        "fault_logits": np.concatenate(fault_logits_all),
        "fault_targets": np.concatenate(fault_targets_all),
        "condition_logits": np.concatenate(cond_logits_all),
        "condition_targets": np.concatenate(cond_targets_all),
        "meta": meta_all,
        "risk_threshold": float(risk_threshold),
    }


def compute_binary_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pred_prob = np.asarray(preds).reshape(-1)
    tgt = np.asarray(targets).reshape(-1).astype(np.int64)
    pred_label = (pred_prob > threshold).astype(np.int64)
    out["acc"] = float(np.mean(pred_label == tgt))
    try:
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            balanced_accuracy_score,
            f1_score,
            recall_score,
        )
        if np.unique(tgt).size >= 2:
            out["auc"] = float(roc_auc_score(tgt, pred_prob))
            out["pr_auc"] = float(average_precision_score(tgt, pred_prob))
            out["balanced_acc"] = float(balanced_accuracy_score(tgt, pred_label))
            out["f1"] = float(f1_score(tgt, pred_label, zero_division=0))
            out["normal_recall"] = float(recall_score(tgt, pred_label, pos_label=0, zero_division=0))
        else:
            out["auc"] = 0.0
            out["pr_auc"] = 0.0
            out["balanced_acc"] = 0.0
            out["f1"] = 0.0
            out["normal_recall"] = 0.0
    except Exception:
        out["auc"] = 0.0
        out["pr_auc"] = 0.0
        out["balanced_acc"] = 0.0
        out["f1"] = 0.0
        out["normal_recall"] = 0.0
    return out


def compute_multiclass_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    y_true = np.asarray(targets).reshape(-1).astype(np.int64)
    logits_arr = np.asarray(logits)
    if logits_arr.size == 0 or y_true.size == 0:
        return {
            "acc": 0.0,
            "macro_f1": 0.0,
            "labels": [],
            "confusion_matrix": [],
        }
    if logits_arr.ndim == 1:
        logits_arr = logits_arr.reshape(-1, 1)
    n = min(int(logits_arr.shape[0]), int(y_true.shape[0]))
    if n <= 0:
        return {
            "acc": 0.0,
            "macro_f1": 0.0,
            "labels": [],
            "confusion_matrix": [],
        }
    logits_arr = logits_arr[:n]
    y_true = y_true[:n]
    y_pred = np.argmax(logits_arr, axis=1).astype(np.int64)
    out["acc"] = float(np.mean(y_pred == y_true))
    try:
        from sklearn.metrics import f1_score, confusion_matrix

        out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist()) if y_true.size > 0 else [0]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        out["labels"] = labels
        out["confusion_matrix"] = cm.tolist()
    except Exception:
        out["macro_f1"] = 0.0
        out["labels"] = []
        out["confusion_matrix"] = []
    return out


def save_multiclass_metrics_csv(
    csv_path: Path,
    metrics: Dict[str, Any],
) -> None:
    rows: Dict[str, Any] = {
        "acc": metrics.get("acc", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "labels": json.dumps(metrics.get("labels", []), ensure_ascii=False),
        "confusion_matrix": json.dumps(metrics.get("confusion_matrix", []), ensure_ascii=False),
    }
    save_metrics_csv(csv_path, rows)


def apply_risk_direction(probs: np.ndarray, direction: str) -> np.ndarray:
    """方向校准：normal=原分数，inverted=1-p。"""
    p = np.asarray(probs).reshape(-1).astype(np.float64)
    if str(direction).lower() in ("inverted", "inverse", "neg", "reverse", "-1"):
        return 1.0 - p
    return p


def calibrate_risk_postprocess(
    targets: np.ndarray,
    probs: np.ndarray,
    default_threshold: float = 0.5,
    allow_invert: bool = True,
    tune_threshold: bool = True,
    metric: str = "balanced_acc",
) -> Dict[str, float]:
    """
    在验证集联合校准 score direction + threshold。
    metric: balanced_acc / f1
    """
    y = np.asarray(targets).reshape(-1).astype(np.int64)
    p = np.asarray(probs).reshape(-1).astype(np.float64)

    if y.size == 0:
        return {"direction": "normal", "threshold": float(default_threshold), "score": 0.0}

    directions = ["normal", "inverted"] if allow_invert else ["normal"]
    if tune_threshold:
        threshold_grid = np.linspace(0.01, 0.99, 99)
    else:
        threshold_grid = np.array([float(default_threshold)], dtype=np.float64)

    best = {
        "direction": "normal",
        "threshold": float(default_threshold),
        "score": -1.0,
        "auc": -1.0,
    }

    for direction in directions:
        s = apply_risk_direction(p, direction)
        dir_auc = float(compute_binary_metrics(s, y, threshold=0.5).get("auc", 0.0))
        for th in threshold_grid:
            m = compute_binary_metrics(s, y, threshold=float(th))
            score = float(m.get("f1", 0.0)) if metric == "f1" else float(m.get("balanced_acc", 0.0))
            if (score > best["score"]) or (
                abs(score - best["score"]) <= 1e-12 and dir_auc > best["auc"]
            ):
                best = {
                    "direction": direction,
                    "threshold": float(th),
                    "score": score,
                    "auc": dir_auc,
                }

    return best


def calibrate_binary_threshold(
    targets: np.ndarray,
    probs: np.ndarray,
    default_threshold: float = 0.5,
) -> float:
    """在验证集上按 F1 最大化选择阈值。"""
    try:
        from sklearn.metrics import precision_recall_curve
        p, r, thresholds = precision_recall_curve(targets, probs)
        if thresholds is None or len(thresholds) == 0:
            return default_threshold
        f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-8, None)
        best_idx = int(np.nanargmax(f1))
        th = float(thresholds[best_idx])
        return max(0.01, min(0.99, th))
    except Exception:
        return default_threshold


def plot_hi_predictions(preds: np.ndarray, targets: np.ndarray, meta: List, output_dir: str):
    """绘制 HI 预测曲线（按 bearing 分组）"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 按 bearing_id 分组
    bearing_groups = {}
    for i, m in enumerate(meta):
        bearing_id = m['bearing_id']
        if bearing_id not in bearing_groups:
            bearing_groups[bearing_id] = {'t': [], 'pred': [], 'target': []}
        bearing_groups[bearing_id]['t'].append(m['t'])
        bearing_groups[bearing_id]['pred'].append(preds[i])
        bearing_groups[bearing_id]['target'].append(targets[i])
    
    # 为每个 bearing 绘图
    for bearing_id, data in bearing_groups.items():
        # 按 t 排序
        indices = np.argsort(data['t'])
        t_sorted = np.array(data['t'])[indices]
        pred_sorted = np.array(data['pred'])[indices]
        target_sorted = np.array(data['target'])[indices]
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_sorted, target_sorted, 'o-', label='真实值', alpha=0.7)
        plt.plot(t_sorted, pred_sorted, 's-', label='预测值', alpha=0.7)
        plt.xlabel('时间 t')
        plt.ylabel('健康指数 HI')
        plt.title(f'Bearing {bearing_id} - 健康指数预测')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'hi_bearing_{bearing_id}.png', dpi=150)
        plt.close()


def plot_risk_predictions(preds: np.ndarray, targets: np.ndarray, meta: List, output_dir: str):
    """绘制风险预测曲线（按 bearing 分组）"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 按 bearing_id 分组
    bearing_groups = {}
    for i, m in enumerate(meta):
        bearing_id = m['bearing_id']
        if bearing_id not in bearing_groups:
            bearing_groups[bearing_id] = {'t': [], 'pred': [], 'target': []}
        bearing_groups[bearing_id]['t'].append(m['t'])
        bearing_groups[bearing_id]['pred'].append(preds[i])
        bearing_groups[bearing_id]['target'].append(targets[i])
    
    # 为每个 bearing 绘图
    for bearing_id, data in bearing_groups.items():
        # 按 t 排序
        indices = np.argsort(data['t'])
        t_sorted = np.array(data['t'])[indices]
        pred_sorted = np.array(data['pred'])[indices]
        target_sorted = np.array(data['target'])[indices]
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_sorted, target_sorted, 'o-', label='真实标签', alpha=0.7)
        plt.plot(t_sorted, pred_sorted, 's-', label='预测概率', alpha=0.7)
        plt.axhline(y=0.5, color='r', linestyle='--', label='阈值 0.5', alpha=0.5)
        plt.xlabel('时间 t')
        plt.ylabel('风险概率')
        plt.title(f'Bearing {bearing_id} - 风险预测')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'risk_bearing_{bearing_id}.png', dpi=150)
        plt.close()


def dump_split_files_for_cwru(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    split_dump_dir: str,
) -> None:
    """导出切分索引与工况统计，便于复现实验。"""
    output_dir = Path(split_dump_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _extract(dataset):
        indices = [int(i) for i in dataset.indices.tolist()]
        condition_counts: Dict[str, int] = {}
        cond_ids = getattr(dataset, "condition_ids", None)
        if cond_ids is not None:
            for idx in indices:
                cid = str(cond_ids[idx])
                condition_counts[cid] = condition_counts.get(cid, 0) + 1
        return indices, condition_counts

    train_indices, train_conditions = _extract(train_loader.dataset)
    val_indices, val_conditions = _extract(val_loader.dataset)
    test_indices, test_conditions = _extract(test_loader.dataset)

    (output_dir / "train_indices.json").write_text(
        json.dumps(train_indices, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "val_indices.json").write_text(
        json.dumps(val_indices, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "test_indices.json").write_text(
        json.dumps(test_indices, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "train_conditions.json").write_text(
        json.dumps(train_conditions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "val_conditions.json").write_text(
        json.dumps(val_conditions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "test_conditions.json").write_text(
        json.dumps(test_conditions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"切分文件已导出: {output_dir}")


def dump_split_files_for_sound_cache(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    split_dump_dir: str,
) -> None:
    """导出 sound_api_cache 的样本切分文件（含标签与路径），便于复现实验。"""
    output_dir = Path(split_dump_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _dump_samples(name: str, dataset) -> None:
        csv_path = output_dir / f"{name}_samples.csv"
        samples = getattr(dataset, "samples", [])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["bearing_id", "t", "T", "fault_label", "condition_id", "npz_path"])
            for s in samples:
                writer.writerow([
                    s.get("bearing_id", ""),
                    s.get("t", ""),
                    s.get("T", ""),
                    s.get("fault_label", ""),
                    s.get("condition_id", ""),
                    s.get("npz_path", ""),
                ])

    _dump_samples("train", train_loader.dataset)
    _dump_samples("val", val_loader.dataset)
    _dump_samples("test", test_loader.dataset)
    print(f"sound_api_cache 切分文件已导出: {output_dir}")


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def build_run_name(args: argparse.Namespace) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [ts, args.data_source, args.task]
    if args.task == "risk" and args.horizon is not None:
        parts.append(f"h{args.horizon}")
    if args.data_source == "sound_api_cache":
        idx_tag = "full"
        if args.index_path:
            idx_tag = "small" if "small" in Path(args.index_path).name.lower() else "custom"
        parts.append(f"idx_{idx_tag}")
    return _safe_name("_".join(parts))


def save_run_config(run_dir: Path, args: argparse.Namespace, device: torch.device) -> None:
    cfg = vars(args).copy()
    cfg["resolved_device"] = str(device)
    cfg["created_at"] = datetime.now().isoformat()
    (run_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_metrics_csv(metrics_path: Path, metrics: Dict[str, float]) -> None:
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def get_head_checkpoint_name(task: str) -> str:
    if task == "arcface":
        return "arcface_head.pth"
    if task == "multi":
        return "multi_head.pth"
    return f"{task}_head.pth"


def save_risk_predictions_csv(
    csv_path: Path,
    preds: np.ndarray,
    targets: np.ndarray,
    meta: List,
    threshold: float = 0.5,
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bearing_id", "t", "T", "target", "pred_prob", "pred_label", "npz_path"])
        for i in range(len(preds)):
            m = meta[i] if i < len(meta) else {}
            p = float(preds[i])
            writer.writerow([
                m.get("bearing_id", ""),
                m.get("t", ""),
                m.get("T", ""),
                float(targets[i]),
                p,
                int(p > threshold),
                m.get("npz_path", ""),
            ])


def save_risk_condition_metrics_csv(
    csv_path: Path,
    preds: np.ndarray,
    targets: np.ndarray,
    meta: List,
    threshold: float = 0.5,
) -> None:
    """按 condition_id 导出 risk 指标，便于跨工况分析。"""
    groups: Dict[str, List[int]] = {}
    for i in range(len(preds)):
        m = meta[i] if i < len(meta) else {}
        cid = str(m.get("condition_id", "unknown") or "unknown")
        groups.setdefault(cid, []).append(i)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition_id", "n_samples", "acc", "auc", "pr_auc", "balanced_acc", "f1", "normal_recall"])
        for cid in sorted(groups.keys()):
            idx = groups[cid]
            p = preds[idx]
            t = targets[idx]
            m = compute_binary_metrics(p, t, threshold=threshold)
            writer.writerow([
                cid,
                len(idx),
                m.get("acc", ""),
                m.get("auc", ""),
                m.get("pr_auc", ""),
                m.get("balanced_acc", ""),
                m.get("f1", ""),
                m.get("normal_recall", ""),
            ])


def save_risk_domain_metrics_csv(
    csv_path: Path,
    rows: List[Dict[str, float]],
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain_id", "n_samples", "acc", "auc", "pr_auc", "balanced_acc", "f1", "normal_recall"])
        for r in rows:
            writer.writerow([
                r.get("domain_id", ""),
                r.get("n_samples", 0),
                r.get("acc", ""),
                r.get("auc", ""),
                r.get("pr_auc", ""),
                r.get("balanced_acc", ""),
                r.get("f1", ""),
                r.get("normal_recall", ""),
            ])


def estimate_pos_weight_from_loader(train_loader: DataLoader) -> float:
    neg = 0
    pos = 0
    for batch in train_loader:
        if len(batch) == 3:
            _, y, _ = batch
        else:
            _, y = batch
        if isinstance(y, dict):
            y_tensor = y.get("risk")
            if y_tensor is None:
                continue
            y_np = y_tensor.detach().cpu().numpy().astype(np.int64).reshape(-1)
        else:
            y_np = y.detach().cpu().numpy().astype(np.int64).reshape(-1)
        pos += int((y_np > 0).sum())
        neg += int((y_np == 0).sum())
    if pos <= 0:
        return 1.0
    return float(max(1.0, neg / max(1, pos)))


def semantic_check_report(
    data_source: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Dict:
    def _summarize_loader(name: str, loader: DataLoader) -> Dict:
        ds = loader.dataset
        report = {"split": name, "n_samples": len(ds)}
        if hasattr(ds, "samples"):
            samples = list(getattr(ds, "samples"))
            rb = [int(s.get("risk_binary", 1 if int(s.get("fault_label", 0)) > 0 else 0)) for s in samples]
            cond = [str(s.get("condition_id", "unknown") or "unknown") for s in samples]
            sw = [float(s.get("sample_weight", 1.0)) for s in samples]
            report["risk_binary_dist"] = {str(k): int(v) for k, v in zip(*np.unique(rb, return_counts=True))}
            report["condition_id_dist"] = {str(k): int(v) for k, v in zip(*np.unique(cond, return_counts=True))}
            report["sample_weight_min"] = float(np.min(sw)) if len(sw) else 0.0
            report["sample_weight_max"] = float(np.max(sw)) if len(sw) else 0.0
        else:
            idx = getattr(ds, "indices", None)
            labels = getattr(ds, "labels", None)
            condition_ids = getattr(ds, "condition_ids", None)
            if idx is not None and labels is not None:
                y = labels[idx]
                report["risk_binary_dist"] = {str(k): int(v) for k, v in zip(*np.unique((y > 0).astype(np.int64), return_counts=True))}
            if idx is not None and condition_ids is not None:
                c = np.array(condition_ids, dtype=object)[idx]
                report["condition_id_dist"] = {str(k): int(v) for k, v in zip(*np.unique(c, return_counts=True))}
            report["sample_weight_min"] = 1.0
            report["sample_weight_max"] = 1.0
        return report

    return {
        "data_source": data_source,
        "splits": [
            _summarize_loader("train", train_loader),
            _summarize_loader("val", val_loader),
            _summarize_loader("test", test_loader),
        ],
    }


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def main():
    parser = argparse.ArgumentParser(description='训练模型')
    parser.add_argument(
        '--data_source',
        type=str,
        choices=['cwru', 'sound', 'sound_api', 'sound_api_cache'],
        default='sound_api_cache',
        help='数据源（默认 sound_api_cache，作为 XJTU 主线训练入口）'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['hi', 'risk', 'arcface', 'multi'],
        default='risk',
        help='任务类型: hi (健康指数回归), risk (风险预测), arcface (分类), multi (risk+fault+condition 多头)'
    )
    parser.add_argument(
        '--risk_semantics',
        type=str,
        choices=['unified_binary'],
        default='unified_binary',
        help='risk 语义定义（默认统一为 healthy/faulty 二值）'
    )
    parser.add_argument(
        '--use_condition_weight',
        action='store_true',
        default=True,
        help='risk 任务启用工况权重（默认开启）'
    )
    parser.add_argument(
        '--no_condition_weight',
        action='store_true',
        help='关闭 risk 工况权重（调试/对照实验用）'
    )
    parser.add_argument(
        '--eval_protocol',
        type=str,
        choices=['in_domain', 'leave_one_domain_out'],
        default='leave_one_domain_out',
        help='评估协议（默认留一域测试）'
    )
    parser.add_argument(
        '--test_domain',
        type=str,
        choices=['cwru', 'sound_api_cache'],
        default='cwru',
        help='leave_one_domain_out 时的目标测试域'
    )
    parser.add_argument(
        '--semantic_check_only',
        action='store_true',
        help='仅做语义一致性检查并输出报告，不训练'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='批大小 (默认: 128 for cwru, 8 for sound/sound_api)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='训练轮数'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='学习率'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'adamw'],
        default='adamw',
        help='优化器类型（默认 adamw）'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='权重衰减系数'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['none', 'cosine', 'plateau'],
        default='cosine',
        help='学习率调度器（默认 cosine）'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        help='调度器最小学习率'
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=10,
        help='早停耐心轮数（<=0 关闭）'
    )
    parser.add_argument(
        '--early_stop_min_delta',
        type=float,
        default=1e-4,
        help='早停最小提升阈值'
    )
    parser.add_argument(
        '--split_ratio',
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        help='数据集划分比例 [train val test]'
    )
    parser.add_argument(
        '--split_mode',
        type=str,
        choices=['random', 'leave_one_condition_out'],
        default='random',
        help='CWRU 切分方式: random 或 leave_one_condition_out'
    )
    parser.add_argument(
        '--sound_split_mode',
        type=str,
        choices=['bearing', 'leave_one_condition_out'],
        default='bearing',
        help='sound_api_cache 切分方式: bearing 或 leave_one_condition_out'
    )
    parser.add_argument(
        '--condition_map_path',
        type=str,
        default=None,
        help='bearing_id->condition_id 映射文件(.csv/.json)，用于 leave_one_condition_out'
    )
    parser.add_argument(
        '--condition_policy',
        type=str,
        choices=['xjtu_3cond', 'none'],
        default='xjtu_3cond',
        help='未提供 condition_map 时的工况推断策略'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='cwru_processed',
        help='CWRU 处理后数据目录 (signals.npy/labels.npy)'
    )
    parser.add_argument(
        '--master_labels_path',
        type=str,
        default=None,
        help='统一标签表 CSV 路径，包含 sample_id/fault_binary/condition_id'
    )
    parser.add_argument(
        '--label_source_policy',
        type=str,
        default='any',
        help='标签来源过滤策略: any/dataset_rule/alarm/manual'
    )
    parser.add_argument(
        '--label_version',
        type=str,
        default='latest',
        help='标签版本过滤，latest 表示不过滤版本'
    )
    parser.add_argument(
        '--test_condition_id',
        type=str,
        default=None,
        help='leave_one_condition_out 时指定测试工况 ID'
    )
    parser.add_argument(
        '--split_dump_dir',
        type=str,
        default='experiments/outputs/splits',
        help='切分导出目录'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=None,
        help='风险预测的时间窗口（仅用于 risk 任务）'
    )
    parser.add_argument(
        '--risk_threshold',
        type=float,
        default=0.5,
        help='风险预测阈值（默认 0.5）'
    )
    parser.add_argument(
        '--calibrate_threshold',
        action='store_true',
        help='在验证集自动校准风险阈值（F1 最优）'
    )
    parser.add_argument(
        '--calibrate_sign',
        action='store_true',
        default=True,
        help='在验证集自动校准风险分数方向（normal / inverted，默认开启）'
    )
    parser.add_argument(
        '--no_calibrate_sign',
        action='store_true',
        help='关闭风险分数方向自动校准'
    )
    parser.add_argument(
        '--risk_calibration_metric',
        type=str,
        choices=['balanced_acc', 'f1'],
        default='balanced_acc',
        help='风险后处理校准目标（默认 balanced_acc）'
    )
    parser.add_argument(
        '--risk_score_direction',
        type=str,
        choices=['auto', 'normal', 'inverted'],
        default='auto',
        help='风险分数方向: auto(验证集自动校准) / normal / inverted'
    )
    # sound_api_cache 相关参数
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='datasets/sound_api/cache_npz',
        help='NPZ 缓存目录（仅用于 sound_api_cache）'
    )
    parser.add_argument(
        '--index_path',
        type=str,
        default=None,
        help='index.csv 路径（可选，仅用于 sound_api_cache）'
    )
    # sound_api 相关参数
    parser.add_argument(
        '--audio_dir',
        type=str,
        default=None,
        help='音频文件目录 (仅用于 sound_api 数据源)'
    )
    parser.add_argument(
        '--sound_data_dir',
        type=str,
        default='声音能量曲线数据',
        help='本地 xlsx 声音数据目录 (仅用于 sound 数据源)'
    )
    parser.add_argument(
        '--sound_log1p_volume',
        action='store_true',
        default=True,
        help='sound 数据源使用 log1p(volume)+density（默认开启）'
    )
    parser.add_argument(
        '--no_sound_log1p_volume',
        action='store_true',
        help='关闭 sound 数据源的 log1p(volume) 变换（对照实验）'
    )
    parser.add_argument(
        '--sound_metadata_path',
        type=str,
        default='cwru_processed/metadata.json',
        help='sound 数据源标签元数据路径 (默认使用 CWRU metadata.json)'
    )
    parser.add_argument(
        '--api_cache_dir',
        type=str,
        default='sound_api_cache',
        help='API响应缓存目录 (仅用于 sound_api 数据源)'
    )
    parser.add_argument(
        '--use_api_cache',
        action='store_true',
        default=True,
        help='是否使用API缓存 (仅用于 sound_api 数据源)'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='仅加载 checkpoint 在测试集上评估，不训练'
    )
    parser.add_argument(
        '--allow_cwru_train',
        action='store_true',
        help='允许在 CWRU 上训练（默认关闭，CWRU 仅用于评估）'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='checkpoint 目录 (eval_only 时使用)'
    )
    parser.add_argument(
        '--init_checkpoint_dir',
        type=str,
        default=None,
        help='训练前初始化权重目录（可选），包含 backbone.pth 和 <task>_head.pth'
    )
    parser.add_argument(
        '--init_backbone_only',
        action='store_true',
        help='初始化时仅加载 backbone.pth，跳过 head 权重检查和加载'
    )
    parser.add_argument(
        '--freeze_backbone',
        action='store_true',
        help='仅训练任务 head，冻结 backbone（用于目标域小步适配）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        metavar='cuda|cpu|xpu|dml',
        help='运行设备: cuda(NVIDIA)/cpu/xpu(Intel IPEX)/dml(Windows 核显，推荐 pip install torch-directml)'
    )
    parser.add_argument(
        '--max_test_samples',
        type=int,
        default=None,
        help='快速验证时仅用测试集前 N 个样本（eval_only 时有效）'
    )
    parser.add_argument(
        '--fast_eval',
        action='store_true',
        help='约 5 分钟内跑完：64 样本 + batch32，仅 eval_only 时有效'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        metavar='N',
        help='DataLoader 的 num_workers（多进程加载，本机 4060+12900H 可试 --workers 6）'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='使用混合精度 (FP16) 训练，加速并省显存（仅 CUDA）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='全局随机种子（默认 42）'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='启用确定性模式（更可复现，但可能略慢）'
    )
    parser.add_argument(
        '--model_scale',
        type=str,
        choices=['base', 'large', 'xlarge'],
        default='base',
        help='模型规模：base(默认) / large / xlarge(更大)'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=None,
        help='embedding 维度（默认: base=512, large=768, xlarge=1024）'
    )
    parser.add_argument(
        '--head_dropout',
        type=float,
        default=0.0,
        help='头部 dropout 比例（默认 0.0，扩容模型可用 0.1）'
    )
    parser.add_argument(
        '--loss_w_risk',
        type=float,
        default=1.0,
        help='multi ?? risk ???????? 1.0?'
    )
    parser.add_argument(
        '--loss_w_fault',
        type=float,
        default=1.0,
        help='multi ?? fault ???????? 1.0?'
    )
    parser.add_argument(
        '--loss_w_condition',
        type=float,
        default=0.5,
        help='multi ?? condition ???????? 0.5?'
    )
    parser.add_argument(
        '--runs_root',
        type=str,
        default='experiments/runs',
        help='实验自动归档根目录'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='实验归档目录名（默认自动生成）'
    )
    parser.add_argument(
        '--no_archive',
        action='store_true',
        help='关闭自动归档（不建议）'
    )
    
    args = parser.parse_args()
    if getattr(args, "no_condition_weight", False):
        args.use_condition_weight = False
    if getattr(args, "no_calibrate_sign", False):
        args.calibrate_sign = False
    if getattr(args, "no_sound_log1p_volume", False):
        args.sound_log1p_volume = False
    forced_risk_direction = (
        args.risk_score_direction
        if args.task in ("risk", "multi") and args.risk_score_direction in ("normal", "inverted")
        else None
    )

    # 默认策略：XJTU 主线训练，CWRU 仅评估
    if args.data_source == 'cwru' and (not args.allow_cwru_train) and (not args.eval_only):
        args.eval_only = True
        print("策略提示: 当前默认将 CWRU 作为评估集，已自动启用 --eval_only。")
        print("如需在 CWRU 上训练，请显式添加 --allow_cwru_train。")

    if args.eval_only and args.fast_eval:
        if not args.max_test_samples or args.max_test_samples > 64:
            args.max_test_samples = 64

    set_global_seed(args.seed, deterministic=args.deterministic)
    
    # 参数提示：risk 任务现使用真实 fault_label，不再强制依赖 horizon
    if args.task == 'risk' and args.horizon is None:
        print("提示: risk 任务当前使用真实标签（fault_label），未提供 --horizon 也可训练。")
    
    # sound_api_cache + arcface 已支持（NPZ 中还原 fault_label 后过滤无标签样本）
    # 根据数据源设置默认 batch_size
    if args.batch_size is None:
        batch_size = 8 if args.data_source in ['sound', 'sound_api', 'sound_api_cache'] else 128
    else:
        batch_size = args.batch_size
    
    epochs = args.epochs
    lr = args.lr
    split_ratio = tuple(args.split_ratio)
    if args.embedding_dim:
        embedding_dim = args.embedding_dim
    elif args.model_scale == "xlarge":
        embedding_dim = 1024
    elif args.model_scale == "large":
        embedding_dim = 768
    else:
        embedding_dim = 512

    if args.device is not None:
        if args.device not in ('cuda', 'cpu', 'xpu', 'dml'):
            raise ValueError("--device 仅支持 cuda / cpu / xpu / dml")
        if args.device == 'xpu':
            try:
                import intel_extension_for_pytorch as ipex  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "使用 --device xpu 需先安装: pip install intel-extension-for-pytorch"
                )
            device = torch.device("xpu")
            if not torch.xpu.is_available():
                raise RuntimeError("未检测到 Intel XPU，请确认已安装核显驱动与 intel-extension-for-pytorch")
        elif args.device == 'dml':
            try:
                import torch_directml
                device = torch_directml.device()
            except ImportError:
                raise RuntimeError(
                    "使用 --device dml 需先安装: pip install torch-directml （Windows 下用 Intel/AMD 核显加速）"
                )
        else:
            device = torch.device(args.device)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("指定了 --device cuda 但当前环境无可用 GPU")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            try:
                import torch_directml
                device = torch_directml.device()
            except ImportError:
                try:
                    import intel_extension_for_pytorch as ipex  # noqa: F401
                    device = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
                except ImportError:
                    device = torch.device("cpu")
    use_dml = (args.device == "dml")
    print(f"使用设备: {device}")
    print(f"数据源: {args.data_source}")
    print(f"任务: {args.task}")
    print(f"批大小: {batch_size}, 训练轮数: {epochs}, 学习率: {lr}")
    print(f"模型规模: {args.model_scale}, embedding_dim: {embedding_dim}, head_dropout: {args.head_dropout}")
    print(f"随机种子: {args.seed}, deterministic={args.deterministic}")
    if args.task == "multi" and args.data_source != "cwru":
        raise ValueError("task=multi 当前仅支持 CWRU，请使用 --data_source cwru")
    if args.data_source == "sound_api_cache":
        print("运行模式: XJTU 优先主线（sound_api_cache）")
    elif args.data_source == "cwru":
        mode = "仅评估" if args.eval_only and not args.allow_cwru_train else "训练/评估"
        print(f"运行模式: CWRU {mode}")

    # 自动归档目录（每次实验独立保存）
    if args.no_archive:
        run_dir = Path(".")
        ckpt_out_dir = Path("checkpoints")
        outputs_dir = Path("experiments/outputs")
        plots_dir = outputs_dir / "plots"
        metrics_path = outputs_dir / "metrics.csv"
        overall_metrics_path = outputs_dir / "overall_metrics.csv"
        risk_pred_path = outputs_dir / "risk_predictions.csv"
        risk_cond_metrics_path = outputs_dir / "risk_condition_metrics.csv"
        risk_domain_metrics_path = outputs_dir / "risk_domain_metrics.csv"
        per_condition_metrics_path = outputs_dir / "per_condition_metrics.csv"
        per_domain_metrics_path = outputs_dir / "per_domain_metrics.csv"
        risk_score_direction_path = outputs_dir / "risk_score_direction.txt"
        risk_metrics_path = outputs_dir / "risk_metrics.csv"
        fault_metrics_path = outputs_dir / "fault_metrics.csv"
        condition_metrics_path = outputs_dir / "condition_metrics.csv"
    else:
        run_name = args.run_name or build_run_name(args)
        run_dir = Path(args.runs_root) / run_name
        ckpt_out_dir = run_dir / "checkpoints"
        outputs_dir = run_dir / "outputs"
        plots_dir = outputs_dir / "plots"
        metrics_path = outputs_dir / "metrics.csv"
        overall_metrics_path = outputs_dir / "overall_metrics.csv"
        risk_pred_path = outputs_dir / "risk_predictions.csv"
        risk_cond_metrics_path = outputs_dir / "risk_condition_metrics.csv"
        risk_domain_metrics_path = outputs_dir / "risk_domain_metrics.csv"
        per_condition_metrics_path = outputs_dir / "per_condition_metrics.csv"
        per_domain_metrics_path = outputs_dir / "per_domain_metrics.csv"
        risk_score_direction_path = outputs_dir / "risk_score_direction.txt"
        risk_metrics_path = outputs_dir / "risk_metrics.csv"
        fault_metrics_path = outputs_dir / "fault_metrics.csv"
        condition_metrics_path = outputs_dir / "condition_metrics.csv"
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs" / "splits").mkdir(parents=True, exist_ok=True)
        save_run_config(run_dir, args, device)
        print(f"实验归档目录: {run_dir}")

    def calibrate_risk_with_config(
        targets: np.ndarray,
        probs: np.ndarray,
        default_threshold: float,
    ) -> Dict[str, float]:
        """按 CLI 约束进行风险后处理校准。"""
        if forced_risk_direction in ("normal", "inverted"):
            fixed_scores = apply_risk_direction(probs, forced_risk_direction)
            cal = calibrate_risk_postprocess(
                targets=targets,
                probs=fixed_scores,
                default_threshold=default_threshold,
                allow_invert=False,
                tune_threshold=bool(args.calibrate_threshold),
                metric=args.risk_calibration_metric,
            )
            return {
                "direction": forced_risk_direction,
                "threshold": float(cal.get("threshold", default_threshold)),
                "score": float(cal.get("score", 0.0)),
                "auc": float(cal.get("auc", 0.0)),
            }
        return calibrate_risk_postprocess(
            targets=targets,
            probs=probs,
            default_threshold=default_threshold,
            allow_invert=bool(args.calibrate_sign),
            tune_threshold=bool(args.calibrate_threshold),
            metric=args.risk_calibration_metric,
        )

    # 数据加载
    workers = getattr(args, 'workers', 0)
    if args.data_source == 'sound':
        print("\n加载声音能量曲线数据（从本地xlsx文件）...")
        train_loader, val_loader, test_loader = get_sound_dataloaders(
            batch_size=batch_size,
            split_ratio=split_ratio,
            sound_data_dir=args.sound_data_dir,
            metadata_path=args.sound_metadata_path,
            task=args.task,
            risk_semantics=args.risk_semantics,
            use_log1p_volume=args.sound_log1p_volume,
            seed=args.seed,
            num_workers=workers,
        )
    elif args.data_source == 'sound_api':
        print("\n加载声音能量曲线数据（从API获取）...")
        if args.audio_dir is None:
            raise ValueError("使用 sound_api 数据源时必须指定 --audio_dir 参数")
        train_loader, val_loader, test_loader = get_sound_api_dataloaders(
            audio_dir=args.audio_dir,
            batch_size=batch_size,
            split_ratio=split_ratio,
            cache_dir=args.api_cache_dir,
            use_cache=args.use_api_cache,
            num_workers=workers,
        )
    elif args.data_source == 'sound_api_cache':
        print("\n加载声音能量曲线数据（从NPZ缓存）...")
        train_loader, val_loader, test_loader = get_sound_api_cache_dataloaders(
            batch_size=batch_size,
            split_ratio=split_ratio,
            cache_dir=args.cache_dir,
            index_path=args.index_path,
            split_mode=args.sound_split_mode,
            condition_map_path=args.condition_map_path,
            condition_policy=args.condition_policy,
            test_condition_id=args.test_condition_id,
            task=args.task,
            risk_semantics=args.risk_semantics,
            use_condition_weight=args.use_condition_weight,
            horizon=args.horizon,
            num_workers=workers,
            seed=args.seed,
        )
        split_dump_dir = args.split_dump_dir
        if not args.no_archive and split_dump_dir == 'experiments/outputs/splits':
            split_dump_dir = str(run_dir / "outputs" / "splits")
        dump_split_files_for_sound_cache(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            split_dump_dir=split_dump_dir,
        )
    else:
        print("\n加载 CWRU 处理后的数据...")
        split_dump_dir = args.split_dump_dir
        if not args.no_archive and split_dump_dir == 'experiments/outputs/splits':
            split_dump_dir = str(run_dir / "outputs" / "splits")
        if args.task == "multi":
            train_loader, val_loader, test_loader = get_cwru_multi_dataloaders(
                batch_size=batch_size,
                split_ratio=split_ratio,
                base_dir=args.base_dir,
                split_mode=args.split_mode,
                test_condition_id=args.test_condition_id,
                risk_semantics=args.risk_semantics,
                num_workers=workers,
                seed=args.seed,
            )
        else:
            train_loader, val_loader, test_loader = get_dataloaders(
                batch_size=batch_size,
                split_ratio=split_ratio,
                base_dir=args.base_dir,
                split_mode=args.split_mode,
                master_labels_path=args.master_labels_path,
                label_source_policy=args.label_source_policy,
                label_version=args.label_version,
                risk_semantics=args.risk_semantics,
                test_condition_id=args.test_condition_id,
                num_workers=workers,
                seed=args.seed,
            )
        dump_split_files_for_cwru(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            split_dump_dir=split_dump_dir,
        )

    if args.semantic_check_only:
        report = semantic_check_report(
            data_source=args.data_source,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        sem_path = outputs_dir / "semantic_check.json"
        sem_path.parent.mkdir(parents=True, exist_ok=True)
        sem_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"语义检查完成: {sem_path}")
        return

    # 模型初始化（DirectML 对部分 1D 算子支持有限，失败则回退 CPU）
    backbone = build_backbone(
        in_channels=2,
        embedding_dim=embedding_dim,
        model_scale=args.model_scale,
    )
    try:
        backbone = backbone.to(device)
    except RuntimeError:
        if use_dml:
            device = torch.device("cpu")
            backbone = backbone.to(device)
            print("警告: DirectML 无法加载该模型，已回退到 CPU。")
        else:
            raise

    if args.task == 'hi':
        head = RegressionHead(in_features=embedding_dim, dropout=args.head_dropout).to(device)
        criterion = nn.MSELoss()
        train_fn = train_one_epoch_regression
        if args.data_source == 'sound_api_cache':
            eval_fn = evaluate_regression
        else:
            eval_fn = lambda b, h, c, d, dev: (0.0, 0.0)  # placeholder
    elif args.task == 'risk':
        head = BinaryClassificationHead(in_features=embedding_dim, dropout=args.head_dropout).to(device)
        pos_weight_value = estimate_pos_weight_from_loader(train_loader)
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_tensor)
        print(f"risk pos_weight(train-only): {pos_weight_value:.4f}")
        train_fn = train_one_epoch_binary
        eval_fn = evaluate_binary
    elif args.task == 'multi':
        sample_ds = train_loader.dataset
        condition_vocab = getattr(sample_ds, 'condition_vocab', None) or ['unknown']
        n_condition_classes = int(len(condition_vocab))
        head = MultiTaskHead(
            in_features=embedding_dim,
            n_fault_classes=4,
            n_condition_classes=n_condition_classes,
            dropout=args.head_dropout,
        ).to(device)
        pos_weight_value = estimate_pos_weight_from_loader(train_loader)
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        criterion = {
            'risk': nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_tensor),
            'fault': nn.CrossEntropyLoss(),
            'condition': nn.CrossEntropyLoss(),
        }
        print(
            'multi loss weights: '
            f"risk={args.loss_w_risk}, fault={args.loss_w_fault}, condition={args.loss_w_condition}"
        )
        print(f"multi risk pos_weight(train-only): {pos_weight_value:.4f}")
        train_fn = train_one_epoch_multi
        eval_fn = evaluate_multi
    else:  # arcface
        all_labels = []
        for _, y in train_loader:
            all_labels.append(y)
        num_classes = torch.cat(all_labels).unique().numel()

        if num_classes < 2:
            raise ValueError(f"ArcFace task needs >=2 classes, got {num_classes}")

        head = ArcMarginProduct(in_features=embedding_dim, out_features=num_classes, s=30.0, m=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
        train_fn = train_one_epoch_arcface
        eval_fn = evaluate_arcface

    if args.init_checkpoint_dir:
        init_dir = Path(args.init_checkpoint_dir)
        init_backbone_path = init_dir / 'backbone.pth'
        init_head_path = init_dir / get_head_checkpoint_name(args.task)
        if not init_backbone_path.exists():
            raise FileNotFoundError(f"Init failed, missing: {init_backbone_path}")
        backbone.load_state_dict(torch.load(init_backbone_path, map_location='cpu', weights_only=True))
        if not args.init_backbone_only:
            if not init_head_path.exists():
                raise FileNotFoundError(f"Init failed, missing: {init_head_path}")
            head.load_state_dict(torch.load(init_head_path, map_location='cpu', weights_only=True))
        backbone.to(device)
        head.to(device)
        if args.init_backbone_only:
            print(f"Loaded init backbone only: {init_backbone_path}")
        else:
            print(f"Loaded init checkpoint dir: {init_dir}")

    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        print("已冻结 backbone，仅训练任务 head")

    trainable_backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    trainable_head_params = [p for p in head.parameters() if p.requires_grad]
    param_groups = trainable_backbone_params + trainable_head_params
    if len(param_groups) == 0:
        raise RuntimeError("无可训练参数，请检查 freeze 配置")
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=args.weight_decay,
        )

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=args.min_lr
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=args.min_lr
        )

    best_val_metric = float('inf') if args.task in ['hi', 'risk', 'multi'] else 0.0
    best_risk_threshold = float(args.risk_threshold)
    best_risk_direction = forced_risk_direction or "normal"
    no_improve_epochs = 0
    os.makedirs(ckpt_out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 仅评估：加载 checkpoint，仅在测试集上评估后退出
    if args.eval_only:
        ckpt_dir = Path(args.checkpoint_dir)
        backbone_path = ckpt_dir / "backbone.pth"
        head_path = ckpt_dir / get_head_checkpoint_name(args.task)
        if not backbone_path.exists() or not head_path.exists():
            print(f"未找到 checkpoint: {backbone_path} 或 {head_path}")
            return
        # 先加载到 CPU 再 to(device)，兼容 dml/xpu 等
        backbone.load_state_dict(torch.load(backbone_path, map_location="cpu", weights_only=True))
        head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        backbone.to(device)
        head.to(device)
        if args.task == "risk":
            th_path = ckpt_dir.parent / "outputs" / "risk_threshold.txt"
            if th_path.exists():
                try:
                    best_risk_threshold = float(th_path.read_text(encoding="utf-8").strip())
                    print(f"从归档读取风险阈值: {best_risk_threshold:.4f}")
                except ValueError:
                    print(f"风险阈值文件解析失败，继续使用 --risk_threshold={best_risk_threshold:.4f}")
            dir_path = ckpt_dir.parent / "outputs" / "risk_score_direction.txt"
            if dir_path.exists():
                d_text = dir_path.read_text(encoding="utf-8").strip().lower()
                if d_text in ("normal", "inverted"):
                    best_risk_direction = d_text
                    print(f"从归档读取风险方向: {best_risk_direction}")
            if forced_risk_direction in ("normal", "inverted"):
                best_risk_direction = forced_risk_direction
                print(f"使用固定风险方向: {best_risk_direction}")
            if args.calibrate_threshold or args.calibrate_sign:
                val_loss_tmp, _, val_preds_tmp, val_targets_tmp, _, _, _ = eval_fn(
                    backbone, head, criterion, val_loader, device, threshold=best_risk_threshold
                )
                cal = calibrate_risk_with_config(
                    targets=val_targets_tmp,
                    probs=val_preds_tmp,
                    default_threshold=best_risk_threshold,
                )
                best_risk_threshold = float(cal["threshold"])
                best_risk_direction = str(cal["direction"])
                print(
                    f"验证集后处理校准: direction={best_risk_direction}, "
                    f"threshold={best_risk_threshold:.4f}, score={cal.get('score', 0.0):.4f}"
                )
        # 快速验证：仅用测试集前 N 个样本，fast_eval 时用大 batch 减少迭代
        if args.task == "multi":
            th_path = ckpt_dir.parent / "outputs" / "risk_threshold.txt"
            if th_path.exists():
                try:
                    best_risk_threshold = float(th_path.read_text(encoding="utf-8").strip())
                    print(f"Load archived risk threshold: {best_risk_threshold:.4f}")
                except ValueError:
                    print(f"Risk threshold parse failed, keep --risk_threshold={best_risk_threshold:.4f}")
            dir_path = ckpt_dir.parent / "outputs" / "risk_score_direction.txt"
            if dir_path.exists():
                d_text = dir_path.read_text(encoding="utf-8").strip().lower()
                if d_text in ("normal", "inverted"):
                    best_risk_direction = d_text
                    print(f"Load archived risk direction: {best_risk_direction}")
            if forced_risk_direction in ("normal", "inverted"):
                best_risk_direction = forced_risk_direction
                print(f"Use forced risk direction: {best_risk_direction}")
            if args.calibrate_threshold or args.calibrate_sign:
                val_eval_tmp = eval_fn(
                    backbone=backbone,
                    head=head,
                    loss_fns=criterion,
                    loss_weights={
                        "risk": float(args.loss_w_risk),
                        "fault": float(args.loss_w_fault),
                        "condition": float(args.loss_w_condition),
                    },
                    dataloader=val_loader,
                    device=device,
                    risk_threshold=best_risk_threshold,
                )
                cal = calibrate_risk_with_config(
                    targets=val_eval_tmp["risk_targets"],
                    probs=val_eval_tmp["risk_probs"],
                    default_threshold=best_risk_threshold,
                )
                best_risk_threshold = float(cal["threshold"])
                best_risk_direction = str(cal["direction"])
                print(
                    f"Val calibration (multi): direction={best_risk_direction}, "
                    f"threshold={best_risk_threshold:.4f}, score={cal.get('score', 0.0):.4f}"
                )

        if getattr(args, "max_test_samples", None) and args.max_test_samples > 0:
            from torch.utils.data import Subset
            n = min(args.max_test_samples, len(test_loader.dataset))
            subset = Subset(test_loader.dataset, range(n))
            eval_batch_size = 32 if getattr(args, "fast_eval", False) else test_loader.batch_size
            test_loader = DataLoader(
                subset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=getattr(args, "workers", 0),
                collate_fn=getattr(test_loader, "collate_fn", None),
            )
            print(f"快速验证: 仅使用测试集前 {n} 个样本, batch_size={eval_batch_size}")
        print("\n" + "=" * 80)
        print("测试集评估 (仅测试集，无训练/验证)")
        print("=" * 80)
        if args.task == 'hi':
            test_loss, test_mae, test_preds, test_targets, test_meta = eval_fn(
                backbone, head, criterion, test_loader, device
            )
            test_rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
            print(f"Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f}")
            if not (getattr(args, "max_test_samples", None) and args.max_test_samples > 0):
                plot_hi_predictions(test_preds, test_targets, test_meta, str(plots_dir))
        elif args.task == 'risk':
            print(f"使用风险阈值: {best_risk_threshold:.4f} | 方向: {best_risk_direction}")
            test_loss, test_acc, test_preds, test_targets, test_meta, test_auc, test_pr_auc = eval_fn(
                backbone, head, criterion, test_loader, device, threshold=best_risk_threshold
            )
            test_scores = apply_risk_direction(test_preds, best_risk_direction)
            m = compute_binary_metrics(test_scores, test_targets, threshold=best_risk_threshold)
            print(
                f"Test Loss: {test_loss:.4f} | Test Acc: {m.get('acc', 0.0):.4f} | "
                f"Test AUC: {m.get('auc', 0.0):.4f} | Test PR-AUC: {m.get('pr_auc', 0.0):.4f}"
            )
            if not (getattr(args, "max_test_samples", None) and args.max_test_samples > 0):
                plot_risk_predictions(test_scores, test_targets, test_meta, str(plots_dir))
                save_risk_predictions_csv(
                    risk_pred_path, test_scores, test_targets, test_meta, threshold=best_risk_threshold
                )
                save_risk_condition_metrics_csv(
                    risk_cond_metrics_path, test_scores, test_targets, test_meta, threshold=best_risk_threshold
                )
                save_risk_condition_metrics_csv(
                    per_condition_metrics_path, test_scores, test_targets, test_meta, threshold=best_risk_threshold
                )
                m["test_loss"] = test_loss
                (outputs_dir / "risk_threshold.txt").write_text(
                    f"{best_risk_threshold:.6f}\n", encoding="utf-8"
                )
                risk_score_direction_path.write_text(f"{best_risk_direction}\n", encoding="utf-8")
                save_metrics_csv(metrics_path, m)
                save_metrics_csv(overall_metrics_path, m)
                save_risk_domain_metrics_csv(
                    risk_domain_metrics_path,
                    [{
                        "domain_id": args.data_source,
                        "n_samples": len(test_targets),
                        "acc": m.get("acc", 0.0),
                        "auc": m.get("auc", 0.0),
                        "pr_auc": m.get("pr_auc", 0.0),
                        "balanced_acc": m.get("balanced_acc", 0.0),
                        "f1": m.get("f1", 0.0),
                        "normal_recall": m.get("normal_recall", 0.0),
                    }],
                )
                save_risk_domain_metrics_csv(
                    per_domain_metrics_path,
                    [{
                        "domain_id": args.data_source,
                        "n_samples": len(test_targets),
                        "acc": m.get("acc", 0.0),
                        "auc": m.get("auc", 0.0),
                        "pr_auc": m.get("pr_auc", 0.0),
                        "balanced_acc": m.get("balanced_acc", 0.0),
                        "f1": m.get("f1", 0.0),
                        "normal_recall": m.get("normal_recall", 0.0),
                    }],
                )
        elif args.task == "multi":
            print(f"Use risk threshold={best_risk_threshold:.4f} | direction={best_risk_direction}")
            test_eval = eval_fn(
                backbone=backbone,
                head=head,
                loss_fns=criterion,
                loss_weights={
                    "risk": float(args.loss_w_risk),
                    "fault": float(args.loss_w_fault),
                    "condition": float(args.loss_w_condition),
                },
                dataloader=test_loader,
                device=device,
                risk_threshold=best_risk_threshold,
            )
            test_scores = apply_risk_direction(test_eval["risk_probs"], best_risk_direction)
            risk_m = compute_binary_metrics(
                test_scores, test_eval["risk_targets"], threshold=best_risk_threshold
            )
            fault_m = compute_multiclass_metrics(test_eval["fault_logits"], test_eval["fault_targets"])
            cond_m = compute_multiclass_metrics(
                test_eval["condition_logits"], test_eval["condition_targets"]
            )
            summary = {
                "test_loss": float(test_eval["loss"]),
                "risk_auc": risk_m.get("auc", 0.0),
                "risk_pr_auc": risk_m.get("pr_auc", 0.0),
                "risk_balanced_acc": risk_m.get("balanced_acc", 0.0),
                "risk_normal_recall": risk_m.get("normal_recall", 0.0),
                "fault_acc": fault_m.get("acc", 0.0),
                "fault_macro_f1": fault_m.get("macro_f1", 0.0),
                "condition_acc": cond_m.get("acc", 0.0),
                "condition_macro_f1": cond_m.get("macro_f1", 0.0),
            }
            print(
                f"Test Loss: {summary['test_loss']:.4f} | "
                f"Risk AUC: {summary['risk_auc']:.4f} | Fault F1: {summary['fault_macro_f1']:.4f} | "
                f"Condition F1: {summary['condition_macro_f1']:.4f}"
            )
            save_metrics_csv(overall_metrics_path, summary)
            save_metrics_csv(metrics_path, summary)
            risk_only = dict(risk_m)
            risk_only["test_loss"] = float(test_eval["loss_risk"])
            save_metrics_csv(risk_metrics_path, risk_only)
            save_multiclass_metrics_csv(fault_metrics_path, fault_m)
            save_multiclass_metrics_csv(condition_metrics_path, cond_m)
            save_risk_condition_metrics_csv(
                per_condition_metrics_path,
                test_scores,
                test_eval["risk_targets"],
                test_eval["meta"],
                threshold=best_risk_threshold,
            )
            (outputs_dir / "risk_threshold.txt").write_text(
                f"{best_risk_threshold:.6f}\n", encoding="utf-8"
            )
            risk_score_direction_path.write_text(f"{best_risk_direction}\n", encoding="utf-8")
        else:
            test_loss, test_acc = eval_fn(backbone, head, criterion, test_loader, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        return

    # 混合精度（仅 CUDA）
    use_amp = getattr(args, "amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        print("使用混合精度 (AMP) 训练")
    print(f"优化器: {args.optimizer}, weight_decay={args.weight_decay}")
    print(f"调度器: {args.scheduler}, min_lr={args.min_lr}")
    if args.early_stop_patience > 0:
        print(
            f"早停: patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}"
        )

    # 训练循环
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")

        if args.task == "multi":
            loss_weights = {
                "risk": float(args.loss_w_risk),
                "fault": float(args.loss_w_fault),
                "condition": float(args.loss_w_condition),
            }
            train_loss, train_metrics = train_fn(
                backbone=backbone,
                head=head,
                optimizer=optimizer,
                loss_fns=criterion,
                loss_weights=loss_weights,
                dataloader=train_loader,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
            )
        else:
            train_loss, train_metric = train_fn(
                backbone, head, optimizer, criterion, train_loader, device,
                use_amp=use_amp, scaler=scaler,
            )
        candidate_risk_threshold = best_risk_threshold
        candidate_risk_direction = best_risk_direction

        if args.task == 'hi':
            val_loss, val_metric, val_preds, val_targets, val_meta = eval_fn(
                backbone, head, criterion, val_loader, device
            )
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val MAE: {val_metric:.4f}"
            )
            metric_name = "MAE"
        elif args.task == 'risk':
            val_loss, val_acc, val_preds, val_targets, val_meta, val_auc, val_pr_auc = eval_fn(
                backbone, head, criterion, val_loader, device, threshold=best_risk_threshold
            )
            if args.calibrate_threshold or args.calibrate_sign:
                cal = calibrate_risk_with_config(
                    targets=val_targets,
                    probs=val_preds,
                    default_threshold=best_risk_threshold,
                )
                candidate_risk_threshold = float(cal["threshold"])
                candidate_risk_direction = str(cal["direction"])
            val_scores = apply_risk_direction(val_preds, candidate_risk_direction)
            val_metrics = compute_binary_metrics(
                val_scores, val_targets, threshold=candidate_risk_threshold
            )
            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_metric:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val AUC(raw): {val_auc:.4f} | Val AUC(cal): {val_metrics.get('auc', 0.0):.4f} | "
                f"Val PR-AUC(cal): {val_metrics.get('pr_auc', 0.0):.4f} | "
                f"dir={candidate_risk_direction} th={candidate_risk_threshold:.4f}"
            )
            val_metric = 1.0 - float(val_metrics.get("auc", 0.0))  # ????
            metric_name = "AUC"
        elif args.task == 'multi':
            val_eval = eval_fn(
                backbone=backbone,
                head=head,
                loss_fns=criterion,
                loss_weights={
                    "risk": float(args.loss_w_risk),
                    "fault": float(args.loss_w_fault),
                    "condition": float(args.loss_w_condition),
                },
                dataloader=val_loader,
                device=device,
                risk_threshold=best_risk_threshold,
            )
            val_preds = val_eval["risk_probs"]
            val_targets = val_eval["risk_targets"]
            if args.calibrate_threshold or args.calibrate_sign:
                cal = calibrate_risk_with_config(
                    targets=val_targets,
                    probs=val_preds,
                    default_threshold=best_risk_threshold,
                )
                candidate_risk_threshold = float(cal["threshold"])
                candidate_risk_direction = str(cal["direction"])
            val_scores = apply_risk_direction(val_preds, candidate_risk_direction)
            risk_val_metrics = compute_binary_metrics(
                val_scores, val_targets, threshold=candidate_risk_threshold
            )
            fault_val_metrics = compute_multiclass_metrics(
                val_eval["fault_logits"], val_eval["fault_targets"]
            )
            cond_val_metrics = compute_multiclass_metrics(
                val_eval["condition_logits"], val_eval["condition_targets"]
            )
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Risk/Fault/Cond Train Acc: {train_metrics.get('risk_acc', 0.0):.4f}/"
                f"{train_metrics.get('fault_acc', 0.0):.4f}/"
                f"{train_metrics.get('condition_acc', 0.0):.4f} | "
                f"Val Loss: {val_eval['loss']:.4f} | "
                f"Val Risk AUC: {risk_val_metrics.get('auc', 0.0):.4f} | "
                f"Val Fault F1: {fault_val_metrics.get('macro_f1', 0.0):.4f} | "
                f"Val Cond F1: {cond_val_metrics.get('macro_f1', 0.0):.4f} | "
                f"dir={candidate_risk_direction} th={candidate_risk_threshold:.4f}"
            )
            score = (
                float(risk_val_metrics.get("auc", 0.0))
                + float(fault_val_metrics.get("macro_f1", 0.0))
                + float(cond_val_metrics.get("macro_f1", 0.0))
            ) / 3.0
            val_metric = 1.0 - score
            val_loss = float(val_eval["loss"])
            metric_name = "Composite"
        else:  # arcface
            val_loss, val_metric = eval_fn(
                backbone, head, criterion, val_loader, device
            )
            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_metric:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_metric:.4f}"
            )
            metric_name = "Acc"

        # 保存最优模型
        is_best = False
        improved = False
        if args.task in ['hi', 'risk', 'multi']:
            if val_metric < (best_val_metric - args.early_stop_min_delta):
                best_val_metric = val_metric
                is_best = True
                improved = True
        else:
            if val_metric > (best_val_metric + args.early_stop_min_delta):
                best_val_metric = val_metric
                is_best = True
                improved = True
        
        if is_best:
            if args.task in ("risk", "multi"):
                best_risk_threshold = float(candidate_risk_threshold)
                best_risk_direction = str(candidate_risk_direction)
                print(
                    f"  -> Calibrated risk postprocess: "
                    f"direction={best_risk_direction}, threshold={best_risk_threshold:.4f}"
                )
            torch.save(backbone.state_dict(), ckpt_out_dir / "backbone.pth")
            torch.save(head.state_dict(), ckpt_out_dir / get_head_checkpoint_name(args.task))
            if args.task in ("risk", "multi"):
                best_display_metric = 1.0 - best_val_metric
            else:
                best_display_metric = best_val_metric
            print(f"  -> New best model saved (Val {metric_name}: {best_display_metric:.4f})")
            if args.task in ("risk", "multi"):
                (outputs_dir / "risk_threshold.txt").write_text(
                    f"{best_risk_threshold:.6f}\n", encoding="utf-8"
                )
                risk_score_direction_path.write_text(
                    f"{best_risk_direction}\n", encoding="utf-8"
                )

        # 学习率调度（plateau 用验证损失，其余按 epoch）
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  -> LR: {current_lr:.6e}")

        # 早停
        if improved:
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                print(
                    f"触发早停: 连续 {no_improve_epochs} 个 epoch 无显著提升，停止训练。"
                )
                break

    # 统一用最佳 checkpoint 做最终测试，避免最后一轮回退
    best_backbone = ckpt_out_dir / "backbone.pth"
    best_head = ckpt_out_dir / get_head_checkpoint_name(args.task)
    if best_backbone.exists() and best_head.exists():
        backbone.load_state_dict(torch.load(best_backbone, map_location="cpu", weights_only=True))
        head.load_state_dict(torch.load(best_head, map_location="cpu", weights_only=True))
        backbone.to(device)
        head.to(device)
        print("已加载最佳 checkpoint 进行最终测试。")

    # 测试集评估
    print("\n" + "=" * 80)
    print("测试集评估")
    print("=" * 80)
    
    if args.task == 'hi':
        test_loss, test_mae, test_preds, test_targets, test_meta = eval_fn(
            backbone, head, criterion, test_loader, device
        )
        test_rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
        print(f"Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f}")
        
        # 绘图
        plot_hi_predictions(test_preds, test_targets, test_meta, str(plots_dir))
        
        # 保存指标
        _metrics = {
            "test_loss": test_loss,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        }
        save_metrics_csv(metrics_path, _metrics)
        save_metrics_csv(overall_metrics_path, _metrics)
    
    elif args.task == 'risk':
        print(f"最终风险阈值: {best_risk_threshold:.4f} | 方向: {best_risk_direction}")
        test_loss, test_acc, test_preds, test_targets, test_meta, test_auc, test_pr_auc = eval_fn(
            backbone, head, criterion, test_loader, device, threshold=best_risk_threshold
        )
        test_scores = apply_risk_direction(test_preds, best_risk_direction)
        risk_metrics = compute_binary_metrics(test_scores, test_targets, threshold=best_risk_threshold)
        print(
            f"Test Loss: {test_loss:.4f} | Test Acc: {risk_metrics.get('acc', 0.0):.4f} | "
            f"Test AUC: {risk_metrics.get('auc', 0.0):.4f} | Test PR-AUC: {risk_metrics.get('pr_auc', 0.0):.4f}"
        )
        
        # 绘图
        plot_risk_predictions(test_scores, test_targets, test_meta, str(plots_dir))
        save_risk_predictions_csv(
            risk_pred_path, test_scores, test_targets, test_meta, threshold=best_risk_threshold
        )
        save_risk_condition_metrics_csv(
            risk_cond_metrics_path, test_scores, test_targets, test_meta, threshold=best_risk_threshold
        )
        save_risk_condition_metrics_csv(
            per_condition_metrics_path, test_scores, test_targets, test_meta, threshold=best_risk_threshold
        )
        risk_metrics["test_loss"] = test_loss
        risk_score_direction_path.write_text(f"{best_risk_direction}\n", encoding="utf-8")
        # 保存指标
        save_metrics_csv(metrics_path, risk_metrics)
        save_metrics_csv(overall_metrics_path, risk_metrics)

        domain_rows = [{
            "domain_id": args.data_source,
            "n_samples": len(test_targets),
            "acc": risk_metrics.get("acc", 0.0),
            "auc": risk_metrics.get("auc", 0.0),
            "pr_auc": risk_metrics.get("pr_auc", 0.0),
            "balanced_acc": risk_metrics.get("balanced_acc", 0.0),
            "f1": risk_metrics.get("f1", 0.0),
            "normal_recall": risk_metrics.get("normal_recall", 0.0),
        }]

        if args.eval_protocol == "leave_one_domain_out" and args.test_domain == "cwru" and args.data_source != "cwru":
            try:
                cwru_train_loader, cwru_val_loader, cwru_test_loader = get_dataloaders(
                    batch_size=batch_size,
                    split_ratio=split_ratio,
                    base_dir=args.base_dir,
                    split_mode="random",
                    master_labels_path=args.master_labels_path,
                    label_source_policy=args.label_source_policy,
                    label_version=args.label_version,
                    risk_semantics=args.risk_semantics,
                    num_workers=workers,
                    seed=args.seed,
                )
                cwru_loss, cwru_acc, cwru_preds, cwru_targets, _, _, _ = evaluate_binary(
                    backbone, head, criterion, cwru_test_loader, device, threshold=best_risk_threshold
                )
                cwru_scores = apply_risk_direction(cwru_preds, best_risk_direction)
                cwru_metrics = compute_binary_metrics(cwru_scores, cwru_targets, threshold=best_risk_threshold)
                cwru_metrics["test_loss"] = cwru_loss
                domain_rows.append({
                    "domain_id": "cwru",
                    "n_samples": len(cwru_targets),
                    "acc": cwru_metrics.get("acc", 0.0),
                    "auc": cwru_metrics.get("auc", 0.0),
                    "pr_auc": cwru_metrics.get("pr_auc", 0.0),
                    "balanced_acc": cwru_metrics.get("balanced_acc", 0.0),
                    "f1": cwru_metrics.get("f1", 0.0),
                    "normal_recall": cwru_metrics.get("normal_recall", 0.0),
                })
                print(
                    f"[Leave-One-Domain] CWRU Test | Loss: {cwru_loss:.4f} | "
                    f"AUC: {cwru_metrics.get('auc', 0.0):.4f} | "
                    f"BalancedAcc: {cwru_metrics.get('balanced_acc', 0.0):.4f} | "
                    f"NormalRecall: {cwru_metrics.get('normal_recall', 0.0):.4f}"
                )
            except Exception as e:
                print(f"跨域评估(CWRU)跳过: {e}")

        save_risk_domain_metrics_csv(risk_domain_metrics_path, domain_rows)
        save_risk_domain_metrics_csv(per_domain_metrics_path, domain_rows)
    
    elif args.task == "multi":
        print(f"Final risk threshold: {best_risk_threshold:.4f} | direction: {best_risk_direction}")
        test_eval = eval_fn(
            backbone=backbone,
            head=head,
            loss_fns=criterion,
            loss_weights={
                "risk": float(args.loss_w_risk),
                "fault": float(args.loss_w_fault),
                "condition": float(args.loss_w_condition),
            },
            dataloader=test_loader,
            device=device,
            risk_threshold=best_risk_threshold,
        )
        test_scores = apply_risk_direction(test_eval["risk_probs"], best_risk_direction)
        risk_metrics = compute_binary_metrics(
            test_scores, test_eval["risk_targets"], threshold=best_risk_threshold
        )
        fault_metrics = compute_multiclass_metrics(
            test_eval["fault_logits"], test_eval["fault_targets"]
        )
        condition_metrics = compute_multiclass_metrics(
            test_eval["condition_logits"], test_eval["condition_targets"]
        )
        summary_metrics = {
            "test_loss": float(test_eval["loss"]),
            "risk_auc": risk_metrics.get("auc", 0.0),
            "risk_pr_auc": risk_metrics.get("pr_auc", 0.0),
            "risk_balanced_acc": risk_metrics.get("balanced_acc", 0.0),
            "risk_normal_recall": risk_metrics.get("normal_recall", 0.0),
            "fault_acc": fault_metrics.get("acc", 0.0),
            "fault_macro_f1": fault_metrics.get("macro_f1", 0.0),
            "condition_acc": condition_metrics.get("acc", 0.0),
            "condition_macro_f1": condition_metrics.get("macro_f1", 0.0),
        }
        print(
            f"Test Loss: {summary_metrics['test_loss']:.4f} | "
            f"Risk AUC: {summary_metrics['risk_auc']:.4f} | "
            f"Fault F1: {summary_metrics['fault_macro_f1']:.4f} | "
            f"Condition F1: {summary_metrics['condition_macro_f1']:.4f}"
        )
        save_metrics_csv(metrics_path, summary_metrics)
        save_metrics_csv(overall_metrics_path, summary_metrics)
        risk_only_metrics = dict(risk_metrics)
        risk_only_metrics["test_loss"] = float(test_eval["loss_risk"])
        save_metrics_csv(risk_metrics_path, risk_only_metrics)
        save_multiclass_metrics_csv(fault_metrics_path, fault_metrics)
        save_multiclass_metrics_csv(condition_metrics_path, condition_metrics)
        save_risk_condition_metrics_csv(
            per_condition_metrics_path,
            test_scores,
            test_eval["risk_targets"],
            test_eval["meta"],
            threshold=best_risk_threshold,
        )
        (outputs_dir / "risk_threshold.txt").write_text(
            f"{best_risk_threshold:.6f}\n", encoding="utf-8"
        )
        risk_score_direction_path.write_text(f"{best_risk_direction}\n", encoding="utf-8")

    else:  # arcface
        test_loss, test_acc = eval_fn(
            backbone, head, criterion, test_loader, device
        )
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        _metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        save_metrics_csv(metrics_path, _metrics)
        save_metrics_csv(overall_metrics_path, _metrics)
        # 输出类别 id 与可读名对应
        n_cls = getattr(head, "out_features", None)
        if n_cls is not None:
            names = [CLASS_ID_TO_NAME.get(i, str(i)) for i in range(n_cls)]
            print(f"类别映射: {', '.join(f'{i}={names[i]}' for i in range(n_cls))}")

    # 归档完整性检查：确保关键产物存在
    if not args.no_archive:
        required_paths = [
            ckpt_out_dir / "backbone.pth",
            ckpt_out_dir / get_head_checkpoint_name(args.task),
            outputs_dir / "metrics.csv",
            outputs_dir / "overall_metrics.csv",
            outputs_dir / "splits",
            plots_dir,
        ]
        if args.task == "risk":
            required_paths.append(outputs_dir / "risk_condition_metrics.csv")
            required_paths.append(outputs_dir / "risk_domain_metrics.csv")
            required_paths.append(outputs_dir / "per_condition_metrics.csv")
            required_paths.append(outputs_dir / "per_domain_metrics.csv")
            required_paths.append(outputs_dir / "risk_score_direction.txt")
        if args.task == "multi":
            required_paths.append(outputs_dir / "risk_metrics.csv")
            required_paths.append(outputs_dir / "fault_metrics.csv")
            required_paths.append(outputs_dir / "condition_metrics.csv")
            required_paths.append(outputs_dir / "per_condition_metrics.csv")
            required_paths.append(outputs_dir / "risk_score_direction.txt")
        missing = [str(p) for p in required_paths if not p.exists()]
        if missing:
            print("警告: 归档存在缺失项:")
            for m in missing:
                print(f"  - {m}")
        else:
            print("归档检查通过：checkpoints / plots / splits 已生成。")


if __name__ == "__main__":
    main()
