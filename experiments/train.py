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
from typing import Tuple, Dict, List, Optional
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

from dl.data_loader import get_dataloaders, get_sound_api_cache_dataloaders
from dl.sound_data_loader import get_sound_dataloaders
from dl.sound_api_data_loader import get_sound_api_dataloaders
from dl.model import build_backbone
from dl.loss import ArcMarginProduct

# 类别 id → 可读名（与 CWRU/项目约定一致），评估/推理时输出用
CLASS_ID_TO_NAME = {0: "Normal", 1: "B", 2: "IR", 3: "OR"}


class RegressionHead(nn.Module):
    """回归头：输出单个标量"""
    def __init__(self, in_features: int = 512):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # (B,)


class BinaryClassificationHead(nn.Module):
    """二分类头：输出单个 logit"""
    def __init__(self, in_features: int = 512):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # (B,)


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
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            features = backbone(x)
            logits = head(features)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += x.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


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
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
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
        choices=['hi', 'risk', 'arcface'],
        default='risk',
        help='任务类型: hi (健康指数回归), risk (风险预测/故障检测), arcface (分类)'
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
    
    args = parser.parse_args()

    # 默认策略：XJTU 主线训练，CWRU 仅评估
    if args.data_source == 'cwru' and (not args.allow_cwru_train) and (not args.eval_only):
        args.eval_only = True
        print("策略提示: 当前默认将 CWRU 作为评估集，已自动启用 --eval_only。")
        print("如需在 CWRU 上训练，请显式添加 --allow_cwru_train。")

    if args.eval_only and args.fast_eval:
        if not args.max_test_samples or args.max_test_samples > 64:
            args.max_test_samples = 64
    
    # 验证参数
    if args.task == 'risk' and args.horizon is None:
        raise ValueError("risk 任务需要指定 --horizon 参数")
    
    # sound_api_cache + arcface 已支持（NPZ 中还原 fault_label 后过滤无标签样本）
    # 根据数据源设置默认 batch_size
    if args.batch_size is None:
        batch_size = 8 if args.data_source in ['sound', 'sound_api', 'sound_api_cache'] else 128
    else:
        batch_size = args.batch_size
    
    epochs = args.epochs
    lr = args.lr
    split_ratio = tuple(args.split_ratio)

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
    if args.data_source == "sound_api_cache":
        print("运行模式: XJTU 优先主线（sound_api_cache）")
    elif args.data_source == "cwru":
        mode = "仅评估" if args.eval_only and not args.allow_cwru_train else "训练/评估"
        print(f"运行模式: CWRU {mode}")

    # 数据加载
    workers = getattr(args, 'workers', 0)
    if args.data_source == 'sound':
        print("\n加载声音能量曲线数据（从本地xlsx文件）...")
        train_loader, val_loader, test_loader = get_sound_dataloaders(
            batch_size=batch_size,
            split_ratio=split_ratio,
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
            task=args.task,
            horizon=args.horizon,
            num_workers=workers,
        )
    else:
        print("\n加载 CWRU 处理后的数据...")
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=batch_size,
            split_ratio=split_ratio,
            base_dir=args.base_dir,
            split_mode=args.split_mode,
            master_labels_path=args.master_labels_path,
            label_source_policy=args.label_source_policy,
            label_version=args.label_version,
            test_condition_id=args.test_condition_id,
            num_workers=workers,
        )
        dump_split_files_for_cwru(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            split_dump_dir=args.split_dump_dir,
        )

    # 模型初始化（DirectML 对部分 1D 算子支持有限，失败则回退 CPU）
    backbone = build_backbone(in_channels=2, embedding_dim=512)
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
        head = RegressionHead(in_features=512).to(device)
        criterion = nn.MSELoss()
        train_fn = train_one_epoch_regression
        if args.data_source == 'sound_api_cache':
            eval_fn = evaluate_regression
        else:
            eval_fn = lambda b, h, c, d, dev: (0.0, 0.0)  # 占位
    elif args.task == 'risk':
        head = BinaryClassificationHead(in_features=512).to(device)
        criterion = nn.BCEWithLogitsLoss()
        train_fn = train_one_epoch_binary
        if args.data_source == 'sound_api_cache':
            eval_fn = evaluate_binary
        else:
            eval_fn = lambda b, h, c, d, dev: (0.0, 0.0, np.array([]), np.array([]), [], 0.0, 0.0)
    else:  # arcface
        # 推断 num_classes
        all_labels = []
        for _, y in train_loader:
            all_labels.append(y)
        num_classes = torch.cat(all_labels).unique().numel()
        
        if num_classes < 2:
            raise ValueError(f"ArcFace 任务需要至少 2 个类别，当前只有 {num_classes} 个")
        
        head = ArcMarginProduct(in_features=512, out_features=num_classes, s=30.0, m=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
        train_fn = train_one_epoch_arcface
        eval_fn = evaluate_arcface

    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    best_val_metric = float('inf') if args.task in ['hi', 'risk'] else 0.0
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("experiments/outputs/plots", exist_ok=True)

    # 仅评估：加载 checkpoint，仅在测试集上评估后退出
    if args.eval_only:
        ckpt_dir = Path(args.checkpoint_dir)
        backbone_path = ckpt_dir / "backbone.pth"
        head_path = ckpt_dir / f"{args.task}_head.pth"
        if not backbone_path.exists() or not head_path.exists():
            print(f"未找到 checkpoint: {backbone_path} 或 {head_path}")
            return
        # 先加载到 CPU 再 to(device)，兼容 dml/xpu 等
        backbone.load_state_dict(torch.load(backbone_path, map_location="cpu", weights_only=True))
        head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        backbone.to(device)
        head.to(device)
        # 快速验证：仅用测试集前 N 个样本，fast_eval 时用大 batch 减少迭代
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
                plot_hi_predictions(test_preds, test_targets, test_meta, "experiments/outputs/plots")
        elif args.task == 'risk':
            test_loss, test_acc, test_preds, test_targets, test_meta, test_auc, test_pr_auc = eval_fn(
                backbone, head, criterion, test_loader, device
            )
            print(
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                f"Test AUC: {test_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f}"
            )
            if not (getattr(args, "max_test_samples", None) and args.max_test_samples > 0):
                plot_risk_predictions(test_preds, test_targets, test_meta, "experiments/outputs/plots")
        else:
            test_loss, test_acc = eval_fn(backbone, head, criterion, test_loader, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        return

    # 混合精度（仅 CUDA）
    use_amp = getattr(args, "amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        print("使用混合精度 (AMP) 训练")

    # 训练循环
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")

        train_loss, train_metric = train_fn(
            backbone, head, optimizer, criterion, train_loader, device,
            use_amp=use_amp, scaler=scaler,
        )
        
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
                backbone, head, criterion, val_loader, device
            )
            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_metric:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val AUC: {val_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f}"
            )
            val_metric = 1.0 - val_auc  # 用于 best model 选择（越小越好）
            metric_name = "AUC"
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
        if args.task in ['hi', 'risk']:
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                is_best = True
        else:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                is_best = True
        
        if is_best:
            torch.save(backbone.state_dict(), os.path.join("checkpoints", "backbone.pth"))
            if args.task == 'arcface':
                torch.save(head.state_dict(), os.path.join("checkpoints", "arcface_head.pth"))
            else:
                torch.save(head.state_dict(), os.path.join("checkpoints", f"{args.task}_head.pth"))
            print(f"  -> New best model saved (Val {metric_name}: {best_val_metric:.4f})")

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
        plot_hi_predictions(test_preds, test_targets, test_meta, "experiments/outputs/plots")
        
        # 保存指标
        with open("experiments/outputs/metrics.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['test_loss', test_loss])
            writer.writerow(['test_mae', test_mae])
            writer.writerow(['test_rmse', test_rmse])
    
    elif args.task == 'risk':
        test_loss, test_acc, test_preds, test_targets, test_meta, test_auc, test_pr_auc = eval_fn(
            backbone, head, criterion, test_loader, device
        )
        print(
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
            f"Test AUC: {test_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f}"
        )
        
        # 绘图
        plot_risk_predictions(test_preds, test_targets, test_meta, "experiments/outputs/plots")
        
        # 保存指标
        with open("experiments/outputs/metrics.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['test_loss', test_loss])
            writer.writerow(['test_acc', test_acc])
            writer.writerow(['test_auc', test_auc])
            writer.writerow(['test_pr_auc', test_pr_auc])
    
    else:  # arcface
        test_loss, test_acc = eval_fn(
            backbone, head, criterion, test_loader, device
        )
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        # 输出类别 id 与可读名对应
        n_cls = getattr(head, "out_features", None)
        if n_cls is not None:
            names = [CLASS_ID_TO_NAME.get(i, str(i)) for i in range(n_cls)]
            print(f"类别映射: {', '.join(f'{i}={names[i]}' for i in range(n_cls))}")


if __name__ == "__main__":
    main()
