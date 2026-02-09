#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的实验运行脚本（提交版本）

这是一个简化示例，展示了如何使用特征提取和模型训练框架。
核心优化参数和跨工况逻辑已在内部实现。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import logging

# 添加项目根目录到 sys.path，确保可以导入 tools/core 等模块
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.load_cwru import CWRUDataLoader
from core.features import FFTFeatureExtractor, MechanismFeatureExtractor
from core.models import RandomForestModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    故障诊断实验的主函数
    
    支持的特征类型：
    - fft: 频域特征
    - mechanism: 物理机理特征
    """
    parser = argparse.ArgumentParser(
        description='基于 CWRU 数据集的故障诊断实验'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='CWRU-dataset-main',
        help='CWRU 数据集路径'
    )
    parser.add_argument(
        '--feature_type',
        type=str,
        choices=['fft', 'mechanism'],
        default='fft',
        help='特征提取方法'
    )
    parser.add_argument(
        '--segment_length',
        type=int,
        default=2048,
        help='信号分段长度'
    )
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='rf_model.pkl',
        help='模型保存路径'
    )
    
    args = parser.parse_args()
    
    # =====================================================================
    # 第1步：数据加载
    # =====================================================================
    logger.info(f"从 {args.data_dir} 加载数据...")
    loader = CWRUDataLoader(args.data_dir)
    
    # 加载数据并分段
    X_raw, y, meta = loader.load_all(
        segment_length=args.segment_length,
        normalize=True
    )
    
    if isinstance(X_raw, list):
        X_raw = np.array(X_raw)
    
    logger.info(f"数据加载完成。形状: {X_raw.shape}")
    
    # =====================================================================
    # 第2步：特征提取
    # =====================================================================
    logger.info(f"使用 {args.feature_type.upper()} 方法提取特征...")
    
    if args.feature_type == 'fft':
        extractor = FFTFeatureExtractor()
    else:  # mechanism
        extractor = MechanismFeatureExtractor()
    
    # 提取特征
    fs = 12000  # CWRU 采样率
    X = extractor.extract(X_raw, fs=fs)
    
    logger.info(f"特征提取完成。形状: {X.shape}")
    
    # =====================================================================
    # 第3步：数据分割
    # =====================================================================
    logger.info("分割训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(
        f"训练集: {X_train.shape[0]} 样本, "
        f"测试集: {X_test.shape[0]} 样本"
    )
    
    # =====================================================================
    # 第4步：模型训练
    # =====================================================================
    logger.info("训练随机森林模型...")
    model = RandomForestModel()
    model.train(X_train, y_train)
    
    # =====================================================================
    # 第5步：模型评估
    # =====================================================================
    logger.info("评估模型性能...")
    
    # 训练集性能
    train_acc = model.predict(X_train)
    train_acc = np.mean(train_acc == y_train)
    
    # 测试集性能
    test_pred = model.predict(X_test)
    test_acc = np.mean(test_pred == y_test)
    
    logger.info(f"训练集准确率: {train_acc:.4f}")
    logger.info(f"测试集准确率: {test_acc:.4f}")
    
    # =====================================================================
    # 第6步：模型保存
    # =====================================================================
    logger.info(f"保存模型到 {args.model_save_path}...")
    model.save(args.model_save_path)
    
    logger.info("实验完成！")


if __name__ == '__main__':
    main()
