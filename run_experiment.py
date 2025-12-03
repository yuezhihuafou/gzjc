import argparse
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# 添加路径以确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.load_cwru import CWRUDataLoader
from core.features import FFTFeatureExtractor, MechanismFeatureExtractor, SoundMetricsExtractor, HybridFeatureExtractor
from core.models import RandomForestModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_feature_extractor(feature_type, fs):
    if feature_type == 'fft':
        logger.info("Using FFT Amplitude and Phase features")
        return FFTFeatureExtractor(include_phase=True)
    elif feature_type == 'mechanism':
        logger.info("Using Mechanism features (Fault Characteristic Frequencies)")
        return MechanismFeatureExtractor()
    elif feature_type == 'sound':
        logger.info("Using Sound Density and Energy features (Proxy)")
        return SoundMetricsExtractor()
    elif feature_type == 'hybrid':
        logger.info("Using Hybrid features (FFT + Mechanism)")
        return HybridFeatureExtractor()
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

def main():
    parser = argparse.ArgumentParser(description='Run Fault Diagnosis Experiment')
    parser.add_argument('--data_dir', type=str, default='CWRU-dataset-main', help='Path to CWRU dataset')
    parser.add_argument('--feature_type', type=str, choices=['fft', 'mechanism', 'sound', 'hybrid'], default='fft', 
                        help='Type of features to use: fft, mechanism, sound, or hybrid (default: fft)')
    parser.add_argument('--segment_length', type=int, default=2048, help='Signal segment length')
    parser.add_argument('--split_mode', type=str, choices=['random', 'cross_load'], default='cross_load',
                        help='Data split mode: random (standard) or cross_load (train on 0-2HP, test on 3HP)')
    parser.add_argument('--noise_snr', type=float, default=None, help='Add Gaussian noise to test set with specified SNR (dB). None means no noise.')
    parser.add_argument('--model_save_path', type=str, default='rf_model.pkl', help='Path to save trained model')
    
    args = parser.parse_args()
    
    # 1. 加载数据
    logger.info(f"Loading data from {args.data_dir}...")
    loader = CWRUDataLoader(args.data_dir)
    # 为了快速演示，这里只加载部分数据或全部数据
    # 注意：MechanismFeatureExtractor 需要 RPM 信息。
    # load_cwru.py 的 load_all 返回的 meta 包含 'rpm'。
    
    X_raw, y, meta = loader.load_all(segment_length=args.segment_length, normalize=True)
    
    # 转换为 numpy 数组 (如果是 list)
    if isinstance(X_raw, list):
        X_raw = np.array(X_raw)
        
    logger.info(f"Data loaded. Shape: {X_raw.shape}")

    # 3. 数据集划分 (先划分，再加噪声，防止训练集被污染)
    # 注意：我们需要先划分原始信号，然后再提取特征。
    # 但目前的逻辑是先提取特征再划分。
    # 为了支持加噪声测试，我们需要调整流程：
    # Load -> Split (Raw) -> Add Noise (Test Raw) -> Extract Features -> Train/Eval
    
    # 重新组织流程：
    
    # 3.1 划分原始数据
    if args.split_mode == 'random':
        logger.info("Using Random Split (Standard)")
        # 为了保持 meta 对应，我们需要手动 split
        indices = np.arange(len(X_raw))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    elif args.split_mode == 'cross_load':
        logger.info("Using Cross-Load Split (Train: 0,1,2 HP | Test: 3 HP)")
        loads = np.array([m.get('load_hp', -1) for m in meta])
        train_idx = np.where(np.isin(loads, [0, 1, 2]))[0]
        test_idx = np.where(loads == 3)[0]
    else:
        raise ValueError(f"Unknown split mode: {args.split_mode}")
        
    X_train_raw = X_raw[train_idx]
    y_train = y[train_idx]
    meta_train = [meta[i] for i in train_idx]
    
    X_test_raw = X_raw[test_idx]
    y_test = y[test_idx]
    meta_test = [meta[i] for i in test_idx]
    
    logger.info(f"Train set (Raw): {X_train_raw.shape}, Test set (Raw): {X_test_raw.shape}")
    
    # 3.2 给测试集加噪声 (如果指定)
    if args.noise_snr is not None:
        logger.info(f"Adding Gaussian Noise to Test Set (SNR={args.noise_snr} dB)...")
        
        def add_noise(signals, snr_db):
            noisy_signals = []
            for sig in signals:
                # P_signal = mean(sig^2)
                p_sig = np.mean(sig**2)
                # SNR_db = 10 * log10(P_sig / P_noise)
                # P_noise = P_sig / 10^(SNR_db/10)
                p_noise = p_sig / (10 ** (snr_db / 10))
                noise = np.random.normal(0, np.sqrt(p_noise), sig.shape)
                noisy_signals.append(sig + noise)
            return np.array(noisy_signals)
            
        X_test_raw = add_noise(X_test_raw, args.noise_snr)
    
    # 2. 特征提取 (分别对训练集和测试集提取)
    logger.info(f"Extracting features: {args.feature_type}...")
    extractor = get_feature_extractor(args.feature_type, fs=12000)
    
    # 准备 RPM
    rpms_train = np.array([m.get('rpm') if m.get('rpm') is not None else 1797 for m in meta_train])
    rpms_test = np.array([m.get('rpm') if m.get('rpm') is not None else 1797 for m in meta_test])
    
    X_train = extractor.extract(X_train_raw, fs=12000, rpm=rpms_train)
    X_test = extractor.extract(X_test_raw, fs=12000, rpm=rpms_test)
    
    logger.info(f"Features extracted. Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 4. 模型训练
    logger.info("Training Random Forest Model...")
    model = RandomForestModel()
    model.train(X_train, y_train)
    
    # 5. 评估
    logger.info("Training Random Forest Model...")
    model = RandomForestModel()
    model.train(X_train, y_train)
    
    # 5. 评估
    logger.info("Evaluating model...")
    results = model.evaluate(X_test, y_test)
    print("\n" + "="*30)
    print(f"Feature Type: {args.feature_type}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("Classification Report:")
    print(results['report'])
    print("="*30)
    
    # 6. 保存模型
    model.save(args.model_save_path)
    logger.info(f"Model saved to {args.model_save_path}")

if __name__ == '__main__':
    main()
