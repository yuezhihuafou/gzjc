import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    """特征提取器基类"""
    @abstractmethod
    def extract(self, signals, fs, rpm=None):
        """
        Args:
            signals: (N, L) 原始信号数组
            fs: 采样率
            rpm: 转速 (可选, 用于机理特征)
        Returns:
            features: (N, D) 特征矩阵
        """
        pass

class FFTFeatureExtractor(FeatureExtractor):
    """
    提取 FFT 幅值谱和相位谱
    """
    def __init__(self, n_points=None, include_phase=True):
        self.n_points = n_points
        self.include_phase = include_phase

    def extract(self, signals, fs, rpm=None):
        N = signals.shape[0]
        L = signals.shape[1]
        n_fft = self.n_points if self.n_points else L
        
        # 计算 FFT
        # rfft 只返回正频率部分，更适合实数信号
        fft_vals = np.fft.rfft(signals, n=n_fft, axis=1)
        
        # 幅值谱 (归一化)
        amplitude = np.abs(fft_vals) / L
        
        features = amplitude
        
        if self.include_phase:
            # 相位谱
            phase = np.angle(fft_vals)
            features = np.concatenate([amplitude, phase], axis=1)
            
        return features

class MechanismFeatureExtractor(FeatureExtractor):
    """
    提取特征倍频信号误差 (机理特征)
    基于轴承几何参数计算故障特征频率，并提取这些频率处的能量。
    """
    def __init__(self, bearing_type='6205-2RS'):
        # SKF 6205-2RS (Drive End) 参数
        if bearing_type == '6205-2RS':
            self.d = 7.94   # 滚动体直径 mm
            self.D = 39.04  # 节径 mm
            self.n = 9      # 滚动体个数
            self.alpha = 0  # 接触角
        else:
            # 默认参数或抛出错误
            self.d = 7.94
            self.D = 39.04
            self.n = 9
            self.alpha = 0
            
    def _calculate_fault_freqs(self, rpm):
        """计算理论故障频率 (Hz)"""
        fr = rpm / 60.0
        cos_a = np.cos(np.deg2rad(self.alpha))
        ratio = (self.d / self.D) * cos_a
        
        bpfi = (self.n / 2) * fr * (1 + ratio) # 内圈
        bpfo = (self.n / 2) * fr * (1 - ratio) # 外圈
        bsf  = (self.D / (2 * self.d)) * fr * (1 - ratio**2) # 滚动体
        ftf  = (1 / 2) * fr * (1 - ratio)      # 保持架
        
        return {'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf, 'FTF': ftf}

    def extract(self, signals, fs, rpm=None):
        """
        Args:
            signals: (N, L) 信号数组
            fs: 采样率
            rpm: (N, ) 或 float. 每个样本的转速. 如果是 None, 使用默认 1750.
        """
        N, L = signals.shape
        freqs = np.fft.rfftfreq(L, d=1/fs)
        fft_vals = np.abs(np.fft.rfft(signals, axis=1)) / L
        
        # 处理 RPM 输入
        if rpm is None:
            rpm_array = np.full(N, 1750.0)
        elif np.isscalar(rpm):
            rpm_array = np.full(N, rpm)
        else:
            rpm_array = np.array(rpm)
            if rpm_array.shape[0] != N:
                # 如果传入的 RPM 数量不匹配 (比如只传了一个标量列表)，尝试广播或报错
                # 这里做一个简单的容错：如果只有一个值，广播；否则报错
                if rpm_array.size == 1:
                    rpm_array = np.full(N, rpm_array.item())
                else:
                    raise ValueError(f"rpm shape {rpm_array.shape} does not match signals shape {N}")

        # 提取特征
        # 由于每个样本的 RPM 可能不同，故障频率也不同，无法使用矩阵广播一次性计算掩码
        # 需要循环处理，或者使用更高级的广播技巧。为了代码清晰，这里使用循环处理每个样本。
        # 考虑到效率，我们可以先计算所有样本的故障频率，然后看是否可以向量化。
        
        # 预计算所有样本的故障频率: (N, 4)
        # BPFI, BPFO, BSF, FTF
        fr = rpm_array / 60.0
        cos_a = np.cos(np.deg2rad(self.alpha))
        ratio = (self.d / self.D) * cos_a
        
        bpfi = (self.n / 2) * fr * (1 + ratio)
        bpfo = (self.n / 2) * fr * (1 - ratio)
        bsf  = (self.D / (2 * self.d)) * fr * (1 - ratio**2)
        ftf  = (1 / 2) * fr * (1 - ratio)
        
        fault_freqs_arr = np.stack([bpfi, bpfo, bsf, ftf], axis=1) # (N, 4)
        
        harmonics = [1, 2, 3]
        bandwidth = 5.0 # Hz
        
        # features: (N, 4 * 3) = (N, 12)
        features = np.zeros((N, len(fault_freqs_arr[0]) * len(harmonics)))
        
        # 这种逐样本循环在 Python 中较慢，但对于几万个样本还可以接受。
        # 为了加速，我们可以利用 numpy 的广播。
        # freqs: (F,)
        # target_freqs: (N, 12)
        
        col_idx = 0
        for f_idx in range(4): # 4个故障频率类型
            for h in harmonics:
                # 当前针对所有样本的目标频率 (N,)
                targets = fault_freqs_arr[:, f_idx] * h
                
                # 我们需要对每个样本 i，找到 freqs 中在 [targets[i]-bw, targets[i]+bw] 范围内的最大值
                # 这是一个逐行的操作。
                # 优化：由于 bandwidth 很小，我们可以近似认为频率索引偏移不大。
                # 但为了准确，我们还是用简单的循环实现核心逻辑，或者使用掩码矩阵。
                
                # 掩码矩阵法 (内存消耗大): (N, F) -> 3万 * 1025 * bool ~ 30MB，完全可以。
                # freqs (1, F)
                # targets (N, 1)
                # mask = (freqs >= targets - bw) & (freqs <= targets + bw)
                
                t_col = targets.reshape(-1, 1)
                f_row = freqs.reshape(1, -1)
                
                # 利用广播生成掩码
                mask = (f_row >= t_col - bandwidth) & (f_row <= t_col + bandwidth) # (N, F)
                
                # 应用掩码提取最大值
                # 注意：如果掩码全为 False (没找到)，max 会报错或需要处理
                # 我们可以将不关心的区域设为 -1，然后取 max
                
                masked_fft = np.where(mask, fft_vals, -1.0)
                max_vals = np.max(masked_fft, axis=1)
                
                # 如果最大值是 -1，说明没找到，设为 0
                max_vals = np.where(max_vals == -1.0, 0.0, max_vals)
                
                features[:, col_idx] = max_vals
                col_idx += 1
                
        return features

class SoundMetricsExtractor(FeatureExtractor):
    """
    声音密度和声音能量曲线提取器
    从预处理的声音能量密度曲线数据中提取特征
    """
    def __init__(self, sound_data_dir='声音能量曲线数据', use_fallback=True):
        """
        Args:
            sound_data_dir: 声音数据目录
            use_fallback: 当声音数据不可用时，是否使用振动信号的 PSD 作为后备方案
        """
        self.use_fallback = use_fallback
        try:
            import sys
            import os
            # 添加 tools 目录到路径
            tools_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools')
            if tools_path not in sys.path:
                sys.path.insert(0, tools_path)
            
            from load_sound import SoundDataLoader
            self.sound_loader = SoundDataLoader(sound_data_dir)
            available_samples = len(self.sound_loader.sheet_to_file)
            print(f"Sound data loader initialized with {available_samples} samples across {len(self.sound_loader.file_mapping)} xlsx files")
        except Exception as e:
            print(f"Warning: Failed to initialize sound loader: {e}")
            self.sound_loader = None
            
    def extract(self, signals, fs, rpm=None, metadata=None):
        """
        Args:
            signals: (N, L) 原始振动信号（用于后备方案）
            fs: 采样率
            rpm: 转速（未使用）
            metadata: 元数据列表，包含每个样本的文件名信息
        Returns:
            features: (N, D) 特征矩阵
        """
        N = signals.shape[0]
        
        # 如果没有声音加载器或元数据，使用后备方案
        if self.sound_loader is None or metadata is None:
            if self.use_fallback:
                return self._extract_fallback(signals, fs)
            else:
                raise ValueError("Sound loader not available and fallback disabled")
        
        # 尝试从声音数据中提取
        features_list = []
        fallback_count = 0
        
        for i in range(N):
            # 从元数据获取文件名
            if i < len(metadata):
                filename = metadata[i].get('filename', '')
                # 移除 .mat 扩展名
                base_name = filename.replace('.mat', '')
                
                # 尝试加载声音曲线
                curves = self.sound_loader.load_sound_curves(base_name)
                
                if curves is not None:
                    # 提取统计特征从声音曲线
                    feat = self._extract_from_curves(curves)
                    features_list.append(feat)
                else:
                    # 使用后备方案
                    if self.use_fallback:
                        feat = self._extract_fallback(signals[i:i+1], fs).flatten()
                        features_list.append(feat)
                        fallback_count += 1
                    else:
                        # 用零填充
                        features_list.append(np.zeros(22))  # 默认22个特征
                        fallback_count += 1
            else:
                # 没有元数据，使用后备
                if self.use_fallback:
                    feat = self._extract_fallback(signals[i:i+1], fs).flatten()
                    features_list.append(feat)
                    fallback_count += 1
                else:
                    features_list.append(np.zeros(22))
                    fallback_count += 1
        
        if fallback_count > 0:
            print(f"Note: {fallback_count}/{N} samples used fallback (PSD-based) features")
        
        return np.array(features_list)
    
    def _extract_from_curves(self, curves):
        """
        从声音曲线中提取统计特征
        
        Args:
            curves: dict with 'frequency', 'volume', 'density'
        Returns:
            features: 1D array of features
        """
        freq = curves['frequency']
        volume = curves['volume']
        density = curves['density']
        
        features = []
        
        # 音量特征 (11个)
        features.extend([
            np.mean(volume),      # 平均音量
            np.std(volume),       # 音量标准差
            np.max(volume),       # 最大音量
            np.min(volume),       # 最小音量
            np.percentile(volume, 25),   # 25分位
            np.percentile(volume, 50),   # 中位数
            np.percentile(volume, 75),   # 75分位
            np.percentile(volume, 90),   # 90分位
            np.sum(volume > np.mean(volume)),  # 高于均值的点数
            float(np.argmax(volume)),    # 最大音量的索引
            np.ptp(volume),       # 峰峰值 (peak-to-peak)
        ])
        
        # 密度特征 (11个)
        features.extend([
            np.mean(density),     # 平均密度
            np.std(density),      # 密度标准差
            np.max(density),      # 最大密度
            np.min(density),      # 最小密度
            np.percentile(density, 25),
            np.percentile(density, 50),
            np.percentile(density, 75),
            np.percentile(density, 90),
            np.sum(density > np.mean(density)),
            float(np.argmax(density)),   # 最大密度的索引
            np.ptp(density),      # 峰峰值
        ])
        
        return np.array(features)
    
    def _extract_fallback(self, signals, fs):
        """
        后备方案：使用振动信号的 PSD 和 RMS
        
        Args:
            signals: (N, L) 或 (1, L)
        Returns:
            features: (N, 22) - 匹配声音特征维度
        """
        # RMS
        rms = np.sqrt(np.mean(signals**2, axis=1)).reshape(-1, 1)
        
        # PSD 统计特征（而不是完整曲线）
        f, Pxx = welch(signals, fs=fs, nperseg=256, axis=1)
        
        # 从 PSD 提取统计特征（模拟声音曲线的统计量）
        psd_mean = np.mean(Pxx, axis=1).reshape(-1, 1)
        psd_std = np.std(Pxx, axis=1).reshape(-1, 1)
        psd_max = np.max(Pxx, axis=1).reshape(-1, 1)
        psd_min = np.min(Pxx, axis=1).reshape(-1, 1)
        psd_p25 = np.percentile(Pxx, 25, axis=1).reshape(-1, 1)
        psd_p50 = np.percentile(Pxx, 50, axis=1).reshape(-1, 1)
        psd_p75 = np.percentile(Pxx, 75, axis=1).reshape(-1, 1)
        psd_p90 = np.percentile(Pxx, 90, axis=1).reshape(-1, 1)
        psd_above_mean = np.sum(Pxx > np.mean(Pxx, axis=1, keepdims=True), axis=1).reshape(-1, 1)
        psd_argmax = np.argmax(Pxx, axis=1).reshape(-1, 1).astype(float)
        psd_ptp = np.ptp(Pxx, axis=1).reshape(-1, 1)
        
        # 构造22维特征：11个模拟音量 + 11个模拟密度
        features = np.concatenate([
            psd_mean, psd_std, psd_max, psd_min, psd_p25, psd_p50, 
            psd_p75, psd_p90, psd_above_mean, psd_argmax, psd_ptp,  # 11个
            psd_mean * 0.8, psd_std * 0.9, psd_max * 0.7, psd_min * 1.1, 
            psd_p25 * 0.95, psd_p50 * 1.05, psd_p75 * 0.85, psd_p90 * 0.9,
            psd_above_mean * 0.8, psd_argmax, rms  # 再11个
        ], axis=1)
        
        return features

class HybridFeatureExtractor(FeatureExtractor):
    """
    混合特征提取器：结合 FFT (幅值+相位) 和 机理特征 (倍频能量)
    """
    def __init__(self):
        self.fft_extractor = FFTFeatureExtractor(include_phase=True)
        self.mechanism_extractor = MechanismFeatureExtractor()
        
    def extract(self, signals, fs, rpm=None):
        # 1. 提取 FFT 特征
        fft_features = self.fft_extractor.extract(signals, fs, rpm)
        
        # 2. 提取机理特征
        mech_features = self.mechanism_extractor.extract(signals, fs, rpm)
        
        # fft_features: (N, D1)
        # mech_features: (N, D2)
        # result: (N, D1 + D2)
        return np.concatenate([fft_features, mech_features], axis=1)
