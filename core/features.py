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
    (目前使用振动信号的 PSD 和 RMS 作为代理，待姚飞提供具体转换算法后替换)
    """
    def extract(self, signals, fs, rpm=None):
        # 1. 声音能量 (Energy/RMS)
        # 简单的 RMS 计算
        rms = np.sqrt(np.mean(signals**2, axis=1)).reshape(-1, 1)
        
        # 2. 声音密度 (Spectral Density)
        # 使用 Welch 方法计算功率谱密度 (PSD)
        # 返回的是 PSD 的统计特征，或者整个 PSD 曲线
        # 这里为了作为输入曲线，我们返回 PSD 曲线
        f, Pxx = welch(signals, fs=fs, nperseg=256, axis=1)
        
        # 将 RMS 和 PSD 拼接
        # 注意：RMS 是标量，PSD 是向量。
        # 如果模型需要曲线输入，通常主要使用 PSD。RMS 可以作为一个额外的通道或特征。
        # 这里我们返回 PSD 曲线作为主要特征
        
        return Pxx

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
        
        # 3. 拼接特征
        # fft_features: (N, D1)
        # mech_features: (N, D2)
        # result: (N, D1 + D2)
        return np.concatenate([fft_features, mech_features], axis=1)
