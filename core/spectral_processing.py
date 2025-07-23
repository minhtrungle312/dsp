"""
Spectral Processing Module - Module xử lý phổ tín hiệu
====================================================

Module này chứa lớp SpectralProcessor để xử lý phổ tín hiệu:
- Enhanced spectral subtraction (trừ phổ cải tiến)
- Fundamental frequency estimation (ước tính tần số cơ bản)
- Harmonic strength calculation (tính toán cường độ âm hài)
- Temporal and frequency smoothing (làm mịn theo thời gian và tần số)

Author: DSP Team
Date: 2025
"""

import numpy as np
from scipy.signal import butter, filtfilt

class SpectralProcessor:
    """
    Lớp xử lý phổ tín hiệu
    Spectral processing class for audio signal analysis and enhancement
    
    Chứa các phương thức:
    - enhanced_spectral_subtraction: Trừ phổ cải tiến
    - estimate_fundamental_frequency: Ước tính tần số cơ bản
    - calculate_harmonic_strength: Tính cường độ âm hài
    - temporal_smoothing: Làm mịn theo thời gian
    - frequency_smoothing: Làm mịn theo tần số
    """
    
    def __init__(self, config):
        """
        Khởi tạo SpectralProcessor
        Initialize SpectralProcessor with configuration
        
        Args:
            config: Đối tượng cấu hình DSP
        """
        self.config = config
        self.sr = config.master_config['sample_rate']  # Tần số lấy mẫu
        self.n_fft = config.stft_config['n_fft']       # Kích thước FFT
        
        print(f"SpectralProcessor initialized with sr={self.sr}Hz, n_fft={self.n_fft}")
    
    def estimate_fundamental_frequency(self, magnitude):
        """
        Ước tính tần số cơ bản sử dụng autocorrelation
        Estimate fundamental frequency using autocorrelation
        
        Args:
            magnitude: Ma trận magnitude của STFT
            
        Returns:
            numpy.ndarray: Mảng tần số cơ bản theo thời gian
        """
        f0_track = []
        
        # Xử lý từng frame
        for frame_idx in range(magnitude.shape[1]):
            frame = magnitude[:, frame_idx]
            
            # Phương pháp autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Lấy phần dương
            
            # Tìm peak (bỏ qua DC component)
            min_period = int(self.sr / 1000)  # Tần số tối đa 1000 Hz
            max_period = int(self.sr / 80)    # Tần số tối thiểu 80 Hz
            
            if len(autocorr) > max_period:
                # Tìm peak trong khoảng hợp lệ
                peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
                f0 = self.sr / peak_idx if peak_idx > 0 else 0
            else:
                f0 = 0
            
            f0_track.append(f0)
        
        return np.array(f0_track)
    
    def calculate_harmonic_strength(self, magnitude_frame, bin_idx, window_size=3):
        """
        Tính cường độ âm hài sử dụng phát hiện peak phổ
        Calculate harmonic strength using spectral peak detection
        
        Args:
            magnitude_frame: Frame magnitude
            bin_idx: Chỉ số bin tần số
            window_size: Kích thước cửa sổ xung quanh peak
            
        Returns:
            float: Cường độ âm hài
        """
        # Xác định vùng cục bộ xung quanh bin
        start_idx = max(0, bin_idx - window_size)
        end_idx = min(len(magnitude_frame), bin_idx + window_size + 1)
        
        local_region = magnitude_frame[start_idx:end_idx]
        peak_value = magnitude_frame[bin_idx]
        
        # Cường độ âm hài = peak / (trung bình các bin xung quanh)
        surrounding_mean = (np.sum(local_region) - peak_value) / (len(local_region) - 1)
        
        return peak_value / (surrounding_mean + 1e-10)
    
    def temporal_smoothing(self, signal, smoothing_constant):
        """
        Áp dụng làm mịn theo thời gian
        Apply temporal smoothing
        
        Args:
            signal: Tín hiệu đầu vào
            smoothing_constant: Hằng số làm mịn
            
        Returns:
            numpy.ndarray: Tín hiệu sau khi làm mịn
        """
        smoothed = np.zeros_like(signal)
        smoothed[:, 0] = signal[:, 0]  # Frame đầu tiên giữ nguyên
        
        # Áp dụng bộ lọc IIR đơn giản
        for i in range(1, signal.shape[1]):
            smoothed[:, i] = smoothing_constant * smoothed[:, i-1] + \
                            (1 - smoothing_constant) * signal[:, i]
        
        return smoothed
    
    def frequency_smoothing(self, signal, window_size):
        """
        Áp dụng làm mịn trong miền tần số
        Apply frequency domain smoothing
        
        Args:
            signal: Tín hiệu đầu vào
            window_size: Kích thước cửa sổ làm mịn
            
        Returns:
            numpy.ndarray: Tín hiệu sau khi làm mịn
        """
        smoothed = np.zeros_like(signal)
        
        # Áp dụng bộ lọc trung bình di động
        for i in range(signal.shape[1]):
            frame = signal[:, i]
            kernel = np.ones(window_size) / window_size  # Kernel trung bình
            smoothed[:, i] = np.convolve(frame, kernel, mode='same')
        
        return smoothed
    
    def preprocess_audio(self, audio):
        """
        Tiền xử lý âm thanh
        Preprocess audio signal
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            
        Returns:
            numpy.ndarray: Tín hiệu sau tiền xử lý
        """
        config = self.config.master_config['preprocessing']
        
        # Bộ lọc thông cao - High-pass filter
        if config['high_pass_freq'] > 0:
            nyquist = self.sr / 2
            high_freq = config['high_pass_freq'] / nyquist
            if high_freq < 1.0:  # Tần số hợp lệ
                b, a = butter(4, high_freq, btype='high')
                audio = filtfilt(b, a, audio)
        
        # Bộ lọc thông thấp - Low-pass filter
        if config['low_pass_freq'] < self.sr / 2:
            nyquist = self.sr / 2
            low_freq = config['low_pass_freq'] / nyquist
            if low_freq < 1.0:  # Tần số hợp lệ
                b, a = butter(4, low_freq, btype='low')
                audio = filtfilt(b, a, audio)
        
        # Chuẩn hóa đầu vào - Normalize input
        if config['normalize_input']:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        
        return audio

if __name__ == "__main__":
    # Test code
    from config.dsp_config import DSPConfiguration
    
    config = DSPConfiguration()
    processor = SpectralProcessor(config)
