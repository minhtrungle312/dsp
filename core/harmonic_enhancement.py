"""
Harmonic Enhancement Module - Module cải thiện âm hài
=================================================

Module này chứa lớp HarmonicEnhancer để cải thiện chất lượng âm hài:
- Spectral harmonic enhancement (cải thiện âm hài phổ)
- Adaptive Wiener filtering (lọc Wiener thích ứng)
- Harmonic-percussive source separation (tách âm hài và percussion)

Author: DSP Team
Date: 2025
"""

from matplotlib import pyplot as plt
import numpy as np
import librosa
import os

class HarmonicEnhancer:
    """
    Lớp cải thiện âm hài
    Harmonic enhancement class for audio signal enhancement
    
    Chứa các phương thức:
    - spectral_harmonic_enhancement: Cải thiện âm hài phổ
    - adaptive_wiener_filter: Bộ lọc Wiener thích ứng
    - harmonic_percussive_separation: Tách âm hài và percussion
    """
    
    def __init__(self, config):
        """
        Khởi tạo HarmonicEnhancer
        Initialize HarmonicEnhancer with configuration
        
        Args:
            config: Đối tượng cấu hình DSP
        """
        self.config = config
        self.sr = config.master_config['sample_rate']  # Tần số lấy mẫu
        self.n_fft = config.stft_config['n_fft']       # Kích thước FFT
        
        # Setup output directory for charts
        self.chart_dir = "./output/chart"
        os.makedirs(self.chart_dir, exist_ok=True)
        
        print(f"HarmonicEnhancer initialized with sr={self.sr}Hz, n_fft={self.n_fft}")
    
    def spectral_harmonic_enhancement(self, stft_matrix, f0_track):
        """
        Cải thiện âm hài phổ sử dụng tỷ lệ âm hài
        Spectral harmonic enhancement using harmonic ratio
        
        Args:
            stft_matrix: Ma trận STFT phức
            f0_track: Mảng tần số cơ bản theo thời gian
            
        Returns:
            numpy.ndarray: STFT matrix sau khi cải thiện âm hài
        """
        config = self.config.harmonic_enhancement_config
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        
        # Tạo bản sao để sửa đổi
        enhanced_magnitude = magnitude.copy()
        
        # Xử lý từng frame
        for frame_idx in range(magnitude.shape[1]):
            f0 = f0_track[frame_idx]
            
            if f0 > 0:  # Nếu phát hiện được F0 hợp lệ
                # Tính toán vị trí các âm hài
                harmonics = [f0 * (i + 1) for i in range(config['harmonics_count'])]
                
                # Tìm bin tần số cho từng âm hài
                for harmonic_freq in harmonics:
                    if harmonic_freq < self.sr / 2:  # Trong dải Nyquist
                        # Tìm bin tần số gần nhất
                        bin_idx = int(harmonic_freq * self.n_fft / self.sr)
                        
                        if bin_idx < len(magnitude):
                            # Tính cường độ âm hài
                            harmonic_strength = self._calculate_harmonic_strength(
                                magnitude[:, frame_idx], bin_idx, window_size=3
                            )
                            
                            # Cải thiện nếu trên ngưỡng
                            if harmonic_strength > config['harmonic_threshold']:
                                enhanced_magnitude[bin_idx, frame_idx] *= config['enhancement_factor']
        
        # Tái tạo STFT với phase gốc
        return enhanced_magnitude * np.exp(1j * phase)
    
    def _calculate_harmonic_strength(self, magnitude_frame, bin_idx, window_size=3):
        """
        Tính cường độ âm hài (private method)
        Calculate harmonic strength using spectral peak detection
        
        Args:
            magnitude_frame: Frame magnitude
            bin_idx: Chỉ số bin tần số
            window_size: Kích thước cửa sổ
            
        Returns:
            float: Cường độ âm hài
        """
        start_idx = max(0, bin_idx - window_size)
        end_idx = min(len(magnitude_frame), bin_idx + window_size + 1)
        
        local_region = magnitude_frame[start_idx:end_idx]
        peak_value = magnitude_frame[bin_idx]
        
        # Cường độ âm hài = peak / (trung bình xung quanh)
        surrounding_mean = (np.sum(local_region) - peak_value) / (len(local_region) - 1)
        
        return peak_value / (surrounding_mean + 1e-10)
    
    def adaptive_wiener_filter(self, stft_matrix, noise_psd):
        """
        Bộ lọc Wiener thích ứng với ước tính SNR cải tiến
        Adaptive Wiener filter with improved SNR estimation
        
        Args:
            stft_matrix: Ma trận STFT phức
            noise_psd: Power spectral density của nhiễu
            
        Returns:
            numpy.ndarray: STFT matrix sau khi lọc
        """
        config = self.config.wiener_filter_config
        
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        
        # Ước tính signal PSD
        signal_psd = magnitude**2
        
        # Làm mịn noise PSD estimation
        smoothed_noise_psd = self._temporal_smoothing(
            noise_psd, config['smoothing_constant']
        )
        
        # Tính SNR
        snr_linear = signal_psd / (smoothed_noise_psd + 1e-10)
        snr_db = 10 * np.log10(snr_linear)
        
        # Áp dụng sàn SNR
        snr_db = np.maximum(snr_db, config['snr_floor_db'])
        snr_linear = 10**(snr_db / 10)
        
        # Tính Wiener gain
        wiener_gain = snr_linear / (snr_linear + 1)
        
        # Áp dụng giới hạn gain
        wiener_gain = np.clip(wiener_gain, config['min_gain'], config['max_gain'])
        
        # Làm mịn trong miền tần số
        smoothed_gain = self._frequency_smoothing(wiener_gain, config['frequency_smoothing'])
        
        # Áp dụng bộ lọc
        filtered_magnitude = magnitude * smoothed_gain
        
        # Tạo SNR chart
        plt.figure(figsize=(10, 4))
        times = np.linspace(0, len(snr_db[0]) * 512 / self.sr, len(snr_db[0]))
        plt.plot(times, np.mean(snr_db, axis=0))
        plt.title('SNR (dB) - Sau Wiener Filter')
        plt.xlabel('Thời gian (s)')
        plt.ylabel('SNR (dB)')
        plt.savefig(os.path.join(self.chart_dir, '2_4_wiener_snr.png'))
        plt.close()
        
        return filtered_magnitude * np.exp(1j * phase)
    
    def _temporal_smoothing(self, signal, smoothing_constant):
        """
        Làm mịn theo thời gian (private method)
        Apply temporal smoothing
        
        Args:
            signal: Tín hiệu đầu vào
            smoothing_constant: Hằng số làm mịn
            
        Returns:
            numpy.ndarray: Tín hiệu sau khi làm mịn
        """
        smoothed = np.zeros_like(signal)
        smoothed[:, 0] = signal[:, 0]
        
        for i in range(1, signal.shape[1]):
            smoothed[:, i] = smoothing_constant * smoothed[:, i-1] + \
                            (1 - smoothing_constant) * signal[:, i]
        
        return smoothed
    
    def _frequency_smoothing(self, signal, window_size):
        """
        Làm mịn trong miền tần số (private method)
        Apply frequency domain smoothing
        
        Args:
            signal: Tín hiệu đầu vào
            window_size: Kích thước cửa sổ
            
        Returns:
            numpy.ndarray: Tín hiệu sau khi làm mịn
        """
        smoothed = np.zeros_like(signal)
        
        for i in range(signal.shape[1]):
            frame = signal[:, i]
            kernel = np.ones(window_size) / window_size
            smoothed[:, i] = np.convolve(frame, kernel, mode='same')
        
        return smoothed
    
    def harmonic_percussive_separation(self, stft_matrix, margin=1.0):
        """
        Tách âm hài và percussion sử dụng librosa
        Harmonic-percussive source separation
        
        Args:
            stft_matrix: Ma trận STFT phức
            margin: Margin cho separation
            
        Returns:
            tuple: (harmonic_stft, percussive_stft)
        """
        try:
            # Sử dụng librosa để tách âm hài và percussion
            stft_harmonic, stft_percussive = librosa.decompose.hpss(
                stft_matrix, margin=margin
            )
            
            return stft_harmonic, stft_percussive
        
        except Exception as e:
            print(f"Error in HPSS: {e}")
            # Trả về original nếu có lỗi
            return stft_matrix, np.zeros_like(stft_matrix)
    
    def enhance_harmonics_with_hpss(self, stft_matrix, f0_track):
        """
        Cải thiện âm hài kết hợp với HPSS
        Enhance harmonics combined with HPSS
        
        Args:
            stft_matrix: Ma trận STFT phức
            f0_track: Mảng tần số cơ bản theo thời gian
            
        Returns:
            numpy.ndarray: STFT matrix sau khi cải thiện
        """
        # Tách âm hài và percussion
        harmonic_stft, percussive_stft = self.harmonic_percussive_separation(stft_matrix)
        
        # Cải thiện phần âm hài
        enhanced_harmonic = self.spectral_harmonic_enhancement(harmonic_stft, f0_track)
        
        # Kết hợp lại với tỷ lệ phù hợp
        config = self.config.harmonic_enhancement_config
        smoothing_factor = config['spectral_envelope_smoothing']
        
        # Trộn âm hài được cải thiện với percussive
        enhanced_stft = enhanced_harmonic * smoothing_factor + \
                       percussive_stft * (1 - smoothing_factor)
        
        return enhanced_stft

if __name__ == "__main__":
    # Test code
    from config.dsp_config import DSPConfiguration
    
    config = DSPConfiguration()
    enhancer = HarmonicEnhancer(config)
    
    # Tạo STFT test
    test_stft = np.random.randn(1025, 100) + 1j * np.random.randn(1025, 100)
    test_f0 = np.random.randint(80, 400, 100)  # F0 random
    
    enhanced = enhancer.spectral_harmonic_enhancement(test_stft, test_f0)
    
    print(f"Original STFT shape: {test_stft.shape}")
    print(f"Enhanced STFT shape: {enhanced.shape}")
    print("HarmonicEnhancer test completed successfully!")
