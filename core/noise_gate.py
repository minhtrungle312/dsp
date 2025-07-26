"""
Noise Gate Processing Module - Module xử lý cổng nhiễu
===================================================

Module này chứa lớp NoiseGateProcessor để xử lý cổng nhiễu và nén âm thanh:
- Advanced noise gate with envelope detection
- Dynamic range compression
- RMS envelope calculation
- Gate control signal generation

Author: DSP Team
Date: 2025
"""

import numpy as np
import os
from matplotlib import pyplot as plt

class NoiseGateProcessor:
    """
    Lớp xử lý cổng nhiễu
    Noise gate processor for audio signal gating and compression
    
    Chứa các phương thức:
    - advanced_noise_gate: Cổng nhiễu tiên tiến
    - dynamic_range_compression: Nén dải động
    - calculate_rms_envelope: Tính envelope RMS
    - generate_gate_control: Tạo tín hiệu điều khiển cổng
    """
    
    def __init__(self, config):
        """
        Khởi tạo NoiseGateProcessor
        Initialize NoiseGateProcessor with configuration
        
        Args:
            config: Đối tượng cấu hình DSP
        """
        self.config = config
        self.sr = config.master_config['sample_rate']  # Tần số lấy mẫu
        
        # Setup output directory for charts
        self.chart_dir = "./output/chart"
        os.makedirs(self.chart_dir, exist_ok=True)
        
        print(f"NoiseGateProcessor initialized with sr={self.sr}Hz")
    
    def calculate_rms_envelope(self, audio, window_size):
        """
        Tính envelope RMS với cửa sổ chồng lấp
        Calculate RMS envelope with overlapping windows
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            window_size: Kích thước cửa sổ RMS
            
        Returns:
            numpy.ndarray: Envelope RMS
        """
        hop_size = window_size // 4  # Chồng lấp 75%
        rms_values = []
        
        # Tính RMS cho từng cửa sổ
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))  # Root Mean Square
            rms_values.append(rms)
        
        # Nội suy về độ dài gốc
        if len(rms_values) > 0:
            rms_envelope = np.interp(
                np.arange(len(audio)),                          # Điểm nội suy
                np.arange(0, len(audio), hop_size)[:len(rms_values)],  # Điểm gốc
                rms_values                                       # Giá trị RMS
            )
        else:
            rms_envelope = np.zeros(len(audio))
        
        return rms_envelope
    
    def generate_gate_control(self, rms_db, config):
        """
        Tạo tín hiệu điều khiển cổng với attack/release
        Generate gate control signal with attack/release
        
        Args:
            rms_db: RMS envelope tính theo dB
            config: Cấu hình noise gate
            
        Returns:
            numpy.ndarray: Tín hiệu điều khiển cổng (0-1)
        """
        # Xác định trạng thái cổng dựa trên ngưỡng
        gate_state = rms_db > config['threshold_db']
        
        # Tính các hằng số thời gian theo samples
        attack_samples = int(config['attack_time_ms'] * self.sr / 1000)
        release_samples = int(config['release_time_ms'] * self.sr / 1000)
        hold_samples = int(config['hold_time_ms'] * self.sr / 1000)
        
        # Máy trạng thái cho điều khiển cổng
        gate_control = np.zeros_like(rms_db)
        current_state = 0  # 0: đóng, 1: đang mở, 2: mở, 3: đang đóng
        hold_counter = 0
        ramp_position = 0
        
        for i in range(len(rms_db)):
            if gate_state[i]:  # Tín hiệu trên ngưỡng
                if current_state == 0:  # Bắt đầu mở cổng
                    current_state = 1
                    ramp_position = 0
                elif current_state == 1:  # Tiếp tục mở cổng
                    ramp_position += 1
                    if ramp_position >= attack_samples:
                        current_state = 2  # Mở hoàn toàn
                elif current_state == 3:  # Dừng đóng cổng
                    current_state = 2
                hold_counter = hold_samples  # Reset hold counter
                
            else:  # Tín hiệu dưới ngưỡng
                if current_state == 2:  # Bắt đầu hold
                    hold_counter -= 1
                    if hold_counter <= 0:
                        current_state = 3  # Bắt đầu đóng cổng
                        ramp_position = 0
                elif current_state == 3:  # Tiếp tục đóng cổng
                    ramp_position += 1
                    if ramp_position >= release_samples:
                        current_state = 0  # Đóng hoàn toàn
            
            # Tính giá trị cổng
            if current_state == 0:  # Đóng
                gate_control[i] = 0
            elif current_state == 1:  # Đang mở
                gate_control[i] = ramp_position / attack_samples if attack_samples > 0 else 1
            elif current_state == 2:  # Mở
                gate_control[i] = 1
            elif current_state == 3:  # Đang đóng
                gate_control[i] = 1 - (ramp_position / release_samples) if release_samples > 0 else 0
        
        return gate_control
    
    def advanced_noise_gate(self, audio):
        """
        Cổng nhiễu tiên tiến với phát hiện envelope
        Advanced noise gate with envelope detection
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            
        Returns:
            numpy.ndarray: Tín hiệu sau khi qua cổng nhiễu
        """
        config = self.config.noise_gate_config
        
        # Tính envelope RMS
        rms_envelope = self.calculate_rms_envelope(audio, config['rms_window_size'])
        
        # Chuyển đổi sang dB
        rms_db = 20 * np.log10(rms_envelope + 1e-10)
        
        # Tạo tín hiệu điều khiển cổng
        gate_control = self.generate_gate_control(rms_db, config)
        
        # Áp dụng cổng với làm mịn
        smoothing_factor = config['smoothing_factor']
        smoothed_gate = np.zeros_like(gate_control)
        smoothed_gate[0] = gate_control[0]
        
        # Làm mịn tín hiệu điều khiển
        for i in range(1, len(gate_control)):
            smoothed_gate[i] = smoothing_factor * smoothed_gate[i-1] + \
                              (1 - smoothing_factor) * gate_control[i]
        
        # Áp dụng cổng lên tín hiệu
        gated_audio = audio * smoothed_gate
        
        # Tạo chart cho RMS Envelope và Gate Control
        plt.figure(figsize=(10, 4))
        times = np.linspace(0, len(rms_db) * 512 / self.sr, len(rms_db))
        plt.plot(times, rms_db, label='RMS Envelope (dB)')
        plt.plot(times, gate_control * np.max(rms_db), label='Gate Control', linestyle='--')
        plt.plot(times, smoothed_gate * np.max(rms_db), label='Smoothed Gate', linestyle='-.')
        plt.axhline(y=config['threshold_db'], color='r', linestyle=':', label='Ngưỡng')
        plt.title('RMS Envelope và Gate Control')
        plt.xlabel('Thời gian (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig(os.path.join(self.chart_dir, '2_6_noise_gating_control.png'))
        plt.close()
        
        return gated_audio
    
    def dynamic_range_compression(self, audio):
        """
        Áp dụng nén dải động
        Apply dynamic range compression
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            
        Returns:
            numpy.ndarray: Tín hiệu sau khi nén
        """
        config = self.config.compressor_config
        
        # Tính envelope cho compressor
        window_size = int(config['rms_window_ms'] * self.sr / 1000)
        envelope = self.calculate_rms_envelope(audio, window_size)
        
        # Chuyển đổi sang dB
        envelope_db = 20 * np.log10(envelope + 1e-10)
        
        # Áp dụng nén với soft knee
        compressed_db = self._apply_compression_curve(envelope_db, config)
        
        # Thêm makeup gain
        compressed_db += config['makeup_gain_db']
        
        # Chuyển đổi về linear và tính gain
        gain = 10**((compressed_db - envelope_db) / 20)
        
        # Làm mịn gain để tránh clicking
        smoothed_gain = self._smooth_gain(gain, config)
        
        # Áp dụng gain lên tín hiệu
        compressed_audio = audio * smoothed_gain
        
        return compressed_audio
    
    def _apply_compression_curve(self, envelope_db, config):
        """
        Áp dụng đường cong nén với soft knee
        Apply compression curve with soft knee
        
        Args:
            envelope_db: Envelope tính theo dB
            config: Cấu hình compressor
            
        Returns:
            numpy.ndarray: Envelope sau khi nén
        """
        threshold_db = config['threshold_db']
        ratio = config['ratio']
        knee_width = config['knee_width_db']
        
        # Tính soft knee
        knee_start = threshold_db - knee_width / 2
        knee_end = threshold_db + knee_width / 2
        
        compressed_db = np.zeros_like(envelope_db)
        
        for i, level in enumerate(envelope_db):
            if level <= knee_start:
                # Dưới knee: không nén
                compressed_db[i] = level
            elif level >= knee_end:
                # Trên knee: nén đầy đủ
                compressed_db[i] = threshold_db + (level - threshold_db) / ratio
            else:
                # Trong knee: nén mềm
                knee_ratio = (level - knee_start) / knee_width
                soft_ratio = 1 + (ratio - 1) * knee_ratio
                compressed_db[i] = threshold_db + (level - threshold_db) / soft_ratio
        
        return compressed_db
    
    def _smooth_gain(self, gain, config):
        """
        Làm mịn gain để tránh clicking
        Smooth gain to avoid clicking
        
        Args:
            gain: Gain array
            config: Cấu hình compressor
            
        Returns:
            numpy.ndarray: Gain sau khi làm mịn
        """
        # Tính attack và release time constants
        attack_samples = int(config['attack_time_ms'] * self.sr / 1000)
        release_samples = int(config['release_time_ms'] * self.sr / 1000)
        
        # Tính smoothing coefficients
        attack_coeff = np.exp(-1.0 / attack_samples) if attack_samples > 0 else 0
        release_coeff = np.exp(-1.0 / release_samples) if release_samples > 0 else 0
        
        smoothed_gain = np.zeros_like(gain)
        smoothed_gain[0] = gain[0]
        
        for i in range(1, len(gain)):
            if gain[i] < smoothed_gain[i-1]:  # Gain giảm (attack)
                smoothed_gain[i] = attack_coeff * smoothed_gain[i-1] + \
                                  (1 - attack_coeff) * gain[i]
            else:  # Gain tăng (release)
                smoothed_gain[i] = release_coeff * smoothed_gain[i-1] + \
                                  (1 - release_coeff) * gain[i]
        
        return smoothed_gain
    
    def multiband_processing(self, audio, bands=[(0, 200), (200, 2000), (2000, 8000)]):
        """
        Xử lý đa băng tần
        Multiband processing for different frequency ranges
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            bands: Danh sách các băng tần (freq_low, freq_high)
            
        Returns:
            numpy.ndarray: Tín hiệu sau khi xử lý đa băng
        """
        from scipy.signal import butter, filtfilt
        
        processed_bands = []
        
        for low_freq, high_freq in bands:
            # Tạo bộ lọc băng thông
            nyquist = self.sr / 2
            low_norm = max(low_freq / nyquist, 0.01)  # Tránh 0
            high_norm = min(high_freq / nyquist, 0.99)  # Tránh 1
            
            if low_norm < high_norm:
                # Bộ lọc băng thông
                b, a = butter(4, [low_norm, high_norm], btype='band')
                band_audio = filtfilt(b, a, audio)
                
                # Xử lý cổng nhiễu cho băng này
                gated_band = self.advanced_noise_gate(band_audio)
                processed_bands.append(gated_band)
            else:
                # Băng tần không hợp lệ
                processed_bands.append(np.zeros_like(audio))
        
        # Kết hợp các băng
        combined_audio = np.sum(processed_bands, axis=0)
        
        return combined_audio

if __name__ == "__main__":
    # Test code
    from config.dsp_config import DSPConfiguration
    
    config = DSPConfiguration()
    processor = NoiseGateProcessor(config)
    
    # Tạo tín hiệu test với noise burst
    test_audio = np.random.randn(22050) * 0.1  # Low level noise
    test_audio[5000:10000] = np.random.randn(5000) * 0.5  # High level signal
    
    # Xử lý
    gated = processor.advanced_noise_gate(test_audio)
    compressed = processor.dynamic_range_compression(gated)
    
    print(f"Original audio shape: {test_audio.shape}")
    print(f"Gated audio shape: {gated.shape}")
    print(f"Compressed audio shape: {compressed.shape}")
    print("NoiseGateProcessor test completed successfully!")
