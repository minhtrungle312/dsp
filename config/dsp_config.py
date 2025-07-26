"""
DSP Configuration Module - Cấu hình DSP
==================================

File này chứa các tham số cấu hình cho hệ thống xử lý tín hiệu số (DSP).
Các tham số này điều khiển tất cả các bước xử lý audio từ tiền xử lý đến hậu xử lý.

Author: DSP Team
Date: 2025
"""

class DSPConfiguration:
    """
    Lớp cấu hình toàn diện cho DSP
    Comprehensive DSP configuration parameters
    
    Chứa tất cả các tham số cần thiết cho:
    - Cấu hình chính (sample rate, bit depth, channels)
    - Tiền xử lý (high-pass, low-pass, normalization)
    - STFT (Short-Time Fourier Transform)
    - Spectral subtraction (trừ phổ)
    - Harmonic enhancement (cải thiện âm hài)
    - Noise gate (cổng nhiễu)
    - Wiener filter (bộ lọc Wiener)
    - Compressor (nén âm thanh)
    - Optimization (tối ưu hóa)
    """
    
    def __init__(self):
        """
        Khởi tạo các tham số cấu hình mặc định
        Initialize default configuration parameters
        """
        
        # Cấu hình chính - Master configuration
        self.master_config = {
            'sample_rate': 22050,          # Tần số lấy mẫu (Hz)
            'bit_depth': 16,               # Độ sâu bit
            'channels': 1,                 # Số kênh (1 = mono, 2 = stereo)
            'preprocessing': {
                'high_pass_freq': 80,      # Tần số high-pass filter (Hz)
                'low_pass_freq': 8000,     # Tần số low-pass filter (Hz)
                'normalize_input': True    # Có chuẩn hóa đầu vào không
            },
        }
        
        # Cấu hình STFT - Short-Time Fourier Transform
        self.stft_config = {
            'n_fft': 2048,                # Kích thước FFT
            'hop_length': 512,            # Bước nhảy giữa các frame
            'window': 'hann',             # Loại cửa sổ (hann, hamming, blackman)
            'center': True,               # Có center padding không
            'pad_mode': 'reflect'         # Chế độ padding
        }
        
        # Tham số Spectral Subtraction - Trừ phổ để giảm nhiễu
        self.spectral_subtraction_config = {
            'alpha': 2.0,                 # Giảm xuống từ 2.0
            'beta': 0.1,                 # Tăng lên từ 0.001
            'noise_estimation_frames': 50, # Giảm xuống để responsive hơn
            'smoothing_factor': 0.7,      # Giảm xuống để giữ chi tiết
            'frequency_smoothing': 3,     # OK
            'snr_threshold_high': 15,     # Ngưỡng SNR cao
            'snr_threshold_low': 0,       # Ngưỡng SNR thấp
        }
        
        # Cấu hình Harmonic Enhancement - Cải thiện âm hài
        self.harmonic_enhancement_config = {
            'algorithm': 'spectral_harmonic_ratio',  # Thuật toán sử dụng
            'fundamental_freq_range': [80, 1000],    # Dải tần số cơ bản (Hz)
            'harmonics_count': 10,                    # Số âm hài cần cải thiện
            'harmonic_threshold': 0.4,               # Ngưỡng phát hiện âm hài
            'enhancement_factor': 4,               # Hệ số cải thiện âm hài
            'spectral_envelope_smoothing': 0.95      # Làm mịn envelope phổ
        }
        
        # Cấu hình Noise Gate - Cổng nhiễu
        self.noise_gate_config = {
            'threshold_db': -35,          # Ngưỡng mở cổng (dB)
            'ratio': 10,                  # Tỷ lệ nén
            'attack_time_ms': 1,          # Thời gian mở cổng (ms)
            'release_time_ms': 100,       # Thời gian đóng cổng (ms)
            'hold_time_ms': 10,           # Thời gian giữ cổng mở (ms)
            'lookahead_ms': 5,            # Thời gian nhìn trước (ms)
            'rms_window_size': 1024,      # Kích thước cửa sổ RMS
            'smoothing_factor': 0.95      # Hệ số làm mịn
        }
        
        # Cấu hình Wiener Filter - Bộ lọc Wiener
        self.wiener_filter_config = {
            'noise_estimation_method': 'vad_based',  # Phương pháp ước tính nhiễu
            'smoothing_constant': 0.98,              # Hằng số làm mịn
            'min_gain': 0.1,                         # Gain tối thiểu
            'max_gain': 3.0,                         # Gain tối đa
            'frequency_smoothing': 5,                # Làm mịn tần số
            'snr_floor_db': -20                      # Sàn SNR (dB)
        }
        
        # Cấu hình Compressor - Nén âm thanh
        self.compressor_config = {
            'threshold_db': -20,          # Ngưỡng nén (dB)
            'ratio': 4.0,                 # Tỷ lệ nén
            'attack_time_ms': 5,          # Thời gian tấn công (ms)
            'release_time_ms': 50,        # Thời gian thả (ms)
            'knee_width_db': 2,           # Độ rộng knee (dB)
            'makeup_gain_db': 3,          # Gain bù (dB)
            'lookahead_ms': 10,           # Thời gian nhìn trước (ms)
            'rms_window_ms': 10           # Cửa sổ RMS (ms)
        }
        
        # Cấu hình Optimization - Tối ưu hóa
        self.optimization_config = {
            'block_processing': True,     # Có xử lý theo khối không
            'block_size': 8192,           # Kích thước khối
            'overlap_blocks': 0.5,        # Tỷ lệ chồng lấp khối
            'multithreading': True,       # Có đa luồng không
            'gpu_acceleration': False,    # Có tăng tốc GPU không
            'memory_efficient': True      # Có tối ưu bộ nhớ không
        }
    
    @classmethod
    def from_json(cls, json_path: str):
        """
        Load configuration from JSON file (for main.py compatibility).
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            DSPConfiguration: Configuration object
        """
        import json
        
        try:
            print(f"Loading configuration from: {json_path}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Create instance with default values
            instance = cls()
            
            # Update master config if provided
            if 'master_config' in config_data:
                instance.master_config.update(config_data['master_config'])
                print(f"  ✓ Updated master config")
            
            # Update STFT config if provided
            if 'stft_config' in config_data:
                instance.stft_config.update(config_data['stft_config'])
                print(f"  ✓ Updated STFT config")
            
            # Update other configs
            for config_name in ['spectral_subtraction_config', 'harmonic_enhancement_config', 'noise_gate_config']:
                if config_name in config_data:
                    setattr(instance, config_name, config_data[config_name])
                    print(f"  ✓ Updated {config_name}")
            
            print(f"  ✓ Configuration loaded successfully")
            return instance
            
        except FileNotFoundError:
            print(f"  ⚠ Config file not found: {json_path}")
            print(f"  Using default configuration")
            return cls()
        except json.JSONDecodeError as e:
            print(f"  ⚠ Invalid JSON format: {e}")
            print(f"  Using default configuration")
            return cls()
        except Exception as e:
            print(f"  ⚠ Error loading config: {e}")
            print(f"  Using default configuration")
            return cls()

    def to_json(self, json_path: str) -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON configuration file
            
        Returns:
            bool: True if successful
        """
        import json
        
        try:
            config_data = {
                'master_config': self.master_config,
                'stft_config': self.stft_config,
                'spectral_subtraction_config': self.spectral_subtraction_config,
                'harmonic_enhancement_config': self.harmonic_enhancement_config,
                'noise_gate_config': self.noise_gate_config
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Configuration saved to: {json_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error saving config: {e}")
            return False

    def validate(self) -> bool:
        """
        Validate configuration parameters (for main.py compatibility).
        
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Validate sample rate
            sr = self.master_config.get('sample_rate', 22050)
            if sr <= 0 or sr > 192000:
                print(f"⚠ Invalid sample rate: {sr}")
                return False
            
            # Validate FFT size
            n_fft = self.stft_config.get('n_fft', 2048)
            if n_fft <= 0 or (n_fft & (n_fft - 1)) != 0:  # Check if power of 2
                print(f"⚠ FFT size should be power of 2: {n_fft}")
                return False
            
            # Validate hop length
            hop_length = self.stft_config.get('hop_length', 512)
            if hop_length <= 0 or hop_length >= n_fft:
                print(f"⚠ Invalid hop length: {hop_length}")
                return False
            
            print("✓ Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"⚠ Configuration validation error: {e}")
            return False
    
    def get_summary(self) -> str:
        """
        Get a summary of current configuration (for main.py compatibility).
        
        Returns:
            str: Configuration summary
        """
        summary = f"""
DSP Configuration Summary:
==========================
Master Config:
  - Sample Rate: {self.master_config.get('sample_rate', 22050)} Hz
  - Channels: {self.master_config.get('channels', 1)}
  - Chunk Size: {self.master_config.get('chunk_size', 1024)}

STFT Config:
  - FFT Size: {self.stft_config.get('n_fft', 2048)}
  - Hop Length: {self.stft_config.get('hop_length', 512)}
  - Window: {self.stft_config.get('window', 'hann')}

Noise Reduction:
  - Spectral Subtraction Alpha: {self.spectral_subtraction_config.get('alpha', 2.0)}
  - Noise Estimation Frames: {self.spectral_subtraction_config.get('noise_estimation_frames', 50)}

Harmonic Enhancement:
  - Algorithm: {self.harmonic_enhancement_config.get('algorithm', 'spectral_harmonic_ratio')}
  - Enhancement Factor: {self.harmonic_enhancement_config.get('enhancement_factor', 1.5)}

Noise Gate:
  - Threshold: {self.noise_gate_config.get('threshold_db', -35)} dB
  - Ratio: {self.noise_gate_config.get('ratio', 10)}
=========================="""
        return summary

    def print_config(self):
        """
        In ra cấu hình hiện tại
        Print current configuration
        """
        print("=" * 50)
        print("DSP CONFIGURATION SUMMARY")
        print("=" * 50)
        
        summary = self.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print("=" * 50)

# Tạo instance mặc định
default_config = DSPConfiguration()

if __name__ == "__main__":
    # Test cấu hình
    config = DSPConfiguration()
    config.print_config()
    config.validate_config()
