"""
Enhanced Post Processor Module - Module hậu xử lý cải tiến
========================================================

Module này chứa lớp EnhancedPostProcessor để hậu xử lý âm thanh:
- Normalization (chuẩn hóa âm lượng)
- Dithering (thêm noise nhỏ để cải thiện chất lượng)
- Final limiting (giới hạn cuối)
- Multi-band processing (xử lý đa băng tần)

Author: DSP Team
Date: 2025
"""

import numpy as np
from scipy.signal import butter, filtfilt

class EnhancedPostProcessor:
    """
    Bộ hậu xử lý cải tiến với điều khiển tham số chi tiết
    Enhanced post-processing with detailed parameter control
    
    Chức năng:
    - Chuẩn hóa âm lượng theo mục tiêu
    - Dithering để cải thiện chất lượng quantization
    - Limiter cuối để tránh clipping
    - Xử lý đa băng tần
    - Stereo enhancement
    """
    
    def __init__(self, config):
        """
        Khởi tạo EnhancedPostProcessor
        Initialize EnhancedPostProcessor with configuration
        
        Args:
            config: Đối tượng cấu hình DSP
        """
        self.config = config
        self.sr = config.master_config['sample_rate']  # Tần số lấy mẫu
        
        # Thống kê xử lý
        self.processing_stats = {
            'total_processed': 0,
            'peak_reductions': 0,
            'total_gain_applied': 0.0,
            'clipping_prevented': 0
        }
        
        print(f"EnhancedPostProcessor initialized with sr={self.sr}Hz")
    
    def normalize_audio(self, audio, target_db=None):
        """
        Chuẩn hóa âm thanh đến mức âm lượng đích
        Normalize audio to target dB level
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            target_db: Mức âm lượng đích (dB), None = dùng config
            
        Returns:
            numpy.ndarray: Tín hiệu đã chuẩn hóa
        """
        if target_db is None:
            target_db = self.config.master_config['postprocessing']['normalization_target_db']
        
        # Tính RMS hiện tại
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            # Tính RMS đích
            target_rms = 10 ** (target_db / 20)
            
            # Tính gain cần thiết
            gain = target_rms / rms
            
            # Áp dụng chuẩn hóa
            normalized = audio * gain
            
            # Ngăn clipping
            max_val = np.max(np.abs(normalized))
            if max_val > 1.0:
                normalized = normalized / max_val
                self.processing_stats['clipping_prevented'] += 1
            
            # Cập nhật thống kê
            self.processing_stats['total_gain_applied'] += 20 * np.log10(gain)
            
            return normalized
        else:
            return audio
    
    def apply_dithering(self, audio, dither_type='triangular'):
        """
        Áp dụng dithering nếu được bật
        Apply dithering if enabled
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            dither_type: Loại dithering ('triangular', 'gaussian', 'uniform')
            
        Returns:
            numpy.ndarray: Tín hiệu sau khi dithering
        """
        if not self.config.master_config['postprocessing']['dithering']:
            return audio
        
        # Tính mức dithering dựa trên bit depth
        bit_depth = self.config.master_config['bit_depth']
        dither_amplitude = 1.0 / (2 ** bit_depth)
        
        # Tạo dither noise theo loại
        if dither_type == 'triangular':
            # Triangular dither (TPDF)
            dither1 = np.random.uniform(-dither_amplitude, dither_amplitude, len(audio))
            dither2 = np.random.uniform(-dither_amplitude, dither_amplitude, len(audio))
            dither_noise = (dither1 + dither2) / 2
        elif dither_type == 'gaussian':
            # Gaussian dither
            dither_noise = np.random.normal(0, dither_amplitude/3, len(audio))
        else:  # uniform
            # Uniform dither (RPDF)
            dither_noise = np.random.uniform(-dither_amplitude, dither_amplitude, len(audio))
        
        return audio + dither_noise
    
    def final_limiter(self, audio, threshold=0.95, attack_time_ms=0.1, release_time_ms=10):
        """
        Áp dụng limiter cuối nếu được bật
        Apply final limiter if enabled
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            threshold: Ngưỡng limiter (0-1)
            attack_time_ms: Thời gian tấn công (ms)
            release_time_ms: Thời gian thả (ms)
            
        Returns:
            numpy.ndarray: Tín hiệu sau khi limit
        """
        if not self.config.master_config['postprocessing']['final_limiter']:
            return audio
        
        # Tính các hằng số thời gian
        attack_samples = int(attack_time_ms * self.sr / 1000)
        release_samples = int(release_time_ms * self.sr / 1000)
        
        # Tính envelope
        envelope = np.abs(audio)
        
        # Làm mịn envelope
        smoothed_envelope = self._smooth_envelope(envelope, attack_samples, release_samples)
        
        # Tính gain reduction
        gain_reduction = np.where(
            smoothed_envelope > threshold,
            threshold / smoothed_envelope,
            1.0
        )
        
        # Áp dụng gain reduction
        limited = audio * gain_reduction
        
        # Cập nhật thống kê
        peaks_reduced = np.sum(gain_reduction < 1.0)
        if peaks_reduced > 0:
            self.processing_stats['peak_reductions'] += peaks_reduced
        
        return limited
    
    def _smooth_envelope(self, envelope, attack_samples, release_samples):
        """
        Làm mịn envelope với attack/release khác nhau
        Smooth envelope with different attack/release times
        
        Args:
            envelope: Envelope đầu vào
            attack_samples: Số samples attack
            release_samples: Số samples release
            
        Returns:
            numpy.ndarray: Envelope đã làm mịn
        """
        # Tính smoothing coefficients
        attack_coeff = np.exp(-1.0 / attack_samples) if attack_samples > 0 else 0
        release_coeff = np.exp(-1.0 / release_samples) if release_samples > 0 else 0
        
        smoothed = np.zeros_like(envelope)
        smoothed[0] = envelope[0]
        
        for i in range(1, len(envelope)):
            if envelope[i] > smoothed[i-1]:  # Tăng (attack)
                smoothed[i] = attack_coeff * smoothed[i-1] + (1 - attack_coeff) * envelope[i]
            else:  # Giảm (release)
                smoothed[i] = release_coeff * smoothed[i-1] + (1 - release_coeff) * envelope[i]
        
        return smoothed
    
    def multiband_processing(self, audio, bands=None):
        """
        Xử lý đa băng tần với normalization riêng biệt
        Multiband processing with separate normalization
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            bands: Danh sách các băng tần [(low, high, target_db), ...]
            
        Returns:
            numpy.ndarray: Tín hiệu sau xử lý đa băng
        """
        if bands is None:
            # Băng tần mặc định
            bands = [
                (20, 200, -26),     # Bass
                (200, 2000, -23),   # Midrange
                (2000, 8000, -20),  # Treble
                (8000, self.sr//2, -18)  # High treble
            ]
        
        processed_bands = []
        
        for low_freq, high_freq, target_db in bands:
            # Tạo bộ lọc băng thông
            band_audio = self._bandpass_filter(audio, low_freq, high_freq)
            
            # Chuẩn hóa băng này
            normalized_band = self.normalize_audio(band_audio, target_db)
            
            processed_bands.append(normalized_band)
        
        # Kết hợp các băng
        combined_audio = np.sum(processed_bands, axis=0)
        
        # Chuẩn hóa cuối
        final_normalized = self.normalize_audio(combined_audio)
        
        return final_normalized
    
    def _bandpass_filter(self, audio, low_freq, high_freq):
        """
        Bộ lọc băng thông
        Bandpass filter
        
        Args:
            audio: Tín hiệu đầu vào
            low_freq: Tần số thấp
            high_freq: Tần số cao
            
        Returns:
            numpy.ndarray: Tín hiệu đã lọc
        """
        nyquist = self.sr / 2
        low_norm = max(low_freq / nyquist, 0.001)  # Tránh 0
        high_norm = min(high_freq / nyquist, 0.999)  # Tránh 1
        
        if low_norm >= high_norm:
            return np.zeros_like(audio)
        
        try:
            # Bộ lọc băng thông Butterworth
            b, a = butter(4, [low_norm, high_norm], btype='band')
            filtered = filtfilt(b, a, audio)
            return filtered
        except Exception as e:
            print(f"Warning: Bandpass filter failed - {e}")
            return audio
    
    def stereo_enhancement(self, audio, enhancement_factor=1.2):
        """
        Cải thiện stereo (nếu audio là stereo)
        Stereo enhancement (if audio is stereo)
        
        Args:
            audio: Tín hiệu âm thanh (mono hoặc stereo)
            enhancement_factor: Hệ số cải thiện stereo
            
        Returns:
            numpy.ndarray: Tín hiệu sau cải thiện
        """
        if audio.ndim == 1:
            # Mono audio - tạo pseudo-stereo
            delay_samples = int(0.01 * self.sr)  # 10ms delay
            
            if len(audio) > delay_samples:
                left = audio
                right = np.concatenate([np.zeros(delay_samples), audio[:-delay_samples]])
                
                # Thêm một chút phase shift
                stereo_audio = np.column_stack([left, right])
                return stereo_audio
            else:
                return audio
        
        elif audio.ndim == 2:
            # Stereo audio - cải thiện stereo width
            left = audio[:, 0]
            right = audio[:, 1]
            
            # Tính mid và side
            mid = (left + right) / 2
            side = (left - right) / 2
            
            # Cải thiện side
            enhanced_side = side * enhancement_factor
            
            # Tái tạo left/right
            enhanced_left = mid + enhanced_side
            enhanced_right = mid - enhanced_side
            
            # Ngăn clipping
            max_val = max(np.max(np.abs(enhanced_left)), np.max(np.abs(enhanced_right)))
            if max_val > 1.0:
                enhanced_left /= max_val
                enhanced_right /= max_val
            
            return np.column_stack([enhanced_left, enhanced_right])
        
        else:
            return audio
    
    def process(self, audio):
        """
        Pipeline hậu xử lý cải tiến chính
        Main enhanced post-processing pipeline
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            
        Returns:
            numpy.ndarray: Tín hiệu sau hậu xử lý
        """
        print("Starting enhanced post-processing...")
        
        try:
            # Bước 1: Chuẩn hóa
            print("  1. Normalization...")
            normalized = self.normalize_audio(audio)
            
            # Bước 2: Dithering
            print("  2. Dithering...")
            dithered = self.apply_dithering(normalized)
            
            # Bước 3: Limiter cuối
            print("  3. Final limiter...")
            limited = self.final_limiter(dithered)
            
            # Bước 4: Kiểm tra clipping cuối
            print("  4. Final clipping check...")
            final_audio = np.clip(limited, -1.0, 1.0)
            
            # Cập nhật thống kê
            self.processing_stats['total_processed'] += 1
            
            print("  ✓ Post-processing completed successfully")
            return final_audio
        
        except Exception as e:
            print(f"✗ Error in post-processing: {e}")
            print("  Returning original audio...")
            return audio  # Trả về audio gốc nếu có lỗi
    
    def process_with_multiband(self, audio, bands=None):
        """
        Xử lý với multiband processing
        Process with multiband processing
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            bands: Cấu hình các băng tần
            
        Returns:
            numpy.ndarray: Tín hiệu sau xử lý
        """
        print("Starting multiband post-processing...")
        
        try:
            # Bước 1: Xử lý đa băng
            print("  1. Multiband processing...")
            multiband_processed = self.multiband_processing(audio, bands)
            
            # Bước 2: Dithering
            print("  2. Dithering...")
            dithered = self.apply_dithering(multiband_processed)
            
            # Bước 3: Limiter cuối
            print("  3. Final limiter...")
            limited = self.final_limiter(dithered)
            
            # Bước 4: Kiểm tra clipping cuối
            print("  4. Final clipping check...")
            final_audio = np.clip(limited, -1.0, 1.0)
            
            # Cập nhật thống kê
            self.processing_stats['total_processed'] += 1
            
            print("  ✓ Multiband post-processing completed successfully")
            return final_audio
        
        except Exception as e:
            print(f"✗ Error in multiband post-processing: {e}")
            print("  Falling back to standard processing...")
            return self.process(audio)
    
    def get_processing_stats(self):
        """
        Lấy thống kê xử lý
        Get processing statistics
        
        Returns:
            dict: Thống kê xử lý
        """
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['avg_gain_applied'] = stats['total_gain_applied'] / stats['total_processed']
            stats['peak_reduction_rate'] = stats['peak_reductions'] / stats['total_processed']
        else:
            stats['avg_gain_applied'] = 0.0
            stats['peak_reduction_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """
        Reset thống kê xử lý
        Reset processing statistics
        """
        self.processing_stats = {
            'total_processed': 0,
            'peak_reductions': 0,
            'total_gain_applied': 0.0,
            'clipping_prevented': 0
        }
    
    def print_stats(self):
        """
        In thống kê xử lý
        Print processing statistics
        """
        stats = self.get_processing_stats()
        
        print("=" * 50)
        print("POST-PROCESSING STATISTICS")
        print("=" * 50)
        print(f"Total processed: {stats['total_processed']}")
        print(f"Peak reductions: {stats['peak_reductions']}")
        print(f"Clipping prevented: {stats['clipping_prevented']}")
        print(f"Avg gain applied: {stats['avg_gain_applied']:.2f} dB")
        print(f"Peak reduction rate: {stats['peak_reduction_rate']:.2f} per file")
        print("=" * 50)

if __name__ == "__main__":
    # Test code
    from config.dsp_config import DSPConfiguration
    
    config = DSPConfiguration()
    processor = EnhancedPostProcessor(config)
    
    # Tạo tín hiệu test với một số peaks
    test_audio = np.random.randn(22050) * 0.5
    test_audio[1000:1010] = 1.5  # Thêm peak
    test_audio[5000:5010] = -1.2  # Thêm peak âm
    
    # Xử lý
    processed = processor.process(test_audio)
    
    print(f"Original audio - Peak: {np.max(np.abs(test_audio)):.3f}")
    print(f"Processed audio - Peak: {np.max(np.abs(processed)):.3f}")
    
    processor.print_stats()
    print("EnhancedPostProcessor test completed!")
