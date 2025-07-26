"""
Advanced DSP Processor Module - Module xử lý DSP tiên tiến
========================================================

Module này chứa lớp AdvancedDSPProcessor - bộ xử lý DSP chính:
- Tích hợp tất cả các module xử lý
- Pipeline xử lý hoàn chỉnh
- Feature extraction (trích xuất đặc trưng)
- Video processing (xử lý video)

Author: DSP Team
Date: 2025
"""

import numpy as np
import librosa
import soundfile as sf
import os
from .spectral_processing import SpectralProcessor
from .harmonic_enhancement import HarmonicEnhancer
from .noise_gate import NoiseGateProcessor
import matplotlib.pyplot as plt

class AdvancedDSPProcessor:
    """
    Bộ xử lý DSP tiên tiến với điều khiển tham số chi tiết
    Enhanced Digital Signal Processing with detailed parameter control
    
    Tích hợp tất cả các module xử lý:
    - SpectralProcessor: Xử lý phổ tín hiệu
    - HarmonicEnhancer: Cải thiện âm hài
    - NoiseGateProcessor: Xử lý cổng nhiễu
    """
    
    def __init__(self, config):
        """
        Khởi tạo AdvancedDSPProcessor
        Initialize AdvancedDSPProcessor with configuration
        
        Args:
            config: Đối tượng cấu hình DSP
        """
        self.config = config
        self.sr = config.master_config['sample_rate']  # Tần số lấy mẫu
        self.n_fft = config.stft_config['n_fft']       # Kích thước FFT
        self.hop_length = config.stft_config['hop_length']  # Hop length
        
        # Khởi tạo các module xử lý con
        self.spectral_processor = SpectralProcessor(config)
        self.harmonic_enhancer = HarmonicEnhancer(config)
        self.noise_gate_processor = NoiseGateProcessor(config)
        
        # Setup output directory for step-by-step audio exports
        self.output_dir = "./output/dsp_steps"
        os.makedirs(self.output_dir, exist_ok=True)
        self.enable_step_export = True  # Enable/disable step-by-step export
        
        # Setup output directory for charts
        self.chart_dir = "./output/chart"
        os.makedirs(self.chart_dir, exist_ok=True)
        
        # Tính các tham số dẫn xuất
        self.frame_duration = self.n_fft / self.sr
        self.hop_duration = self.hop_length / self.sr
        self.overlap_percentage = (self.n_fft - self.hop_length) / self.n_fft * 100
        self.frequency_resolution = self.sr / self.n_fft
        self.time_resolution = self.hop_length / self.sr
    
    def _export_step_audio(self, audio, step_name, timestamp=None):
        """
        Export audio from a specific processing step for analysis.
        
        Args:
            audio: Audio data to export
            step_name: Name of the processing step
            timestamp: Optional timestamp for unique filenames
        """
        if not self.enable_step_export:
            return
            
        try:
            import datetime
            
            if timestamp is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Clean step name for filename
            clean_step_name = step_name.lower().replace(" ", "_").replace("-", "_")
            filename = f"{timestamp}_{clean_step_name}.wav"
            
            # Use forward slashes for compatibility
            filepath = os.path.join(self.output_dir, filename).replace('\\', '/')
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Ensure audio is valid
            if audio is None or len(audio) == 0:
                print(f"    ⚠️  Warning: Cannot export {step_name} - audio is empty")
                return
                
            # Normalize audio to prevent clipping
            audio_normalized = audio.copy()
            if np.max(np.abs(audio_normalized)) > 0:
                audio_normalized = audio_normalized / np.max(np.abs(audio_normalized)) * 0.95
            
            # Export audio
            sf.write(filepath, audio_normalized, self.sr, subtype='PCM_16')
            print(f"    📁 Exported {step_name}: {filename}")
            
        except Exception as e:
            print(f"    ⚠️  Warning: Failed to export {step_name}: {e}")


    def dsp_preprocess(self, audio):
        """
        Enhanced pre-processing for AI model input - Full DSP preprocessing pipeline
        Tiền xử lý nâng cao cho AI model - Pipeline DSP đầy đủ 8 bước
        
        Pipeline: Extract Audio → Spectral Analysis → Noise Reduction → Feature Enhancement
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            
        Returns:
            numpy.ndarray: Audio đã được tiền xử lý đầy đủ
        """
        try:
            
            # Generate timestamp for this preprocessing session
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # step1 : SPECTRAL ANALYSIS

            # STFT analysis
            stft = librosa.stft(
                audio, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                window=self.config.stft_config['window'],
                center=self.config.stft_config['center'],
                pad_mode=self.config.stft_config['pad_mode']
            )

            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft), ref=np.max), sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Audio gốc (STFT)')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Tần số (Hz)')
            plt.savefig(os.path.join(self.chart_dir, '2_1_stft_spectrogram.png'))
            plt.close()

            # Noise profiling
            D_vocal, D_crowd = librosa.decompose.hpss(stft, margin=1.0)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            S_harmonic_db = librosa.amplitude_to_db(np.abs(D_vocal))
            librosa.display.specshow(S_harmonic_db, sr=self.sr, hop_length=512, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Harmonic (Vocal)')

            plt.subplot(1, 2, 2)
            S_percussive_db = librosa.amplitude_to_db(np.abs(D_crowd))
            librosa.display.specshow(S_percussive_db, sr=self.sr, hop_length=512, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Percussive (Tiếng khán giả)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.chart_dir, '2_2_hpss_spectrogram.png'))
            plt.close()

            # step2: ENHANCED SPECTRAL SUBTRACTION
            orig_mag = np.abs(stft)
            noise_mag = np.abs(D_crowd)
            alpha = 1.0
            beta = 0.15
            clean_mag = np.maximum(orig_mag - alpha * noise_mag, beta * orig_mag)
            orig_phase = np.angle(stft)
            clean_stft = clean_mag * np.exp(1j * orig_phase)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            librosa.display.specshow(orig_mag, sr=self.sr, hop_length=512, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Trước Spectral Subtraction')

            plt.subplot(1, 2, 2)
            S_clean_db = librosa.amplitude_to_db(np.abs(clean_stft))
            librosa.display.specshow(S_clean_db, sr=self.sr, hop_length=512, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Sau Spectral Subtraction')
            plt.tight_layout()
            plt.savefig(os.path.join(self.chart_dir, '2_3_spectral_subtraction_spectrogram.png'))
            plt.close()

            # Export spectral subtraction result
            spectral_subtracted_stft = clean_stft
            spectral_subtracted_audio = librosa.istft(
                spectral_subtracted_stft,
                hop_length=self.hop_length,
                window=self.config.stft_config['window']
            )
            self._export_step_audio(spectral_subtracted_audio, "02_dsp_spectral_subtracted", timestamp)
            
            #step 3: WIENER FILTERING
            noise_psd = np.mean(noise_mag**2, axis=1, keepdims=True)
            wiener_filtered = self.harmonic_enhancer.adaptive_wiener_filter(
                spectral_subtracted_stft, noise_psd
            )

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            S_before_db = librosa.amplitude_to_db(np.abs(spectral_subtracted_stft))
            librosa.display.specshow(S_before_db, sr=self.sr, hop_length=512, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Trước Wiener Filter')

            plt.subplot(1, 2, 2)
            S_filtered_db = librosa.amplitude_to_db(wiener_filtered)
            librosa.display.specshow(S_filtered_db, sr=self.sr, hop_length=512, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Sau Wiener Filter')
            plt.tight_layout()
            plt.savefig(os.path.join(self.chart_dir, '2_4_wiener_filter_spectrogram.png'))
            plt.close()
            
            # Export Wiener filtered result
            wiener_filtered_audio = librosa.istft(
                wiener_filtered,
                hop_length=self.hop_length,
                window=self.config.stft_config['window']
            )
            self._export_step_audio(wiener_filtered_audio, "03_dsp_wiener_filtered", timestamp)
            
            # step 4: HARMONIC ENHANCEMENT
            # F0 estimation for harmonic enhancement
            f0_track = self.spectral_processor.estimate_fundamental_frequency(
                np.abs(wiener_filtered)
            )
            print(f"        F0 tracking: {len(f0_track)} frames estimated")
            plt.figure(figsize=(10, 4))
            times = np.linspace(0, len(f0_track) * 512 / self.sr, len(f0_track))
            plt.plot(times, f0_track)
            plt.title('Tần số cơ bản (F0) - Harmonic Enhancement')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Tần số (Hz)')
            plt.savefig(os.path.join(self.chart_dir, '2_5_f0_track.png'))
            plt.close()
            # Harmonic enhancement
            harmonic_enhanced = self.harmonic_enhancer.spectral_harmonic_enhancement(
                wiener_filtered, f0_track
            )

            # Spectrogram
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            S_before_db = librosa.amplitude_to_db(np.abs(wiener_filtered))
            librosa.display.specshow(S_before_db, sr=self.sr, hop_length=512, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Trước Harmonic Enhancement')

            plt.subplot(1, 2, 2)
            S_enhanced_db = librosa.amplitude_to_db(np.abs(harmonic_enhanced))
            librosa.display.specshow(S_enhanced_db, sr=self.sr, hop_length=512, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Sau Harmonic Enhancement')
            plt.tight_layout()
            plt.savefig(os.path.join(self.chart_dir, '2_5_harmonic_enhancement_spectrogram.png'))
            plt.close()
            # Audio reconstruction with enhanced features
            enhanced_audio = librosa.istft(
                harmonic_enhanced, 
                hop_length=self.hop_length,
                window=self.config.stft_config['window'],
                center=self.config.stft_config['center']
            )
            self._export_step_audio(enhanced_audio, "04_dsp_harmonic_enhanced", timestamp)
            
            #step 5: NOISE GATING
            gated_audio = self.noise_gate_processor.advanced_noise_gate(enhanced_audio)
            # Dạng sóng
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            librosa.display.waveshow(enhanced_audio, sr=self.sr)
            plt.title('Dạng sóng - Trước Noise Gating')

            plt.subplot(1, 2, 2)
            librosa.display.waveshow(gated_audio, sr=self.sr)
            plt.title('Dạng sóng - Sau Noise Gating')
            plt.tight_layout()
            plt.savefig(os.path.join(self.chart_dir, '2_6_noise_gating_waveform.png'))
            plt.close()
            self._export_step_audio(gated_audio, "05_dsp_noise_gated", timestamp)
            # FINAL NORMALIZATION
            # Final normalization optimized for AI model input
            final_audio = librosa.util.normalize(gated_audio)
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            librosa.display.waveshow(gated_audio, sr=self.sr)
            plt.title('Dạng sóng - Trước Normalization')

            plt.subplot(1, 2, 2)
            librosa.display.waveshow(final_audio, sr=self.sr)
            plt.title('Dạng sóng - Sau Normalization')
            plt.tight_layout()
            plt.savefig(os.path.join(self.chart_dir, '2_7_normalization_waveform.png'))
            plt.close()

            # Export final preprocessed audio
            self._export_step_audio(final_audio, "05_preprocessed_for_ai", timestamp)
            
            
            return final_audio
            
        except Exception as e:
            print(f"    ✗ Enhanced pre-processing failed: {e}")
            print(f"      Error details: {str(e)}")
            print("    → Falling back to simple preprocessing...")
            
            # Fallback to simple preprocessing
            try:
                normalized_audio = librosa.util.normalize(audio)
                # Export fallback result
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self._export_step_audio(normalized_audio, "fallback_preprocessed_for_ai", timestamp)
                return normalized_audio
            except:
                print("    → Using original audio as last resort...")
                return audio

if __name__ == "__main__":
    # Test code
    from config.dsp_config import DSPConfiguration
    
    config = DSPConfiguration()
    processor = AdvancedDSPProcessor(config)
