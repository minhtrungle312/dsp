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
import subprocess
import os
from .spectral_processing import SpectralProcessor
from .harmonic_enhancement import HarmonicEnhancer
from .noise_gate import NoiseGateProcessor

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
        
        # Tính các tham số dẫn xuất
        self.frame_duration = self.n_fft / self.sr
        self.hop_duration = self.hop_length / self.sr
        self.overlap_percentage = (self.n_fft - self.hop_length) / self.n_fft * 100
        self.frequency_resolution = self.sr / self.n_fft
        self.time_resolution = self.hop_length / self.sr
        
        print(f"AdvancedDSPProcessor Configuration:")
        print(f"  Sample Rate: {self.sr} Hz")
        print(f"  Frame duration: {self.frame_duration:.1f}ms")
        print(f"  Hop duration: {self.hop_duration:.1f}ms")
        print(f"  Overlap: {self.overlap_percentage:.1f}%")
        print(f"  Frequency resolution: {self.frequency_resolution:.2f}Hz")
        print(f"  Time resolution: {self.time_resolution:.1f}ms")
    
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
    
    def set_step_export(self, enable=True):
        """
        Enable or disable step-by-step audio export.
        
        Args:
            enable: True to enable export, False to disable
        """
        self.enable_step_export = enable
        if enable:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"✓ Step-by-step audio export enabled: {self.output_dir}")
        else:
            print("✓ Step-by-step audio export disabled")
    
    def extract_audio_from_video(self, video_path, output_path="temp_audio.wav"):
        """
        Trích xuất âm thanh từ video sử dụng FFmpeg
        Extract audio from video using FFmpeg
        
        Args:
            video_path: Đường dẫn video đầu vào
            output_path: Đường dẫn file âm thanh đầu ra
            
        Returns:
            str: Đường dẫn file âm thanh hoặc None nếu lỗi
        """
        try:
            # Lệnh FFmpeg để trích xuất âm thanh
            cmd = [
                'ffmpeg', '-i', video_path,     # Input video
                '-acodec', 'pcm_s16le',         # Audio codec: 16-bit PCM
                '-ar', str(self.sr),            # Sample rate
                '-ac', '1',                     # Mono channel
                output_path, '-y'               # Output file, overwrite
            ]
            
            # Chạy lệnh và bắt lỗi
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Successfully extracted audio to {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error extracting audio: {e}")
            print(f"  stderr: {e.stderr}")
            return None
        except FileNotFoundError:
            print("✗ Error: FFmpeg not found. Please install FFmpeg.")
            print("  Download from: https://ffmpeg.org/download.html")
            return None
    
    def extract_features(self, audio):
        """
        Trích xuất đặc trưng âm thanh toàn diện
        Extract comprehensive audio features
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            
        Returns:
            dict: Dictionary chứa các đặc trưng âm thanh
        """
        try:
            features = {}
            
            # MFCCs (Mel-frequency cepstral coefficients)
            features['mfccs'] = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
            
            # Đặc trưng phổ - Spectral features
            features['spectral_centroids'] = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
            
            # Đặc trưng chroma - Chroma features
            features['chroma'] = librosa.feature.chroma_stft(y=audio, sr=self.sr)
            
            # Tempo và nhịp - Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
            features['tempo'] = tempo
            features['beats'] = beats
            
            # Đặc trưng mel-spectrogram
            features['mel_spectrogram'] = librosa.feature.melspectrogram(y=audio, sr=self.sr)
            
            # Kết hợp các đặc trưng
            combined_features = np.vstack([
                features['mfccs'],
                features['spectral_centroids'],
                features['spectral_rolloff'],
                features['spectral_bandwidth'],
                features['zero_crossing_rate'],
                features['chroma']
            ])
            
            features['combined_features'] = combined_features
            
            return features
            
        except Exception as e:
            print(f"Warning: Feature extraction failed - {e}")
            return None
    
    def process(self, audio):
        """
        Pipeline xử lý DSP cải tiến chính
        Main enhanced DSP processing pipeline
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            
        Returns:
            dict: Kết quả xử lý với audio đã cải thiện và các thông tin trung gian
        """
        print("Starting enhanced DSP processing...")
        
        # Generate timestamp for this processing session
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Export original audio
            self._export_step_audio(audio, "00_original_input", timestamp)
            
            # Bước 1: Tiền xử lý
            print("  1. Preprocessing...")
            preprocessed = self.spectral_processor.preprocess_audio(audio)
            self._export_step_audio(preprocessed, "01_preprocessed", timestamp)
            
            # Bước 2: Phân tích STFT
            print("  2. STFT analysis...")
            stft = librosa.stft(
                preprocessed, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                window=self.config.stft_config['window'],
                center=self.config.stft_config['center'],
                pad_mode=self.config.stft_config['pad_mode']
            )
            
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Bước 3: Profiling nhiễu
            print("  3. Noise profiling...")
            noise_frames = self.config.spectral_subtraction_config['noise_estimation_frames']
            noise_profile = magnitude[:, :min(noise_frames, magnitude.shape[1])]
            noise_psd = np.mean(noise_profile**2, axis=1, keepdims=True)
            
            # Bước 4: Trừ phổ cải tiến
            print("  4. Enhanced spectral subtraction...")
            enhanced_magnitude = self.spectral_processor.enhanced_spectral_subtraction(
                magnitude, noise_profile
            )
            
            # Export spectral subtraction result
            spectral_subtracted_audio = librosa.istft(
                enhanced_magnitude * np.exp(1j * phase),
                hop_length=self.hop_length,
                window=self.config.stft_config['window']
            )
            self._export_step_audio(spectral_subtracted_audio, "02_spectral_subtracted", timestamp)
            
            # Bước 5: Lọc Wiener
            print("  5. Wiener filtering...")
            wiener_filtered = self.harmonic_enhancer.adaptive_wiener_filter(
                enhanced_magnitude * np.exp(1j * phase), noise_psd
            )
            
            # Export Wiener filtered result
            wiener_filtered_audio = librosa.istft(
                wiener_filtered,
                hop_length=self.hop_length,
                window=self.config.stft_config['window']
            )
            self._export_step_audio(wiener_filtered_audio, "03_wiener_filtered", timestamp)
            
            # Bước 6: Ước tính tần số cơ bản
            print("  6. Fundamental frequency estimation...")
            f0_track = self.spectral_processor.estimate_fundamental_frequency(
                np.abs(wiener_filtered)
            )
            
            # Bước 7: Cải thiện âm hài
            print("  7. Harmonic enhancement...")
            harmonic_enhanced = self.harmonic_enhancer.spectral_harmonic_enhancement(
                wiener_filtered, f0_track
            )
            
            # Export harmonic enhanced result
            harmonic_enhanced_audio = librosa.istft(
                harmonic_enhanced,
                hop_length=self.hop_length,
                window=self.config.stft_config['window']
            )
            self._export_step_audio(harmonic_enhanced_audio, "04_harmonic_enhanced", timestamp)
            
            # Bước 8: Tái tạo âm thanh
            print("  8. Audio reconstruction...")
            enhanced_audio = librosa.istft(
                harmonic_enhanced, 
                hop_length=self.hop_length,
                window=self.config.stft_config['window'],
                center=self.config.stft_config['center']
            )
            
            # Bước 9: Cổng nhiễu
            print("  9. Noise gating...")
            gated_audio = self.noise_gate_processor.advanced_noise_gate(enhanced_audio)
            self._export_step_audio(gated_audio, "05_noise_gated", timestamp)
            
            # Bước 10: Nén dải động
            print("  10. Dynamic range compression...")
            compressed_audio = self.noise_gate_processor.dynamic_range_compression(gated_audio)
            self._export_step_audio(compressed_audio, "06_final_compressed", timestamp)
            
            # Bước 11: Trích xuất đặc trưng
            print("  11. Feature extraction...")
            features = self.extract_features(compressed_audio)
            
            # Print export summary
            print(f"\n📁 Step-by-step audio files exported to: {self.output_dir}/")
            print(f"   Session timestamp: {timestamp}")
            
            # Trả về kết quả
            return {
                'enhanced_audio': compressed_audio,
                'enhanced_magnitude': np.abs(harmonic_enhanced),
                'original_phase': phase,
                'fundamental_frequency': f0_track,
                'features': features,
                'intermediate_results': {
                    'preprocessed': preprocessed,
                    'original_magnitude': magnitude,
                    'spectral_subtracted': enhanced_magnitude,
                    'wiener_filtered': np.abs(wiener_filtered),
                    'harmonic_enhanced': np.abs(harmonic_enhanced),
                    'noise_gated': gated_audio,
                    'noise_psd': noise_psd
                }
            }
        
        except Exception as e:
            print(f"✗ Error in DSP processing: {e}")
            return None
    
    def _calculate_audio_metrics(self, audio):
        """
        Tính toán các metrics chất lượng âm thanh
        Calculate audio quality metrics
        
        Args:
            audio: Tín hiệu âm thanh
            
        Returns:
            dict: Các metrics chất lượng
        """
        metrics = {}
        
        # Metrics cơ bản
        metrics['rms_energy'] = np.sqrt(np.mean(audio**2))
        metrics['peak_amplitude'] = np.max(np.abs(audio))
        metrics['dynamic_range'] = self._calculate_dynamic_range(audio)
        
        # Metrics phổ
        try:
            metrics['spectral_centroid'] = np.mean(
                librosa.feature.spectral_centroid(y=audio, sr=self.sr)
            )
            metrics['spectral_rolloff'] = np.mean(
                librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
            )
            metrics['zero_crossing_rate'] = np.mean(
                librosa.feature.zero_crossing_rate(audio)
            )
        except Exception as e:
            print(f"Warning: Spectral metrics calculation failed - {e}")
            metrics['spectral_centroid'] = 0
            metrics['spectral_rolloff'] = 0
            metrics['zero_crossing_rate'] = 0
        
        # Ước tính SNR
        metrics['estimated_snr'] = self._estimate_snr(audio)
        
        return metrics
    
    def _calculate_dynamic_range(self, audio):
        """
        Tính dải động (dB)
        Calculate dynamic range in dB
        
        Args:
            audio: Tín hiệu âm thanh
            
        Returns:
            float: Dải động tính theo dB
        """
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        if rms > 0:
            return 20 * np.log10(peak / rms)
        else:
            return 0
    
    def _estimate_snr(self, audio):
        """
        Ước tính tỷ lệ tín hiệu/nhiễu
        Estimate Signal-to-Noise Ratio
        
        Args:
            audio: Tín hiệu âm thanh
            
        Returns:
            float: SNR ước tính (dB)
        """
        # Ước tính SNR đơn giản bằng cách sử dụng các phần yên tĩnh
        sorted_audio = np.sort(np.abs(audio))
        noise_floor = np.mean(sorted_audio[:int(0.1 * len(sorted_audio))])  # 10% thấp nhất
        signal_level = np.mean(sorted_audio[int(0.9 * len(sorted_audio)):])  # 10% cao nhất
        
        if noise_floor > 0:
            return 20 * np.log10(signal_level / noise_floor)
        else:
            return float('inf')
    
    def batch_process(self, input_dir, output_dir, file_pattern="*.wav"):
        """
        Xử lý hàng loạt file âm thanh
        Batch process audio files
        
        Args:
            input_dir: Thư mục đầu vào
            output_dir: Thư mục đầu ra
            file_pattern: Mẫu file (*.wav, *.mp3, etc.)
        """
        import glob
        
        # Tạo thư mục đầu ra
        os.makedirs(output_dir, exist_ok=True)
        
        # Tìm tất cả file âm thanh
        audio_files = glob.glob(os.path.join(input_dir, file_pattern))
        
        if not audio_files:
            print(f"No audio files found in {input_dir} with pattern {file_pattern}")
            return
        
        print(f"Found {len(audio_files)} audio files to process...")
        
        # Xử lý từng file
        for i, input_path in enumerate(audio_files):
            print(f"\n--- Processing file {i+1}/{len(audio_files)} ---")
            print(f"Input: {os.path.basename(input_path)}")
            
            try:
                # Load audio
                audio, sr = librosa.load(input_path, sr=self.sr)
                
                # Process
                result = self.process(audio)
                
                if result is not None:
                    # Save processed audio
                    output_filename = f"enhanced_{os.path.basename(input_path)}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    sf.write(output_path, result['enhanced_audio'], sr)
                    print(f"✓ Successfully processed: {output_filename}")
                else:
                    print(f"✗ Failed to process: {os.path.basename(input_path)}")
                    
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(input_path)}: {e}")
        
        print(f"\n--- Batch processing completed ---")
    
    def preprocess_for_ai(self, audio):
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
            print("    - Enhanced pre-processing for AI: Full DSP pipeline (8 steps)...")
            
            # Generate timestamp for this preprocessing session
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # ========== BƯỚC 1: EXTRACT AUDIO (Input) ==========
            print("      Step 1/8: Audio extraction completed")
            self._export_step_audio(audio, "01_ai_preprocessing_input", timestamp)
            
            # ========== BƯỚC 2: INITIAL PREPROCESSING ==========
            print("      Step 2/8: Initial preprocessing...")
            preprocessed = self.spectral_processor.preprocess_audio(audio)
            self._export_step_audio(preprocessed, "02_ai_initial_preprocess", timestamp)
            
            # ========== BƯỚC 3: SPECTRAL ANALYSIS ==========
            print("      Step 3/8: Spectral analysis...")
            
            # STFT analysis
            stft = librosa.stft(
                preprocessed, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                window=self.config.stft_config['window'],
                center=self.config.stft_config['center'],
                pad_mode=self.config.stft_config['pad_mode']
            )
            
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            print(f"        STFT: {magnitude.shape} frequency bins × {magnitude.shape[1]} time frames")
            
            # Noise profiling
            noise_frames = self.config.spectral_subtraction_config['noise_estimation_frames']
            noise_profile = magnitude[:, :min(noise_frames, magnitude.shape[1])]
            noise_psd = np.mean(noise_profile**2, axis=1, keepdims=True)
            print(f"        Noise profiling: {noise_frames} frames analyzed")
            
            # ========== BƯỚC 4: ENHANCED SPECTRAL SUBTRACTION ==========
            print("      Step 4/8: Enhanced spectral subtraction...")
            enhanced_magnitude = self.spectral_processor.enhanced_spectral_subtraction(
                magnitude, noise_profile
            )
            
            # Export spectral subtraction result
            spectral_subtracted_stft = enhanced_magnitude * np.exp(1j * phase)
            spectral_subtracted_audio = librosa.istft(
                spectral_subtracted_stft,
                hop_length=self.hop_length,
                window=self.config.stft_config['window']
            )
            self._export_step_audio(spectral_subtracted_audio, "03_ai_spectral_subtracted", timestamp)
            
            # ========== BƯỚC 5: WIENER FILTERING ==========
            print("      Step 5/8: Wiener filtering...")
            wiener_filtered = self.harmonic_enhancer.adaptive_wiener_filter(
                spectral_subtracted_stft, noise_psd
            )
            
            # Export Wiener filtered result
            wiener_filtered_audio = librosa.istft(
                wiener_filtered,
                hop_length=self.hop_length,
                window=self.config.stft_config['window']
            )
            self._export_step_audio(wiener_filtered_audio, "04_ai_wiener_filtered", timestamp)
            
            # ========== BƯỚC 6: HARMONIC ENHANCEMENT ==========
            print("      Step 6/8: Feature enhancement - Harmonic enhancement...")
            
            # F0 estimation for harmonic enhancement
            f0_track = self.spectral_processor.estimate_fundamental_frequency(
                np.abs(wiener_filtered)
            )
            print(f"        F0 tracking: {len(f0_track)} frames estimated")
            
            # Harmonic enhancement
            harmonic_enhanced = self.harmonic_enhancer.spectral_harmonic_enhancement(
                wiener_filtered, f0_track
            )
            
            # Audio reconstruction with enhanced features
            enhanced_audio = librosa.istft(
                harmonic_enhanced, 
                hop_length=self.hop_length,
                window=self.config.stft_config['window'],
                center=self.config.stft_config['center']
            )
            self._export_step_audio(enhanced_audio, "05_ai_harmonic_enhanced", timestamp)
            
            # ========== BƯỚC 7: NOISE GATING ==========
            print("      Step 7/8: Noise gating...")
            gated_audio = self.noise_gate_processor.advanced_noise_gate(enhanced_audio)
            self._export_step_audio(gated_audio, "06_ai_noise_gated", timestamp)
            
            # ========== BƯỚC 8: DYNAMIC COMPRESSION ==========
            print("      Step 8/8: Dynamic range compression...")
            compressed_audio = self.noise_gate_processor.dynamic_range_compression(gated_audio)
            self._export_step_audio(compressed_audio, "07_ai_compressed", timestamp)
            
            # ========== FINAL NORMALIZATION ==========
            print("      Final: Normalization for AI input...")
            
            # Final normalization optimized for AI model input
            final_audio = librosa.util.normalize(compressed_audio)
            
            # Export final preprocessed audio
            self._export_step_audio(final_audio, "08_preprocessed_for_ai", timestamp)
            
            # Print comprehensive preprocessing summary
            print(f"    ✓ Enhanced pre-processing completed - Full 8-step DSP pipeline")
            print(f"      Input length: {len(audio)} samples ({len(audio)/self.sr:.2f}s)")
            print(f"      Output length: {len(final_audio)} samples ({len(final_audio)/self.sr:.2f}s)")
            
            # Print export summary
            print(f"\n    📁 Pre-processing exports (Session: {timestamp}):")
            print(f"      01_ai_preprocessing_input.wav     - Original input")
            print(f"      02_ai_initial_preprocess.wav      - Initial preprocessing")  
            print(f"      03_ai_spectral_subtracted.wav     - After spectral subtraction")
            print(f"      04_ai_wiener_filtered.wav         - After Wiener filtering")
            print(f"      05_ai_harmonic_enhanced.wav       - After harmonic enhancement")
            print(f"      06_ai_noise_gated.wav             - After noise gating")
            print(f"      07_ai_compressed.wav              - After compression")
            print(f"      08_preprocessed_for_ai.wav        - Final preprocessed for AI")
            print(f"    📊 Total: 8 audio files exported to: {self.output_dir}/")
            
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

    def process_audio(self, audio):
        """
        Simple audio processing method that returns only the enhanced audio.
        Method xử lý âm thanh đơn giản chỉ trả về audio đã cải thiện.
        
        Args:
            audio: Tín hiệu âm thanh đầu vào
            
        Returns:
            numpy.ndarray: Audio đã được xử lý
        """
        try:
            # Use the main process method
            result = self.process(audio)
            
            if result is None or 'enhanced_audio' not in result:
                print("    ✗ DSP processing failed, returning original audio")
                return audio
            
            return result['enhanced_audio']
            
        except Exception as e:
            print(f"    ✗ Audio processing error: {e}")
            return audio

if __name__ == "__main__":
    # Test code
    from config.dsp_config import DSPConfiguration
    
    config = DSPConfiguration()
    processor = AdvancedDSPProcessor(config)
    
    # Tạo tín hiệu test
    test_audio = np.random.randn(22050)  # 1 giây âm thanh
    
    # Xử lý
    result = processor.process(test_audio)
    
    if result is not None:
        print(f"✓ Processing successful!")
        print(f"  Original length: {len(test_audio)} samples")
        print(f"  Enhanced length: {len(result['enhanced_audio'])} samples")
        print(f"  Features extracted: {result['features'] is not None}")
    else:
        print("✗ Processing failed!")
    
    print("AdvancedDSPProcessor test completed!")
