"""
Fancam Voice Enhancer - Core Video Processing Module
==================================================

This module contains the FancamVoiceEnhancer class which handles:
- Video to audio extraction
- AI-powered vocal enhancement using Spleeter
- Advanced DSP noise reduction
- Video reconstruction with enhanced audio

Author: DSP Team
Date: 2025
"""

import os
import time
import subprocess
import tempfile
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import datetime

# Import processing components
from core.dsp_processor import AdvancedDSPProcessor
from processors.spleeter_processor import SpleeterProcessor
from utils.audio_utils import AudioUtils



class FancamVoiceEnhancer:
    """
    Main class for fancam voice enhancement processing.
    
    This class orchestrates the entire pipeline:
    1. Extract audio from input video
    2. Enhance vocals using Spleeter + DSP
    3. Reconstruct video with enhanced audio
    """
    
    def __init__(self, config=None):
        """
        Initialize the FancamVoiceEnhancer.
        
        Args:
            config: DSP configuration object
        """
        self.config = config
        
        print("Initializing Fancam Voice Enhancer...")
        
        # Initialize processing components
        self.dsp_processor = AdvancedDSPProcessor(config)
        self.spleeter_processor = SpleeterProcessor(stems='spleeter:2stems-16kHz')
        self.audio_utils = AudioUtils()
    
    def extract_audio_from_video(self, video_path: str, temp_audio_path: str) -> dict:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video
            temp_audio_path: Path for temporary audio file
            
        Returns:
            dict: Result with audio data and metadata
        """
        start_time = time.time()
        
        try:
            print(f"\nExtracting audio from video...")
            print(f"  Video: {video_path}")
            
            # Extract audio using our improved method
            result = self.audio_utils.extract_audio_from_video(
                video_path, 
                output_audio_path=temp_audio_path,
                sample_rate=self.config.master_config['sample_rate'],
                mono=True
            )
            
            # Handle the result which could be (audio, sr) or (path, sr) or (None, None)
            if result is None or result[0] is None:
                return {
                    'success': False,
                    'error': 'Failed to extract audio from video - returned None',
                    'extraction_time': time.time() - start_time
                }
            
            audio_data, sr = result
            
            # If result is a file path, load it
            if isinstance(audio_data, str):
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_data)
                    if len(audio.shape) > 1:  # Convert stereo to mono
                        audio = np.mean(audio, axis=1)
                except Exception as load_error:
                    return {
                        'success': False,
                        'error': f'Failed to load extracted audio file: {str(load_error)}',
                        'extraction_time': time.time() - start_time
                    }
            else:
                audio = audio_data
            
            # Export original extracted audio
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dsp_processor._export_step_audio(audio, "00_original_extracted_from_video", timestamp)
            
            extraction_time = time.time() - start_time
            
            print(f"✓ Audio extraction completed")
            print(f"  Duration: {len(audio)/sr:.2f}s")
            print(f"  Sample Rate: {sr}Hz")
            print(f"  Extraction Time: {extraction_time:.2f}s")
            
            return {
                'success': True,
                'audio': audio,
                'sample_rate': sr,
                'extraction_time': extraction_time,
                'duration': len(audio)/sr
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Audio extraction failed: {str(e)}',
                'extraction_time': time.time() - start_time
            }
    
    def enhance_vocals(self, audio: np.ndarray, sr: int) -> dict:
        """
        Enhanced vocal processing pipeline.
        
        Phase 1: Pre-processing → Phase 2: AI Separation → 
        Phase 3: Audio Reconstruction → Phase 4: Audio Enhancement
        
        Args:
            audio: Audio data array
            sr: Sample rate
            
        Returns:
            dict: Result with enhanced audio
        """
        start_time = time.time()
        
        try:
            print(f"\nEnhancing vocals...")
            
            # Export original audio at the start
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dsp_processor._export_step_audio(audio, "Original_extracted_from_video", timestamp)
            
            # Phase 1: DSP
            preprocessed_audio = self.dsp_processor.dsp_preprocess(audio)
            
            # Phase 2: AI-powered vocal separation
            # Create temporary file for Spleeter (it needs a file path)
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_audio_path = temp_file.name
                
            try:
                # Save preprocessed audio to temporary file
                sf.write(temp_audio_path, preprocessed_audio, sr, subtype='PCM_16')
                
                # Use file-based separation
                separated_audio_paths = self.spleeter_processor.separate_audio(temp_audio_path)
                
                if separated_audio_paths is None:
                    separated_audio = None
                else:
                    # Load the separated audio files back into memory
                    separated_audio = {}
                    for stem, file_path in separated_audio_paths.items():
                        if os.path.exists(file_path):
                            audio_data, _ = sf.read(file_path)
                            # Convert to mono if stereo
                            if len(audio_data.shape) == 2:
                                audio_data = np.mean(audio_data, axis=1)
                            separated_audio[stem] = audio_data
                            print(f"    Loaded {stem}: shape {audio_data.shape}")
                        else:
                            print(f"    Warning: {stem} file not found: {file_path}")
                    
                    if not separated_audio:
                        separated_audio = None
                    else:
                        # Export AI-separated components
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        if 'vocals' in separated_audio:
                            self.dsp_processor._export_step_audio(separated_audio['vocals'], "02a_ai_vocals_separated", timestamp)
                        if 'accompaniment' in separated_audio:
                            self.dsp_processor._export_step_audio(separated_audio['accompaniment'], "02b_ai_accompaniment_separated", timestamp)
                        
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
            # Phase 3: Audio Reconstruction
            print("\n  Phase 3: Audio Reconstruction...")
            reconstruction_result = self.reconstruct_audio_from_ai(
                preprocessed_audio, separated_audio, sr
            )
                
            if not reconstruction_result['success']:
                return {
                    'success': False,
                    'error': reconstruction_result['error'],
                    'enhancement_time': time.time() - start_time
                }
                
            reconstructed_audio = reconstruction_result['reconstructed_audio']
                
            # Export reconstructed audio step
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dsp_processor._export_step_audio(reconstructed_audio, "03_ai_reconstructed", timestamp)

            # export video
            final_audio = preprocessed_audio

            # Tạo thư mục chart nếu chưa có
            chart_dir = "./output/chart"
            os.makedirs(chart_dir, exist_ok=True)
            
            # So sánh Original Audio vs Final Enhanced Audio
            plt.figure(figsize=(15, 10))
            
            # 1. Waveform comparison
            plt.subplot(3, 2, 1)
            librosa.display.waveshow(audio, sr=sr)
            plt.title('Dạng sóng - Original Audio')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Biên độ')
            
            plt.subplot(3, 2, 2)
            librosa.display.waveshow(final_audio, sr=sr)
            plt.title('Dạng sóng - Final Enhanced Audio')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Biên độ')
            
            # 2. Spectrogram comparison
            plt.subplot(3, 2, 3)
            original_stft = librosa.stft(audio, n_fft=2048, hop_length=512, window='hann')
            S_original_db = librosa.amplitude_to_db(np.abs(original_stft))
            librosa.display.specshow(S_original_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Original Audio')
            
            plt.subplot(3, 2, 4)
            final_stft = librosa.stft(final_audio, n_fft=2048, hop_length=512, window='hann')
            S_final_db = librosa.amplitude_to_db(np.abs(final_stft))
            librosa.display.specshow(S_final_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Final Enhanced Audio')
            
            # 3. RMS energy comparison
            plt.subplot(3, 2, 5)
            frame_length = 2048
            hop_length = 512
            original_rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            final_rms = librosa.feature.rms(y=final_audio, frame_length=frame_length, hop_length=hop_length)[0]
            times = librosa.frames_to_time(range(len(original_rms)), sr=sr, hop_length=hop_length)
            
            plt.plot(times, original_rms, label='Original Audio', alpha=0.8)
            plt.plot(times, final_rms, label='Final Enhanced Audio', alpha=0.8)
            plt.title('RMS Energy Comparison')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('RMS Energy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 4. Audio statistics comparison
            plt.subplot(3, 2, 6)
            stats_original = {
                'RMS': np.sqrt(np.mean(audio**2)),
                'Peak': np.max(np.abs(audio)),
                'Crest Factor': np.max(np.abs(audio)) / np.sqrt(np.mean(audio**2)) if np.sqrt(np.mean(audio**2)) > 0 else 0
            }
            stats_final = {
                'RMS': np.sqrt(np.mean(final_audio**2)),
                'Peak': np.max(np.abs(final_audio)),
                'Crest Factor': np.max(np.abs(final_audio)) / np.sqrt(np.mean(final_audio**2)) if np.sqrt(np.mean(final_audio**2)) > 0 else 0
            }
            
            x_pos = np.arange(len(stats_original))
            width = 0.35
            
            plt.bar(x_pos - width/2, list(stats_original.values()), width, label='Original Audio', alpha=0.8)
            plt.bar(x_pos + width/2, list(stats_final.values()), width, label='Final Enhanced Audio', alpha=0.8)
            
            plt.title('Audio Statistics Comparison')
            plt.xlabel('Metrics')
            plt.ylabel('Value')
            plt.xticks(x_pos, list(stats_original.keys()), rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'4_final_comparison_original_vs_enhanced_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Generated comparison chart: 4_final_comparison_original_vs_enhanced_{timestamp}.png")
            print(f"  Original Audio - RMS: {stats_original['RMS']:.6f}, Peak: {stats_original['Peak']:.6f}")
            print(f"  Enhanced Audio - RMS: {stats_final['RMS']:.6f}, Peak: {stats_final['Peak']:.6f}")

            # Export final enhanced audio step
            self.dsp_processor._export_step_audio(final_audio, "04_ai_final_enhanced", timestamp)
            
            enhancement_time = time.time() - start_time
            
            print(f"\n✓ Vocal enhancement completed")
            print(f"  Total processing time: {enhancement_time:.2f}s")
            
            return {
                'success': True,
                'enhanced_audio': final_audio,
                'enhancement_time': enhancement_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Vocal enhancement failed: {str(e)}',
                'enhancement_time': time.time() - start_time
            }
    
    def reconstruct_video(self, original_video_path: str, enhanced_audio_path: str, 
                         output_video_path: str) -> dict:
        """
        Reconstruct video with enhanced audio.
        
        Args:
            original_video_path: Path to original video
            enhanced_audio_path: Path to enhanced audio file
            output_video_path: Path for output video
            
        Returns:
            dict: Result of reconstruction
        """
        start_time = time.time()
        
        try:
            print(f"\nReconstructing video with enhanced audio...")
            print(f"  Original video: {original_video_path}")
            print(f"  Enhanced audio: {enhanced_audio_path}")
            print(f"  Output video: {output_video_path}")
            
            # Use ffmpeg to combine video and enhanced audio
            cmd = [
                'ffmpeg',
                '-i', original_video_path,  # Input video
                '-i', enhanced_audio_path,  # Input enhanced audio
                '-c:v', 'copy',  # Copy video stream without re-encoding
                '-c:a', 'aac',   # Encode audio as AAC
                '-b:a', '192k',  # Audio bitrate
                '-map', '0:v:0', # Map video from first input
                '-map', '1:a:0', # Map audio from second input
                '-shortest',     # Match duration to shortest stream
                '-y',            # Overwrite output file
                output_video_path
            ]
            
            # Run ffmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            reconstruction_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✓ Video reconstruction completed")
                print(f"  Reconstruction time: {reconstruction_time:.2f}s")
                print(f"  Output video: {output_video_path}")
                
                return {
                    'success': True,
                    'reconstruction_time': reconstruction_time
                }
            else:
                return {
                    'success': False,
                    'error': f'FFmpeg failed: {result.stderr}',
                    'reconstruction_time': reconstruction_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Video reconstruction failed: {str(e)}',
                'reconstruction_time': time.time() - start_time
            }
    
    def process_video(self, input_video_path: str, output_video_path: str) -> dict:
        """
        Process a complete fancam video enhancement.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path for output enhanced video
            
        Returns:
            dict: Complete processing result
        """
        total_start_time = time.time()
        
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, 'temp_audio.wav')
        temp_enhanced_audio_path = os.path.join(temp_dir, 'enhanced_audio.wav')
        
        try:
            # Step 1: Extract audio from video
            extraction_result = self.extract_audio_from_video(input_video_path, temp_audio_path)
            
            if not extraction_result['success']:
                return {
                    'success': False,
                    'error': extraction_result['error'],
                    'processing_time': time.time() - total_start_time
                }
            
            # Step 2: Enhance vocals
            enhancement_result = self.enhance_vocals(
                extraction_result['audio'], 
                extraction_result['sample_rate']
            )
            
            if not enhancement_result['success']:
                return {
                    'success': False,
                    'error': enhancement_result['error'],
                    'processing_time': time.time() - total_start_time
                }
            
            # Step 3: Save enhanced audio
            print("Saving enhanced audio...")
            try:
                # Validate audio before saving
                if enhancement_result['enhanced_audio'] is None:
                    raise ValueError("Enhanced audio is None")
                
                enhanced_audio = enhancement_result['enhanced_audio']
                
                # Check for invalid values
                if np.any(np.isnan(enhanced_audio)) or np.any(np.isinf(enhanced_audio)):
                    print("Warning: Invalid values in enhanced audio, cleaning...")
                    enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Normalize if too loud
                if np.max(np.abs(enhanced_audio)) > 1.0:
                    enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio))
                    print("Audio normalized to prevent clipping")
                
                # Save with error handling
                sf.write(temp_enhanced_audio_path, 
                        enhanced_audio, 
                        extraction_result['sample_rate'],
                        subtype='PCM_16')
                
                print(f"✓ Enhanced audio saved: {temp_enhanced_audio_path}")
                
            except Exception as save_error:
                return {
                    'success': False,
                    'error': f'Failed to save enhanced audio: {str(save_error)}',
                    'processing_time': time.time() - total_start_time
                }
            
            # Step 4: Reconstruct video
            reconstruction_result = self.reconstruct_video(
                input_video_path, 
                temp_enhanced_audio_path, 
                output_video_path
            )
            
            if not reconstruction_result['success']:
                return {
                    'success': False,
                    'error': reconstruction_result['error'],
                    'processing_time': time.time() - total_start_time
                }
            
            total_processing_time = time.time() - total_start_time
            
            return {
                'success': True,
                'processing_time': total_processing_time,
                'details': {
                    'duration': extraction_result['duration'],
                    'extraction_time': extraction_result['extraction_time'],
                    'enhancement_time': enhancement_result['enhancement_time'],
                    'reconstruction_time': reconstruction_result['reconstruction_time']
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'processing_time': time.time() - total_start_time
            }
        
        finally:
            # Cleanup temporary files
            try:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                if os.path.exists(temp_enhanced_audio_path):
                    os.remove(temp_enhanced_audio_path)
                os.rmdir(temp_dir)
            except:
                pass  # Ignore cleanup errors
    

    def reconstruct_audio_from_ai(self, original_audio: np.ndarray, separated_audio: dict, sr: int) -> dict:
        try:
            # Tạo timestamp cho tất cả file export
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Tạo thư mục output/chart nếu chưa có
            chart_dir = "./output/chart"
            os.makedirs(chart_dir, exist_ok=True)
            
            # Bước 1: Extract vocals và accompaniment
            vocals = separated_audio.get('vocals', original_audio)
            accompaniment = separated_audio.get('accompaniment', np.zeros_like(original_audio))
            
            # Bước 2: Validation & length sync
            if not isinstance(vocals, np.ndarray):
                raise TypeError(f"Vocals must be numpy.ndarray, got {type(vocals)}")
            
            # Sync lengths for vocals
            if len(vocals) != len(original_audio):
                if len(vocals) > len(original_audio):
                    vocals = vocals[:len(original_audio)]
                else:
                    vocals = np.pad(vocals, (0, len(original_audio) - len(vocals)))
            
            # Sync lengths for accompaniment  
            if len(accompaniment) != len(original_audio):
                if len(accompaniment) > len(original_audio):
                    accompaniment = accompaniment[:len(original_audio)]
                else:
                    accompaniment = np.pad(accompaniment, (0, len(original_audio) - len(accompaniment)))
            
            # Bước 3: Tính RMS và threshold
            vocals_rms = np.sqrt(np.mean(vocals**2))
            gate_threshold = max(vocals_rms * 0.01, vocals_rms * 0.02, 0.0001)
            
            print(f"    Reconstruct Debug - Vocals RMS: {vocals_rms:.6f}, Gate threshold: {gate_threshold:.6f}")
            
            # Chuẩn bị STFT cho visualization
            n_fft = 2048
            hop_length = 512
            vocals_stft = librosa.stft(vocals, n_fft=n_fft, hop_length=hop_length, window='hann')
            original_stft = librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length, window='hann')
            S_vocals_db = librosa.amplitude_to_db(np.abs(vocals_stft))
            S_original_db = librosa.amplitude_to_db(np.abs(original_stft))
            
            # Bước 4: Apply Noise Gate
            gated_vocals = np.where(np.abs(vocals) > gate_threshold, vocals, 0.0)
            
            # Export step 4 (sau noise gate)
            self.dsp_processor._export_step_audio(gated_vocals, f"reconstruct_04_gated_vocals", timestamp)
            
            # Tính STFT cho gated vocals
            gated_vocals_stft = librosa.stft(gated_vocals, n_fft=n_fft, hop_length=hop_length, window='hann')
            
            # 3. Bước 4: Spectrogram so sánh (Vocals vs. Gated Vocals)
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            librosa.display.specshow(S_vocals_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Vocals before Noise Gate (Step 1)')
            
            plt.subplot(2, 1, 2)
            S_gated_vocals_db = librosa.amplitude_to_db(np.abs(gated_vocals_stft))
            librosa.display.specshow(S_gated_vocals_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Gated Vocals (Step 4)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'3_4_reconstruct_04_noise_gate_spectrograms_{timestamp}.png'))
            plt.close()
            
            # 4. Bước 4: Dạng sóng so sánh (Vocals vs. Gated Vocals)
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(vocals, sr=sr)
            plt.title('Dạng sóng - Vocals trước Noise Gate (Bước 1)')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Biên độ')
            
            plt.subplot(2, 1, 2)
            librosa.display.waveshow(gated_vocals, sr=sr)
            plt.title('Dạng sóng - Gated Vocals (Bước 4)')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Biên độ')
            
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'3_4_reconstruct_04_noise_gate_waveforms_{timestamp}.png'))
            plt.close()
            
            # Debug info về noise gate
            below_threshold = np.sum(np.abs(vocals) <= gate_threshold)
            print(f"    Reconstruct Debug - Samples below threshold: {below_threshold}/{len(vocals)} ({100*below_threshold/len(vocals):.1f}%)")
            
            # Bước 5: Audio Mixing
            reconstructed_audio = gated_vocals * 0.9 + accompaniment * 0.1
            
            # Export step 5 (sau mixing)
            self.dsp_processor._export_step_audio(reconstructed_audio, f"reconstruct_05_mixed_audio", timestamp)
            
            # Tính STFT cho reconstructed audio
            reconstructed_stft = librosa.stft(reconstructed_audio, n_fft=n_fft, hop_length=hop_length, window='hann')
            
            # 5. Bước 5: Spectrogram so sánh (Original vs. Mixed Audio)
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            librosa.display.specshow(S_original_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Original Audio (00_original_extracted_from_video.wav)')
            
            plt.subplot(2, 1, 2)
            S_reconstructed_db = librosa.amplitude_to_db(np.abs(reconstructed_stft))
            librosa.display.specshow(S_reconstructed_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Mixed Audio (Bước 5)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'3_5_reconstruct_05_mixed_spectrograms_{timestamp}.png'))
            plt.close()
            
            # 6. Bước 5: Dạng sóng so sánh (Original vs. Mixed Audio)
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(original_audio, sr=sr)
            plt.title('Dạng sóng - Original Audio (00_original_extracted_from_video.wav)')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Biên độ')
            
            plt.subplot(2, 1, 2)
            librosa.display.waveshow(reconstructed_audio, sr=sr)
            plt.title('Dạng sóng - Mixed Audio (Bước 5)')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Biên độ')
            
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'3_5_reconstruct_05_mixed_waveforms_{timestamp}.png'))
            plt.close()
            
            # Bước 7: Clipping Prevention
            max_val = np.max(np.abs(reconstructed_audio))
            print(f"    Reconstruct Debug - Max value before clipping: {max_val:.6f}")
            
            reconstructed_audio_before_clipping = None
            
            if max_val > 0.9:
                reconstructed_audio_before_clipping = reconstructed_audio.copy()
                reconstructed_audio = np.tanh(reconstructed_audio * 0.9) * 0.8
                print(f"    Reconstruct Debug - Applied soft clipping")
                
                # Export step 7a (trước khi clipping)
                self.dsp_processor._export_step_audio(reconstructed_audio_before_clipping, f"reconstruct_07a_before_clipping", timestamp)
            
            # Export final result
            self.dsp_processor._export_step_audio(reconstructed_audio, f"reconstruct_07_final_result", timestamp)
            
            # 7. Bước 7: Spectrogram so sánh (Mixed vs. Final Result)
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            if reconstructed_audio_before_clipping is not None:
                before_clipping_stft = librosa.stft(reconstructed_audio_before_clipping, n_fft=n_fft, hop_length=hop_length, window='hann')
                S_before_clipping_db = librosa.amplitude_to_db(np.abs(before_clipping_stft))
                title = 'Spectrogram - Mixed Audio trước Clipping (Bước 7a)'
            else:
                S_before_clipping_db = S_reconstructed_db
                title = 'Spectrogram - Mixed Audio (Bước 5)'
            librosa.display.specshow(S_before_clipping_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title(title)
            
            plt.subplot(2, 1, 2)
            final_stft = librosa.stft(reconstructed_audio, n_fft=n_fft, hop_length=hop_length, window='hann')
            S_final_db = librosa.amplitude_to_db(np.abs(final_stft))
            librosa.display.specshow(S_final_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram - Final Result (Bước 7)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'3_7_reconstruct_07_final_spectrograms_{timestamp}.png'))
            plt.close()
            
            # 8. Bước 7: Dạng sóng so sánh (Mixed vs. Final Result)
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            if reconstructed_audio_before_clipping is not None:
                librosa.display.waveshow(reconstructed_audio_before_clipping, sr=sr)
                title = 'Dạng sóng - Mixed Audio trước Clipping (Bước 7a)'
            else:
                librosa.display.waveshow(reconstructed_audio, sr=sr)
                title = 'Dạng sóng - Mixed Audio (Bước 5)'
            plt.title(title)
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Biên độ')
            
            plt.subplot(2, 1, 2)
            librosa.display.waveshow(reconstructed_audio, sr=sr)
            plt.title('Dạng sóng - Final Result (Bước 7)')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Biên độ')
            
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'3_7_reconstruct_07_final_waveforms_{timestamp}.png'))
            plt.close()
            
            # Final debug info
            final_rms = np.sqrt(np.mean(reconstructed_audio**2))
            final_max = np.max(np.abs(reconstructed_audio))
            print(f"    Reconstruct Debug - Final RMS: {final_rms:.6f}, Final Max: {final_max:.6f}")
            
            return {
                'success': True,
                'reconstructed_audio': reconstructed_audio,
                'clean_vocals': gated_vocals
            }
            
        except Exception as e:
            print(f"    Reconstruct Error: {str(e)}")
            return {'success': False, 'error': str(e)}