"""
Audio Utilities Module - Module tiện ích âm thanh cho Fancam Voice Enhancement
===========================================================================

Module này chứa lớp AudioUtils với các tiện ích xử lý âm thanh:
- Load/save audio files
- Extract audio from video files
- Audio format conversion
- Audio manipulation utilities

Author: DSP Team
Date: 2025
"""

import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import resample
import tempfile
from pathlib import Path

# Import moviepy with availability check
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    VideoFileClip = None

# Import ffmpeg-python as fallback
try:
    import ffmpeg
    FFMPEG_PYTHON_AVAILABLE = True
except ImportError:
    FFMPEG_PYTHON_AVAILABLE = False

class AudioUtils:
    """
    Audio utilities class for fancam voice enhancement
    
    Chức năng:
    - Load/save audio với nhiều format
    - Extract audio từ video files
    - Chuyển đổi sample rate
    - Audio manipulation utilities
    """
    
    def __init__(self):
        """
        Khởi tạo AudioUtils
        Initialize AudioUtils
        """
        # Supported audio and video formats
        self.supported_formats = {
            'input': ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'],
            'output': ['.wav', '.flac']  # Formats we can write
        }
        
        print("AudioUtils initialized")
    
    def load_audio(self, file_path, sr=None, mono=True, duration=None, offset=0.0):
        """
        Load file âm thanh với các tùy chọn
        Load audio file with options
        
        Args:
            file_path: Đường dẫn file âm thanh hoặc video
            sr: Sample rate đích (None = giữ nguyên)
            mono: Có chuyển về mono không
            duration: Độ dài âm thanh cần load (giây)
            offset: Vị trí bắt đầu load (giây)
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Kiểm tra format
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Nếu là video file, chuyển đổi sang audio trước
            video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
            audio_file_path = file_path
            temp_audio_path = None
            
            if file_ext in video_formats:
                print(f"Detected video file: {file_path}")
                print("Converting to audio for librosa processing...")
                
                # Chuyển đổi video sang audio
                temp_audio_path = self.convert_video_to_audio(file_path)
                
                if temp_audio_path and os.path.exists(temp_audio_path):
                    audio_file_path = temp_audio_path
                    print(f"✓ Video converted to temporary audio: {temp_audio_path}")
                else:
                    raise Exception("Failed to convert video to audio")
            
            elif file_ext not in self.supported_formats['input']:
                print(f"Warning: {file_ext} might not be supported")
            
            # Load audio using librosa
            audio, sample_rate = librosa.load(
                audio_file_path,
                sr=sr,
                mono=mono,
                duration=duration,
                offset=offset
            )
            
            # Cleanup temporary file if created
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    print(f"✓ Cleaned up temporary audio file")
                except:
                    print(f"Warning: Could not clean up temporary file: {temp_audio_path}")
            
            print(f"Loaded audio: {file_path}")
            print(f"  Shape: {audio.shape}")
            print(f"  Sample rate: {sample_rate}Hz")
            print(f"  Duration: {len(audio)/sample_rate:.2f}s")
            
            return audio, sample_rate
            
        except Exception as e:
            print(f"Error loading audio {file_path}: {e}")
            return None, None
    
    def save_audio(self, audio, file_path, sample_rate, bit_depth=16):
        """
        Lưu file âm thanh
        Save audio file
        
        Args:
            audio: Dữ liệu âm thanh
            file_path: Đường dẫn file đầu ra
            sample_rate: Sample rate
            bit_depth: Độ sâu bit (16, 24, 32)
            
        Returns:
            bool: True nếu thành công
        """
        try:
            # Kiểm tra format output
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats['output']:
                print(f"Warning: {file_ext} might not be supported for output")
            
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Chuẩn hóa audio để tránh clipping
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
                print("  Audio normalized to prevent clipping")
            
            # Chọn subtype dựa trên bit depth
            subtype_map = {
                16: 'PCM_16',
                24: 'PCM_24',
                32: 'PCM_32'
            }
            subtype = subtype_map.get(bit_depth, 'PCM_16')
            
            # Lưu file
            sf.write(file_path, audio, sample_rate, subtype=subtype)
            
            print(f"Saved audio: {file_path}")
            print(f"  Sample rate: {sample_rate}Hz")
            print(f"  Bit depth: {bit_depth}-bit")
            
            return True
            
        except Exception as e:
            print(f"Error saving audio {file_path}: {e}")
            return False
    
    def convert_sample_rate(self, audio, original_sr, target_sr):
        """
        Chuyển đổi sample rate
        Convert sample rate
        
        Args:
            audio: Dữ liệu âm thanh
            original_sr: Sample rate gốc
            target_sr: Sample rate đích
            
        Returns:
            numpy.ndarray: Audio với sample rate mới
        """
        if original_sr == target_sr:
            return audio
        
        try:
            # Sử dụng librosa cho chất lượng tốt hơn
            resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
            
            print(f"Resampled: {original_sr}Hz -> {target_sr}Hz")
            print(f"  Original length: {len(audio)} samples")
            print(f"  Resampled length: {len(resampled)} samples")
            
            return resampled
            
        except Exception as e:
            print(f"Error resampling audio: {e}")
            return audio
    
    def trim_audio(self, audio, start_time, end_time, sample_rate):
        """
        Cắt âm thanh theo thời gian
        Trim audio by time
        
        Args:
            audio: Dữ liệu âm thanh
            start_time: Thời gian bắt đầu (giây)
            end_time: Thời gian kết thúc (giây)
            sample_rate: Sample rate
            
        Returns:
            numpy.ndarray: Audio đã cắt
        """
        try:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Kiểm tra bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if start_sample >= end_sample:
                print("Error: Invalid trim times")
                return audio
            
            trimmed = audio[start_sample:end_sample]
            
            print(f"Trimmed audio: {start_time:.2f}s - {end_time:.2f}s")
            print(f"  Original length: {len(audio)/sample_rate:.2f}s")
            print(f"  Trimmed length: {len(trimmed)/sample_rate:.2f}s")
            
            return trimmed
            
        except Exception as e:
            print(f"Error trimming audio: {e}")
            return audio
    
    def pad_audio(self, audio, target_length, pad_mode='constant'):
        """
        Padding âm thanh đến độ dài mục tiêu
        Pad audio to target length
        
        Args:
            audio: Dữ liệu âm thanh
            target_length: Độ dài mục tiêu (samples)
            pad_mode: Chế độ padding ('constant', 'edge', 'reflect')
            
        Returns:
            numpy.ndarray: Audio đã padding
        """
        current_length = len(audio)
        
        if current_length >= target_length:
            return audio[:target_length]  # Cắt nếu dài hơn
        
        try:
            pad_length = target_length - current_length
            
            if pad_mode == 'constant':
                padded = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            elif pad_mode == 'edge':
                padded = np.pad(audio, (0, pad_length), mode='edge')
            elif pad_mode == 'reflect':
                padded = np.pad(audio, (0, pad_length), mode='reflect')
            else:
                # Default to constant
                padded = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            
            print(f"Padded audio: {current_length} -> {target_length} samples")
            print(f"  Pad mode: {pad_mode}")
            
            return padded
            
        except Exception as e:
            print(f"Error padding audio: {e}")
            return audio
    
    def normalize_audio(self, audio, method='peak', target_level=-3):
        """
        Chuẩn hóa âm thanh
        Normalize audio
        
        Args:
            audio: Dữ liệu âm thanh
            method: Phương pháp ('peak', 'rms', 'lufs')
            target_level: Mức đích (dB)
            
        Returns:
            numpy.ndarray: Audio đã chuẩn hóa
        """
        try:
            if method == 'peak':
                # Peak normalization
                peak = np.max(np.abs(audio))
                if peak > 0:
                    target_linear = 10 ** (target_level / 20)
                    normalized = audio * (target_linear / peak)
                else:
                    normalized = audio
                    
            elif method == 'rms':
                # RMS normalization
                rms = np.sqrt(np.mean(audio**2))
                if rms > 0:
                    target_rms = 10 ** (target_level / 20)
                    normalized = audio * (target_rms / rms)
                else:
                    normalized = audio
                    
            else:  # Default to peak
                peak = np.max(np.abs(audio))
                if peak > 0:
                    target_linear = 10 ** (target_level / 20)
                    normalized = audio * (target_linear / peak)
                else:
                    normalized = audio
            
            # Prevent clipping
            normalized = np.clip(normalized, -1.0, 1.0)
            
            print(f"Normalized audio using {method} method to {target_level}dB")
            
            return normalized
            
        except Exception as e:
            print(f"Error normalizing audio: {e}")
            return audio
    
    def mix_audio(self, audio1, audio2, ratio1=0.5, ratio2=0.5):
        """
        Trộn hai tín hiệu âm thanh
        Mix two audio signals
        
        Args:
            audio1: Tín hiệu âm thanh thứ nhất
            audio2: Tín hiệu âm thanh thứ hai
            ratio1: Tỷ lệ audio1 (0-1)
            ratio2: Tỷ lệ audio2 (0-1)
            
        Returns:
            numpy.ndarray: Audio đã trộn
        """
        try:
            # Đảm bảo cùng độ dài
            min_length = min(len(audio1), len(audio2))
            audio1_trimmed = audio1[:min_length]
            audio2_trimmed = audio2[:min_length]
            
            # Trộn với tỷ lệ
            mixed = audio1_trimmed * ratio1 + audio2_trimmed * ratio2
            
            # Normalize để tránh clipping
            if np.max(np.abs(mixed)) > 1.0:
                mixed = mixed / np.max(np.abs(mixed))
            
            print(f"Mixed audio: ratio1={ratio1}, ratio2={ratio2}")
            print(f"  Result length: {len(mixed)} samples")
            
            return mixed
            
        except Exception as e:
            print(f"Error mixing audio: {e}")
            return audio1
    
    def apply_fade(self, audio, fade_in_duration=0.1, fade_out_duration=0.1, sample_rate=22050):
        """
        Áp dụng fade in/out
        Apply fade in/out
        
        Args:
            audio: Dữ liệu âm thanh
            fade_in_duration: Thời gian fade in (giây)
            fade_out_duration: Thời gian fade out (giây)
            sample_rate: Sample rate
            
        Returns:
            numpy.ndarray: Audio đã fade
        """
        try:
            faded = audio.copy()
            
            # Fade in
            fade_in_samples = int(fade_in_duration * sample_rate)
            if fade_in_samples > 0 and fade_in_samples < len(audio):
                fade_in_curve = np.linspace(0, 1, fade_in_samples)
                faded[:fade_in_samples] *= fade_in_curve
            
            # Fade out
            fade_out_samples = int(fade_out_duration * sample_rate)
            if fade_out_samples > 0 and fade_out_samples < len(audio):
                fade_out_curve = np.linspace(1, 0, fade_out_samples)
                faded[-fade_out_samples:] *= fade_out_curve
            
            print(f"Applied fade: in={fade_in_duration}s, out={fade_out_duration}s")
            
            return faded
            
        except Exception as e:
            print(f"Error applying fade: {e}")
            return audio
    
    def get_audio_info(self, file_path):
        """
        Lấy thông tin file âm thanh
        Get audio file information
        
        Args:
            file_path: Đường dẫn file âm thanh
            
        Returns:
            dict: Thông tin file âm thanh
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            # Load metadata only
            info = sf.info(file_path)
            
            audio_info = {
                'file_path': file_path,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                'duration_seconds': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype,
                'frames': info.frames
            }
            
            return audio_info
            
        except Exception as e:
            print(f"Error getting audio info {file_path}: {e}")
            return None

    def is_audio_format_supported(self, file_path):
        """
        Check if file format is supported for input.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            bool: True if format is supported
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in self.supported_formats['input']

    def extract_audio_from_video(self, video_path, output_audio_path=None, 
                               sample_rate=44100, mono=True):
        """
        Extract audio from video file using ffmpeg with fallback strategies.
        
        Args:
            video_path: Path to video file
            output_audio_path: Path for extracted audio (optional)
            sample_rate: Target sample rate
            mono: Convert to mono
            
        Returns:
            tuple: (audio_data, sample_rate) or (audio_path, sample_rate) if saved
        """
        try:
            print(f"Extracting audio from video: {video_path}")
            
            # Check if video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Strategy 1: Try moviepy for video processing
            try:
                print("  Trying moviepy for audio extraction...")
                if MOVIEPY_AVAILABLE:
                    video = VideoFileClip(video_path)
                    audio_clip = video.audio
                    
                    # Create temporary file for audio extraction
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_audio_file = temp_file.name
                    temp_file.close()
                    
                    try:
                        # Extract audio to temporary file
                        audio_clip.write_audiofile(temp_audio_file, logger=None, verbose=False)
                        
                        # Load with librosa or soundfile
                        try:
                            audio, sr = librosa.load(
                                temp_audio_file, 
                                sr=sample_rate, 
                                mono=mono
                            )
                        except:
                            import soundfile as sf
                            audio, sr = sf.read(temp_audio_file)
                            if len(audio.shape) > 1 and mono:
                                audio = np.mean(audio, axis=1)
                        
                        print(f"✓ Successfully extracted audio using moviepy")
                        print(f"  Duration: {len(audio)/sr:.2f}s")
                        print(f"  Sample rate: {sr}Hz")
                        
                        # Save to output file if specified
                        if output_audio_path:
                            self.save_audio(audio, output_audio_path, sr)
                            return output_audio_path, sr
                        else:
                            return audio, sr
                            
                    finally:
                        # Cleanup
                        audio_clip.close()
                        video.close()
                        if os.path.exists(temp_audio_file):
                            os.remove(temp_audio_file)
                else:
                    raise ImportError("MoviePy not available")
                    
            except Exception as moviepy_error:
                print(f"  MoviePy extraction failed: {moviepy_error}")
                
                # Strategy 2: Use ffmpeg to extract to temporary WAV file
                try:
                    print("  Trying ffmpeg extraction...")
                    import subprocess
                    import tempfile
                    
                    # Generate temporary audio file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_audio_path = temp_file.name
                    temp_file.close()
                    
                    # Use ffmpeg to extract audio with specific format
                    cmd = [
                        'ffmpeg', '-i', video_path,
                        '-ar', str(sample_rate),
                        '-ac', '1' if mono else '2',
                        '-acodec', 'pcm_s16le',  # Use PCM format for better compatibility
                        '-y',  # Overwrite output file
                        temp_audio_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"✓ Successfully extracted audio using ffmpeg")
                        
                        # Strategy 3: Try multiple ways to load the extracted file
                        audio = None
                        sr = sample_rate
                        
                        # Try soundfile first (more reliable than librosa for WAV)
                        try:
                            print("  Trying soundfile to load extracted audio...")
                            import soundfile as sf
                            audio, sr = sf.read(temp_audio_path)
                            if len(audio.shape) > 1 and mono:
                                audio = np.mean(audio, axis=1)
                            print(f"✓ Successfully loaded with soundfile")
                        except Exception as sf_error:
                            print(f"  Soundfile failed: {sf_error}")
                            
                            # Fallback to librosa
                            try:
                                print("  Trying librosa to load extracted audio...")
                                audio, sr = librosa.load(temp_audio_path, sr=sample_rate, mono=mono)
                                print(f"✓ Successfully loaded with librosa")
                            except Exception as librosa_error2:
                                print(f"  Librosa failed on extracted file: {librosa_error2}")
                        
                        # Clean up temp file
                        try:
                            os.unlink(temp_audio_path)
                        except:
                            pass
                        
                        if audio is not None:
                            print(f"  Final audio shape: {audio.shape}")
                            print(f"  Duration: {len(audio)/sr:.2f}s")
                            
                            # Save to final output if specified
                            if output_audio_path:
                                self.save_audio(audio, output_audio_path, sr)
                                return output_audio_path, sr
                            else:
                                return audio, sr
                        else:
                            raise Exception("Failed to load extracted audio with any method")
                    else:
                        raise Exception(f"FFmpeg failed: {result.stderr}")
                        
                except Exception as ffmpeg_error:
                    print(f"  FFmpeg extraction failed: {ffmpeg_error}")
                    raise Exception(f"Failed to extract audio from video. Both librosa and ffmpeg failed.")
        
        except Exception as e:
            print(f"Error extracting audio from video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def is_video_format(self, file_path):
        """
        Check if file is a video format.
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if video format
        """
        video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in video_formats

    def is_audio_format(self, file_path):
        """
        Check if file is an audio format.
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if audio format
        """
        audio_formats = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in audio_formats

    def convert_video_to_audio(self, video_path, output_audio_path=None, temp_dir=None):
        """
        Chuyển đổi video (MP4, AVI, etc.) sang file audio WAV
        Convert video (MP4, AVI, etc.) to WAV audio file
        
        Args:
            video_path: Đường dẫn file video
            output_audio_path: Đường dẫn file audio đầu ra (None = tự động tạo)
            temp_dir: Thư mục tạm (None = sử dụng temp mặc định)
            
        Returns:
            str: Đường dẫn file audio đã chuyển đổi hoặc None nếu lỗi
        """
        try:
            # Kiểm tra file video tồn tại
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Tạo đường dẫn output nếu không có
            if output_audio_path is None:
                if temp_dir is None:
                    temp_dir = tempfile.gettempdir()
                
                video_name = Path(video_path).stem
                output_audio_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
            
            print(f"Converting video to audio: {video_path} -> {output_audio_path}")
            
            # Phương pháp 1: Sử dụng moviepy (ưu tiên)
            if MOVIEPY_AVAILABLE:
                return self._convert_with_moviepy(video_path, output_audio_path)
            
            # Phương pháp 2: Sử dụng ffmpeg-python
            elif FFMPEG_PYTHON_AVAILABLE:
                return self._convert_with_ffmpeg_python(video_path, output_audio_path)
            
            # Phương pháp 3: Sử dụng ffmpeg command line
            else:
                return self._convert_with_ffmpeg_cli(video_path, output_audio_path)
                
        except Exception as e:
            print(f"Error converting video to audio: {e}")
            return None
    
    def _convert_with_moviepy(self, video_path, output_audio_path):
        """Chuyển đổi sử dụng moviepy"""
        try:
            
            # Load video và extract audio
            video = VideoFileClip(video_path)
            audio = video.audio
            
            # Lưu audio as WAV
            audio.write_audiofile(output_audio_path, logger=None, verbose=False)
            
            # Cleanup
            audio.close()
            video.close()
            
            print(f"✓ Video converted to audio using moviepy: {output_audio_path}")
            return output_audio_path
            
        except Exception as e:
            print(f"moviepy conversion failed: {e}")
            return None
    
    def _convert_with_ffmpeg_python(self, video_path, output_audio_path):
        """Chuyển đổi sử dụng ffmpeg-python"""
        try:
            import ffmpeg
            
            # Extract audio using ffmpeg-python
            (
                ffmpeg
                .input(video_path)
                .output(output_audio_path, acodec='pcm_s16le', ac=1, ar='44100')
                .overwrite_output()
                .run(quiet=True)
            )
            
            print(f"✓ Video converted to audio using ffmpeg-python: {output_audio_path}")
            return output_audio_path
            
        except Exception as e:
            print(f"ffmpeg-python conversion failed: {e}")
            return None
    
    def _convert_with_ffmpeg_cli(self, video_path, output_audio_path):
        """Chuyển đổi sử dụng ffmpeg command line"""
        try:
            import subprocess
            
            # Construct ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # Audio codec
                '-ac', '1',  # Mono
                '-ar', '44100',  # Sample rate
                '-y',  # Overwrite output
                output_audio_path
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Video converted to audio using ffmpeg CLI: {output_audio_path}")
                return output_audio_path
            else:
                print(f"ffmpeg CLI failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"ffmpeg CLI conversion failed: {e}")
            return None
    # ...existing code...
    
if __name__ == "__main__":
    # Test code
    utils = AudioUtils()
    
    # Test với file giả
    test_audio = np.random.randn(22050)  # 1 second of audio
    
    # Test save/load
    test_path = "test_audio.wav"
    if utils.save_audio(test_audio, test_path, 22050):
        loaded_audio, sr = utils.load_audio(test_path)
        if loaded_audio is not None:
            print(f"✓ Save/load test successful: {len(loaded_audio)} samples")
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
    
    print("AudioUtils test completed!")
