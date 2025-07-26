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
    
    def extract_audio_from_video(self, video_path, output_audio_path=None, 
                               sample_rate=44100, mono=True):
        """
        Extract audio from video file using ffmpeg with moviepy fallback.
        
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
            
            # Strategy 1: Try FFmpeg first (faster and more reliable)
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
                    
                    # Try multiple ways to load the extracted file
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
                        except Exception as librosa_error:
                            print(f"  Librosa failed on extracted file: {librosa_error}")
                    
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
                
                # Strategy 2: Fallback to moviepy for video processing
                try:
                    print("  Trying moviepy as fallback...")
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
                    raise Exception(f"Failed to extract audio from video. Both FFmpeg and MoviePy failed.")
        
        except Exception as e:
            print(f"Error extracting audio from video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
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
    
if __name__ == "__main__":
    # Test code
    utils = AudioUtils()
