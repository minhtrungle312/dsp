"""
Spleeter Processor Module - Module xử lý tách nguồn âm thanh AI
============================================================

Module này chứa lớp SpleeterProcessor để tách nguồn âm thanh sử dụng AI:
- Tách vocals và accompaniment
- Xử lý batch nhiều file
- Tính toán metrics chất lượng
- Cleanup tự động

Author: DSP Team
Date: 2025
"""

import os
import shutil
import numpy as np
import warnings

# Early warning suppression for TensorFlow/Spleeter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Kiểm tra Spleeter availability
try:
    from spleeter.separator import Separator
    from spleeter.audio.adapter import AudioAdapter
    SPLEETER_AVAILABLE = True
except ImportError:
    print("Warning: Spleeter not available. Install with: pip install spleeter")
    SPLEETER_AVAILABLE = False

class SpleeterProcessor:
    """
    Bộ xử lý tách nguồn âm thanh dựa trên Spleeter
    Spleeter-based source separation processor
    
    Chức năng:
    - Tách vocals ra khỏi accompaniment
    - Xử lý batch nhiều file
    - Tính toán metrics chất lượng tách
    - Quản lý file tự động
    """
    
    def __init__(self, stems='spleeter:2stems-16kHz', model_dir=None):
        """
        Khởi tạo SpleeterProcessor
        Initialize SpleeterProcessor
        
        Args:
            stems: Loại model Spleeter (2stems, 4stems, 5stems)
            model_dir: Thư mục chứa model (None = tự động tải)
        """
        self.stems = stems
        self.model_dir = model_dir
        self.separator = None
        self.audio_adapter = None
        self.is_initialized = False
        
        # Các model có sẵn
        self.available_models = {
            'spleeter:2stems-16kHz': 'Tách thành vocals và accompaniment (16kHz)',
            'spleeter:2stems-8kHz': 'Tách thành vocals và accompaniment (8kHz)',
            'spleeter:4stems-16kHz': 'Tách thành vocals, drums, bass, other (16kHz)',
            'spleeter:5stems-16kHz': 'Tách thành vocals, drums, bass, piano, other (16kHz)'
        }
        
        # Thống kê xử lý
        self.processing_stats = {
            'total_processed': 0,
            'successful_separations': 0,
            'failed_separations': 0,
            'total_processing_time': 0.0
        }
        
        # Khởi tạo Spleeter
        self._initialize_spleeter()
    
    def _initialize_spleeter(self):
        """
        Khởi tạo Spleeter với error handling
        Initialize Spleeter with error handling
        """
        if not SPLEETER_AVAILABLE:
            print("✗ Spleeter not available")
            return
        
        try:
            print(f"Initializing Spleeter with {self.stems}...")
            
            # Khởi tạo separator
            if self.model_dir:
                self.separator = Separator(self.stems, params_filename=self.model_dir)
            else:
                self.separator = Separator(self.stems)
            
            # Khởi tạo audio adapter
            self.audio_adapter = AudioAdapter.default()
            
            self.is_initialized = True
            print("✓ Spleeter initialized successfully")
            print(f"  Model: {self.available_models.get(self.stems, 'Unknown model')}")
            print("✓ AudioAdapter initialized successfully")
            
        except Exception as e:
            print(f"✗ Spleeter initialization failed: {e}")
            print("  Please check your Spleeter installation and model files")
            self.separator = None
            self.audio_adapter = None
            self.is_initialized = False
    
    def is_available(self):
        """
        Kiểm tra Spleeter có khả dụng không
        Check if Spleeter is available and initialized
        
        Returns:
            bool: True nếu Spleeter sẵn sàng sử dụng
        """
        return self.is_initialized and self.separator is not None
    
    def separate_audio(self, audio_path, output_dir="spleeter_output"):
        """
        Tách âm thanh sử dụng Spleeter
        Separate audio using Spleeter
        
        Args:
            audio_path: Đường dẫn file âm thanh đầu vào
            output_dir: Thư mục lưu kết quả tách
            
        Returns:
            dict: Dictionary chứa đường dẫn các file đã tách
        """
        if not self.is_available():
            print("✗ Spleeter not available, skipping separation")
            return None
        
        try:
            import time
            start_time = time.time()
            
            print(f"Separating audio with Spleeter...")
            print(f"  Input: {os.path.basename(audio_path)}")
            print(f"  Model: {self.stems}")
            
            # Kiểm tra file đầu vào
            if not os.path.exists(audio_path):
                print(f"✗ Input file not found: {audio_path}")
                return None
            
            # Load audio using AudioAdapter
            waveform, sample_rate = self.audio_adapter.load(audio_path)
            print(f"  Loaded audio: {waveform.shape}, {sample_rate}Hz")
            
            # Thực hiện tách
            separated = self.separator.separate(waveform)
            
            # Tạo thư mục output
            os.makedirs(output_dir, exist_ok=True)
            
            # Lưu các track đã tách
            output_paths = {}
            
            for stem, audio_data in separated.items():
                output_path = os.path.join(output_dir, f"{stem}.wav")
                
                # Save audio using AudioAdapter
                self.audio_adapter.save(output_path, audio_data, sample_rate)
                
                output_paths[stem] = output_path
                print(f"  ✓ Saved {stem}: {audio_data.shape} -> {output_path}")
            
            # Cập nhật thống kê
            processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            self.processing_stats['successful_separations'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            
            print(f"  ✓ Separation completed in {processing_time:.2f}s")
            return output_paths
            
        except Exception as e:
            print(f"✗ Spleeter separation failed: {e}")
            self.processing_stats['total_processed'] += 1
            self.processing_stats['failed_separations'] += 1
            return None
    
    def extract_vocals(self, audio_path, output_path="vocals.wav"):
        """
        Trích xuất vocals sử dụng Spleeter
        Extract vocals using Spleeter
        
        Args:
            audio_path: Đường dẫn file âm thanh đầu vào
            output_path: Đường dẫn file vocals đầu ra
            
        Returns:
            str: Đường dẫn file vocals hoặc None nếu lỗi
        """
        # Tách âm thanh
        separated_paths = self.separate_audio(audio_path)
        
        if separated_paths and 'vocals' in separated_paths:
            # Copy vocals to desired output path
            shutil.copy2(separated_paths['vocals'], output_path)
            print(f"✓ Vocals extracted to {output_path}")
            
            # Cleanup temporary files nếu output_path khác
            if output_path != separated_paths['vocals']:
                self._cleanup_temp_files(separated_paths)
            
            return output_path
        else:
            print("✗ Failed to extract vocals")
            return None
    
    def extract_accompaniment(self, audio_path, output_path="accompaniment.wav"):
        """
        Trích xuất accompaniment sử dụng Spleeter
        Extract accompaniment using Spleeter
        
        Args:
            audio_path: Đường dẫn file âm thanh đầu vào
            output_path: Đường dẫn file accompaniment đầu ra
            
        Returns:
            str: Đường dẫn file accompaniment hoặc None nếu lỗi
        """
        # Tách âm thanh
        separated_paths = self.separate_audio(audio_path)
        
        if separated_paths and 'accompaniment' in separated_paths:
            # Copy accompaniment to desired output path
            shutil.copy2(separated_paths['accompaniment'], output_path)
            print(f"✓ Accompaniment extracted to {output_path}")
            
            # Cleanup temporary files nếu output_path khác
            if output_path != separated_paths['accompaniment']:
                self._cleanup_temp_files(separated_paths)
            
            return output_path
        else:
            print("✗ Failed to extract accompaniment")
            return None
    
    def batch_separate_audio(self, input_dir, output_dir, audio_extensions=None):
        """
        Tách âm thanh hàng loạt
        Batch separate audio files
        
        Args:
            input_dir: Thư mục chứa file âm thanh đầu vào
            output_dir: Thư mục lưu kết quả
            audio_extensions: Danh sách extension file âm thanh
            
        Returns:
            dict: Thống kê xử lý batch
        """
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        
        if not os.path.exists(input_dir):
            print(f"✗ Input directory not found: {input_dir}")
            return None
        
        # Tìm tất cả file âm thanh
        audio_files = []
        for ext in audio_extensions:
            import glob
            audio_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
            audio_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
        
        if not audio_files:
            print(f"✗ No audio files found in {input_dir}")
            return None
        
        print(f"Found {len(audio_files)} audio files to process...")
        
        # Tạo thư mục output
        os.makedirs(output_dir, exist_ok=True)
        
        # Thống kê batch
        batch_stats = {
            'total_files': len(audio_files),
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0
        }
        
        # Xử lý từng file
        for i, audio_path in enumerate(audio_files):
            print(f"\n--- Processing file {i+1}/{len(audio_files)} ---")
            print(f"File: {os.path.basename(audio_path)}")
            
            try:
                # Tạo thư mục con cho file này
                file_name = os.path.splitext(os.path.basename(audio_path))[0]
                file_output_dir = os.path.join(output_dir, file_name)
                
                # Tách âm thanh
                result = self.separate_audio(audio_path, file_output_dir)
                
                if result:
                    batch_stats['successful_files'] += 1
                    print(f"✓ Successfully processed: {file_name}")
                else:
                    batch_stats['failed_files'] += 1
                    print(f"✗ Failed to process: {file_name}")
                
                batch_stats['processed_files'] += 1
                
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(audio_path)}: {e}")
                batch_stats['failed_files'] += 1
                batch_stats['processed_files'] += 1
        
        print(f"\n--- Batch processing completed ---")
        print(f"Total files: {batch_stats['total_files']}")
        print(f"Successful: {batch_stats['successful_files']}")
        print(f"Failed: {batch_stats['failed_files']}")
        
        return batch_stats
    
    def calculate_separation_quality(self, original_path, vocals_path, accompaniment_path):
        """
        Tính toán chất lượng tách
        Calculate separation quality metrics
        
        Args:
            original_path: Đường dẫn file gốc
            vocals_path: Đường dẫn file vocals
            accompaniment_path: Đường dẫn file accompaniment
            
        Returns:
            dict: Metrics chất lượng tách
        """
        try:
            import librosa
            
            # Load các file
            original, sr = librosa.load(original_path, sr=None)
            vocals, _ = librosa.load(vocals_path, sr=sr)
            accompaniment, _ = librosa.load(accompaniment_path, sr=sr)
            
            # Tái tạo từ các thành phần
            reconstructed = vocals + accompaniment
            
            # Tính các metrics
            metrics = {}
            
            # Signal-to-Distortion Ratio (SDR)
            mse = np.mean((original - reconstructed)**2)
            signal_power = np.mean(original**2)
            metrics['sdr_db'] = 10 * np.log10(signal_power / (mse + 1e-10))
            
            # Correlation coefficient
            metrics['correlation'] = np.corrcoef(original, reconstructed)[0, 1]
            
            # Energy preservation
            original_energy = np.sum(original**2)
            reconstructed_energy = np.sum(reconstructed**2)
            metrics['energy_preservation'] = reconstructed_energy / original_energy
            
            # Spectral similarity
            original_spec = np.abs(librosa.stft(original))
            reconstructed_spec = np.abs(librosa.stft(reconstructed))
            spectral_correlation = np.corrcoef(
                original_spec.flatten(), 
                reconstructed_spec.flatten()
            )[0, 1]
            metrics['spectral_similarity'] = spectral_correlation
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating separation quality: {e}")
            return None
    
    def _cleanup_temp_files(self, file_paths):
        """
        Dọn dẹp file tạm
        Cleanup temporary files
        
        Args:
            file_paths: Dictionary chứa đường dẫn file
        """
        try:
            for path in file_paths.values():
                if os.path.exists(path):
                    os.remove(path)
            
            # Xóa thư mục nếu rỗng
            if file_paths:
                dir_path = os.path.dirname(list(file_paths.values())[0])
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
            
        except Exception as e:
            print(f"Warning: Cleanup failed - {e}")
    
    def get_processing_stats(self):
        """
        Lấy thống kê xử lý
        Get processing statistics
        
        Returns:
            dict: Thống kê xử lý
        """
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_separations'] / stats['total_processed']
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['avg_processing_time'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """
        Reset thống kê xử lý
        Reset processing statistics
        """
        self.processing_stats = {
            'total_processed': 0,
            'successful_separations': 0,
            'failed_separations': 0,
            'total_processing_time': 0.0
        }
    
    def print_stats(self):
        """
        In thống kê xử lý
        Print processing statistics
        """
        stats = self.get_processing_stats()
        
        print("=" * 50)
        print("SPLEETER PROCESSING STATISTICS")
        print("=" * 50)
        print(f"Total processed: {stats['total_processed']}")
        print(f"Successful: {stats['successful_separations']}")
        print(f"Failed: {stats['failed_separations']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Avg processing time: {stats['avg_processing_time']:.2f}s")
        print(f"Total processing time: {stats['total_processing_time']:.2f}s")
        print("=" * 50)
    
    def separate_audio_array(self, audio_data, sample_rate=22050, output_dir="spleeter_output"):
        """
        Tách âm thanh từ numpy array (in-memory processing)
        Separate audio from numpy array (in-memory processing)
        
        Args:
            audio_data: Numpy array chứa dữ liệu âm thanh
            sample_rate: Sample rate của âm thanh
            output_dir: Thư mục lưu kết quả (optional)
            
        Returns:
            dict: Dictionary chứa numpy arrays của các track đã tách
        """
        if not self.is_available():
            print("✗ Spleeter not available, skipping separation")
            return None
        
        try:
            import time
            import tempfile
            import soundfile as sf
            
            start_time = time.time()
            
            print(f"Separating audio array with Spleeter...")
            print(f"  Input shape: {audio_data.shape}, {sample_rate}Hz")
            print(f"  Model: {self.stems}")
            
            # Ensure audio is in correct format for Spleeter
            if len(audio_data.shape) == 1:
                # Convert mono to stereo for Spleeter
                waveform = np.stack([audio_data, audio_data], axis=-1)
            elif len(audio_data.shape) == 2 and audio_data.shape[1] == 1:
                # Convert mono column to stereo
                waveform = np.repeat(audio_data, 2, axis=1)
            else:
                # Already stereo or multi-channel
                waveform = audio_data
            
            print(f"  Prepared waveform shape: {waveform.shape}")
            
            # Thực hiện tách
            separated = self.separator.separate(waveform)
            
            # Chuyển đổi về format cần thiết và lưu file (optional)
            result_arrays = {}
            output_paths = {}
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            for stem, audio_data_sep in separated.items():
                # Convert back to mono if needed
                if len(audio_data_sep.shape) == 2:
                    # Take left channel or average both channels
                    if audio_data_sep.shape[1] == 2:
                        mono_audio = np.mean(audio_data_sep, axis=1)
                    else:
                        mono_audio = audio_data_sep[:, 0]
                else:
                    mono_audio = audio_data_sep
                
                result_arrays[stem] = mono_audio
                
                # Optionally save to file
                if output_dir:
                    output_path = os.path.join(output_dir, f"{stem}.wav")
                    sf.write(output_path, audio_data_sep, sample_rate, subtype='PCM_16')
                    output_paths[stem] = output_path
                    print(f"  ✓ Saved {stem}: {audio_data_sep.shape} -> {output_path}")
                else:
                    print(f"  ✓ Processed {stem}: {mono_audio.shape}")
            
            # Cập nhật thống kê
            processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            self.processing_stats['successful_separations'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            
            print(f"  ✓ Separation completed in {processing_time:.2f}s")
            
            # Return both arrays and file paths
            return {
                'arrays': result_arrays,
                'files': output_paths if output_dir else None
            }
            
        except Exception as e:
            print(f"✗ Spleeter array separation failed: {e}")
            import traceback
            traceback.print_exc()
            self.processing_stats['total_processed'] += 1
            self.processing_stats['failed_separations'] += 1
            return None

if __name__ == "__main__":
    # Test code
    processor = SpleeterProcessor()
    
    if processor.is_available():
        print("✓ SpleeterProcessor is ready!")
        processor.print_stats()
    else:
        print("✗ SpleeterProcessor is not available")
        print("Please install Spleeter: pip install spleeter")
    
    print("SpleeterProcessor test completed!")
