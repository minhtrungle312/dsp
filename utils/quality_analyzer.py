"""
Quality Analyzer Module - Module phân tích chất lượng âm thanh
=============================================================

Module này chứa lớp QualityAnalyzer để phân tích chất lượng âm thanh:
- Tính toán các metrics chất lượng
- So sánh audio trước và sau xử lý
- Tạo báo cáo chi tiết
- Đánh giá hiệu quả xử lý

Author: DSP Team
Date: 2025
"""

import os
import numpy as np
import librosa
from datetime import datetime

class QualityAnalyzer:
    """
    Lớp phân tích chất lượng âm thanh
    Quality analyzer for audio signal assessment
    
    Chức năng:
    - Tính toán metrics chất lượng âm thanh
    - So sánh audio gốc và đã xử lý
    - Tạo báo cáo so sánh chi tiết
    - Đánh giá hiệu quả các thuật toán
    """
    
    def __init__(self):
        """
        Khởi tạo QualityAnalyzer
        Initialize QualityAnalyzer
        """
        # Ngưỡng đánh giá chất lượng
        self.quality_thresholds = {
            'snr_excellent': 20,    # dB
            'snr_good': 10,         # dB
            'snr_poor': 5,          # dB
            'thd_excellent': 0.01,  # %
            'thd_good': 0.05,       # %
            'thd_poor': 0.1         # %
        }
        
        print("QualityAnalyzer initialized")
    
    def analyze_file(self, audio_path, sr=None):
        """
        Phân tích file âm thanh
        Analyze audio file
        
        Args:
            audio_path: Đường dẫn file âm thanh
            sr: Sample rate (None = auto detect)
            
        Returns:
            dict: Các metrics chất lượng
        """
        try:
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return None
            
            # Load audio
            audio, sample_rate = librosa.load(audio_path, sr=sr)
            
            # Tính các metrics
            metrics = self.calculate_metrics(audio, sample_rate)
            
            # Thêm thông tin file
            metrics['file_info'] = {
                'file_path': audio_path,
                'file_size_mb': os.path.getsize(audio_path) / (1024 * 1024),
                'duration_seconds': len(audio) / sample_rate,
                'sample_rate': sample_rate,
                'num_samples': len(audio)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing audio file {audio_path}: {e}")
            return None
    
    def analyze_audio_data(self, audio_data, sample_rate):
        """
        Analyze audio data directly (for demo.py compatibility).
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            dict: Quality metrics dictionary
        """
        try:
            print(f"Analyzing audio data: {len(audio_data)} samples at {sample_rate}Hz")
            
            # Use the analyze_file method by first creating a temporary file
            import tempfile
            import soundfile as sf
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sample_rate)
                
                # Analyze the temporary file
                metrics = self.analyze_file(tmp_file.name, sample_rate)
                
                # Clean up
                import os
                os.unlink(tmp_file.name)
                
                return metrics
                
        except Exception as e:
            print(f"Warning: Audio data analysis failed: {e}")
            # Return basic metrics on failure
            return {
                'rms_level': float(np.sqrt(np.mean(audio_data**2))),
                'peak_level': float(np.max(np.abs(audio_data))),
                'snr': 0.0,
                'thd': 0.0,
                'duration': len(audio_data) / sample_rate
            }

    def analyze_audio_file(self, file_path: str) -> dict:
        """
        Analyze audio file and return quality metrics (for main.py compatibility).
        
        Args:
            file_path: Path to audio file
            
        Returns:
            dict: Quality metrics dictionary
        """
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            print(f"Analyzing audio file: {file_path}")
            print(f"  Duration: {len(audio_data)/sample_rate:.2f}s")
            print(f"  Sample rate: {sample_rate}Hz")
            
            # Calculate metrics using existing method
            metrics = self.calculate_metrics(audio_data, sample_rate)
            
            if metrics:
                print(f"  ✓ Analysis completed with {len(metrics)} metrics")
                return metrics
            else:
                print("  ⚠ Analysis returned empty metrics")
                return self._get_default_metrics()
                
        except Exception as e:
            print(f"  ✗ Error analyzing file {file_path}: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> dict:
        """
        Get default metrics when analysis fails.
        
        Returns:
            dict: Default quality metrics
        """
        return {
            'rms_energy': 0.0,
            'peak_amplitude': 0.0,
            'dynamic_range_db': 0.0,
            'zero_crossing_rate': 0.0,
            'spectral_centroid': 0.0,
            'spectral_rolloff': 0.0,
            'spectral_bandwidth': 0.0,
            'spectral_flatness': 0.0,
            'estimated_snr_db': 0.0,
            'thd_percent': 0.0,
            'silence_ratio': 0.0,
            'loudness_lufs': -60.0,
            'tempo_bpm': 0.0,
            'rhythm_regularity': 0.0,
            'mfcc_mean': 0.0,
            'mfcc_std': 0.0,
            'chroma_mean': 0.0,
            'chroma_std': 0.0
        }
    
    def calculate_metrics(self, audio, sample_rate):
        """
        Tính toán các metrics chất lượng âm thanh
        Calculate audio quality metrics
        
        Args:
            audio: Dữ liệu âm thanh
            sample_rate: Sample rate
            
        Returns:
            dict: Dictionary chứa các metrics
        """
        metrics = {}
        
        try:
            # Metrics cơ bản - Basic metrics (safe calculations)
            metrics['rms_energy'] = np.sqrt(np.mean(audio**2))
            metrics['peak_amplitude'] = np.max(np.abs(audio))
            metrics['dynamic_range_db'] = self._calculate_dynamic_range(audio)
            
            # Librosa-based metrics (with error handling)
            try:
                metrics['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
            except Exception as e:
                print(f"Warning: Zero crossing rate calculation failed: {e}")
                metrics['zero_crossing_rate'] = 0.0
            
            try:
                # Metrics phổ tần - Spectral metrics  
                metrics['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
                metrics['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
                metrics['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))
                metrics['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=audio))
            except Exception as e:
                print(f"Warning: Spectral metrics calculation failed: {e}")
                metrics['spectral_centroid'] = 0.0
                metrics['spectral_rolloff'] = 0.0
                metrics['spectral_bandwidth'] = 0.0
                metrics['spectral_flatness'] = 0.0
            
            # Metrics âm thanh - Audio metrics (safe calculations)
            metrics['estimated_snr_db'] = self._estimate_snr(audio)
            metrics['thd_percent'] = self._estimate_thd(audio, sample_rate)
            metrics['silence_ratio'] = self._calculate_silence_ratio(audio)
            metrics['loudness_lufs'] = self._estimate_loudness(audio)
            
            # Metrics temporal - Temporal metrics
            try:
                tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
                metrics['tempo_bpm'] = tempo
                metrics['rhythm_regularity'] = self._calculate_rhythm_regularity(beats)
            except:
                metrics['tempo_bpm'] = 0
                metrics['rhythm_regularity'] = 0
            
            # MFCC features
            try:
                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
                metrics['mfcc_mean'] = np.mean(mfccs)
                metrics['mfcc_std'] = np.std(mfccs)
            except:
                metrics['mfcc_mean'] = 0
                metrics['mfcc_std'] = 0
            
            # Chroma features
            try:
                chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
                metrics['chroma_mean'] = np.mean(chroma)
                metrics['chroma_std'] = np.std(chroma)
            except:
                metrics['chroma_mean'] = 0
                metrics['chroma_std'] = 0
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    def _calculate_dynamic_range(self, audio):
        """
        Tính dải động (dB)
        Calculate dynamic range in dB
        
        Args:
            audio: Dữ liệu âm thanh
            
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
            audio: Dữ liệu âm thanh
            
        Returns:
            float: SNR ước tính (dB)
        """
        # Phương pháp đơn giản: sử dụng percentile
        sorted_audio = np.sort(np.abs(audio))
        
        # Noise floor (10% thấp nhất)
        noise_floor = np.mean(sorted_audio[:int(0.1 * len(sorted_audio))])
        
        # Signal level (10% cao nhất)
        signal_level = np.mean(sorted_audio[int(0.9 * len(sorted_audio)):])
        
        if noise_floor > 0:
            return 20 * np.log10(signal_level / noise_floor)
        else:
            return float('inf')
    
    def _estimate_thd(self, audio, sample_rate):
        """
        Ước tính Total Harmonic Distortion
        Estimate Total Harmonic Distortion
        
        Args:
            audio: Dữ liệu âm thanh
            sample_rate: Sample rate
            
        Returns:
            float: THD ước tính (%)
        """
        try:
            # Tính FFT
            fft = np.fft.fft(audio)
            fft_magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
            
            # Chỉ lấy phần dương
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
            
            # Tìm fundamental frequency (peak lớn nhất)
            fundamental_idx = np.argmax(positive_magnitude[1:]) + 1  # Bỏ qua DC
            fundamental_freq = positive_freqs[fundamental_idx]
            fundamental_magnitude = positive_magnitude[fundamental_idx]
            
            # Tìm harmonics (2f, 3f, 4f, ...)
            harmonic_power = 0
            for harmonic in range(2, 6):  # 2nd to 5th harmonics
                harmonic_freq = fundamental_freq * harmonic
                
                # Tìm bin gần nhất
                harmonic_idx = np.argmin(np.abs(positive_freqs - harmonic_freq))
                
                if harmonic_idx < len(positive_magnitude):
                    harmonic_power += positive_magnitude[harmonic_idx]**2
            
            # Tính THD
            fundamental_power = fundamental_magnitude**2
            if fundamental_power > 0:
                thd = np.sqrt(harmonic_power / fundamental_power) * 100
                return min(thd, 100)  # Cap at 100%
            else:
                return 0
                
        except Exception as e:
            print(f"Warning: THD calculation failed - {e}")
            return 0
    
    def _calculate_silence_ratio(self, audio, threshold_db=-40):
        """
        Tính tỷ lệ im lặng
        Calculate silence ratio
        
        Args:
            audio: Dữ liệu âm thanh
            threshold_db: Ngưỡng im lặng (dB)
            
        Returns:
            float: Tỷ lệ im lặng (0-1)
        """
        # Chuyển về dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # Đếm samples dưới ngưỡng
        silence_samples = np.sum(audio_db < threshold_db)
        
        return silence_samples / len(audio)
    
    def _estimate_loudness(self, audio):
        """
        Ước tính loudness (LUFS đơn giản)
        Estimate loudness (simple LUFS)
        
        Args:
            audio: Dữ liệu âm thanh
            
        Returns:
            float: Loudness ước tính (LUFS)
        """
        # Đây là ước tính đơn giản, không phải LUFS thực sự
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            # Chuyển về dB và điều chỉnh để gần với LUFS
            loudness = 20 * np.log10(rms) - 0.691  # Offset để gần với LUFS
            return loudness
        else:
            return -float('inf')
    
    def _calculate_rhythm_regularity(self, beats):
        """
        Tính độ đều đặn của nhịp
        Calculate rhythm regularity
        
        Args:
            beats: Mảng beat times
            
        Returns:
            float: Độ đều đặn (0-1)
        """
        if len(beats) < 3:
            return 0
        
        # Tính interval giữa các beats
        intervals = np.diff(beats)
        
        if len(intervals) == 0:
            return 0
        
        # Tính coefficient of variation (CV)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval > 0:
            cv = std_interval / mean_interval
            # Chuyển CV thành regularity (CV thấp = regularity cao)
            regularity = 1 / (1 + cv)
            return regularity
        else:
            return 0
    
    def compare_audio(self, original_metrics, processed_metrics):
        """
        So sánh metrics giữa audio gốc và đã xử lý
        Compare metrics between original and processed audio
        
        Args:
            original_metrics: Metrics của audio gốc
            processed_metrics: Metrics của audio đã xử lý
            
        Returns:
            dict: Kết quả so sánh
        """
        comparison = {}
        
        try:
            # SNR improvement
            snr_original = original_metrics.get('estimated_snr_db', 0)
            snr_processed = processed_metrics.get('estimated_snr_db', 0)
            comparison['snr_improvement_db'] = snr_processed - snr_original
            
            # Dynamic range change
            dr_original = original_metrics.get('dynamic_range_db', 0)
            dr_processed = processed_metrics.get('dynamic_range_db', 0)
            comparison['dynamic_range_change_db'] = dr_processed - dr_original
            
            # THD change
            thd_original = original_metrics.get('thd_percent', 0)
            thd_processed = processed_metrics.get('thd_percent', 0)
            comparison['thd_change_percent'] = thd_processed - thd_original
            
            # RMS change
            rms_original = original_metrics.get('rms_energy', 0)
            rms_processed = processed_metrics.get('rms_energy', 0)
            if rms_original > 0:
                comparison['rms_change_db'] = 20 * np.log10(rms_processed / rms_original)
            else:
                comparison['rms_change_db'] = 0
            
            # Spectral changes
            sc_original = original_metrics.get('spectral_centroid', 0)
            sc_processed = processed_metrics.get('spectral_centroid', 0)
            comparison['spectral_centroid_change_hz'] = sc_processed - sc_original
            
            # Silence ratio change
            silence_original = original_metrics.get('silence_ratio', 0)
            silence_processed = processed_metrics.get('silence_ratio', 0)
            comparison['silence_ratio_change'] = silence_processed - silence_original
            
            # Overall quality assessment
            comparison['quality_assessment'] = self._assess_overall_quality(comparison)
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing audio metrics: {e}")
            return {}
    
    def _assess_overall_quality(self, comparison):
        """
        Đánh giá chất lượng tổng thể
        Assess overall quality improvement
        
        Args:
            comparison: Kết quả so sánh
            
        Returns:
            dict: Đánh giá chất lượng
        """
        assessment = {
            'score': 0,
            'rating': 'Poor',
            'improvements': [],
            'degradations': []
        }
        
        score = 0
        
        # SNR improvement (trọng số cao)
        snr_improvement = comparison.get('snr_improvement_db', 0)
        if snr_improvement > 5:
            score += 30
            assessment['improvements'].append('Excellent noise reduction')
        elif snr_improvement > 2:
            score += 20
            assessment['improvements'].append('Good noise reduction')
        elif snr_improvement > 0:
            score += 10
            assessment['improvements'].append('Moderate noise reduction')
        else:
            assessment['degradations'].append('No SNR improvement')
        
        # THD change (trọng số trung bình)
        thd_change = comparison.get('thd_change_percent', 0)
        if thd_change < -0.5:
            score += 15
            assessment['improvements'].append('Reduced distortion')
        elif thd_change > 0.5:
            score -= 10
            assessment['degradations'].append('Increased distortion')
        
        # Dynamic range (trọng số thấp)
        dr_change = comparison.get('dynamic_range_change_db', 0)
        if dr_change > 2:
            score += 10
            assessment['improvements'].append('Improved dynamic range')
        elif dr_change < -5:
            score -= 5
            assessment['degradations'].append('Reduced dynamic range')
        
        # Silence reduction
        silence_change = comparison.get('silence_ratio_change', 0)
        if silence_change < -0.1:
            score += 5
            assessment['improvements'].append('Reduced silence/noise')
        
        # Determine rating
        assessment['score'] = max(0, min(100, score))
        
        if score >= 40:
            assessment['rating'] = 'Excellent'
        elif score >= 25:
            assessment['rating'] = 'Good'
        elif score >= 10:
            assessment['rating'] = 'Fair'
        else:
            assessment['rating'] = 'Poor'
        
        return assessment
    
    def create_comparison_report(self, original_path, processed_path, output_report="quality_report.txt"):
        """
        Tạo báo cáo so sánh chất lượng chi tiết
        Create detailed quality comparison report
        
        Args:
            original_path: Đường dẫn file gốc
            processed_path: Đường dẫn file đã xử lý
            output_report: Đường dẫn file báo cáo
            
        Returns:
            list: Nội dung báo cáo
        """
        print("Generating quality comparison report...")
        
        # Phân tích cả hai file
        original_metrics = self.analyze_file(original_path)
        processed_metrics = self.analyze_file(processed_path)
        
        if not original_metrics or not processed_metrics:
            print("Error: Could not analyze audio files")
            return None
        
        # So sánh
        comparison = self.compare_audio(original_metrics, processed_metrics)
        
        # Tạo báo cáo
        report_content = []
        report_content.append("=" * 80)
        report_content.append("FANCAM AUDIO ENHANCEMENT QUALITY REPORT")
        report_content.append("=" * 80)
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # File information
        report_content.append("FILE INFORMATION:")
        report_content.append("-" * 40)
        report_content.append(f"Original file: {os.path.basename(original_path)}")
        report_content.append(f"Processed file: {os.path.basename(processed_path)}")
        report_content.append(f"Original size: {original_metrics['file_info']['file_size_mb']:.2f} MB")
        report_content.append(f"Processed size: {processed_metrics['file_info']['file_size_mb']:.2f} MB")
        report_content.append(f"Duration: {original_metrics['file_info']['duration_seconds']:.2f}s")
        report_content.append("")
        
        # Original metrics
        report_content.append("ORIGINAL AUDIO METRICS:")
        report_content.append("-" * 40)
        self._add_metrics_to_report(report_content, original_metrics)
        
        # Processed metrics
        report_content.append("")
        report_content.append("PROCESSED AUDIO METRICS:")
        report_content.append("-" * 40)
        self._add_metrics_to_report(report_content, processed_metrics)
        
        # Comparison
        report_content.append("")
        report_content.append("IMPROVEMENT ANALYSIS:")
        report_content.append("-" * 40)
        self._add_comparison_to_report(report_content, comparison)
        
        # Overall assessment
        assessment = comparison.get('quality_assessment', {})
        report_content.append("")
        report_content.append("OVERALL QUALITY ASSESSMENT:")
        report_content.append("-" * 40)
        report_content.append(f"Score: {assessment.get('score', 0)}/100")
        report_content.append(f"Rating: {assessment.get('rating', 'Unknown')}")
        
        if assessment.get('improvements'):
            report_content.append("\\nImprovements:")
            for improvement in assessment['improvements']:
                report_content.append(f"  ✓ {improvement}")
        
        if assessment.get('degradations'):
            report_content.append("\\nDegradations:")
            for degradation in assessment['degradations']:
                report_content.append(f"  ✗ {degradation}")
        
        # Lưu báo cáo
        try:
            with open(output_report, 'w', encoding='utf-8') as f:
                f.write('\\n'.join(report_content))
            print(f"Quality report saved to: {output_report}")
        except Exception as e:
            print(f"Error saving report: {e}")
        
        return report_content
    
    def _add_metrics_to_report(self, report_content, metrics):
        """
        Thêm metrics vào báo cáo
        Add metrics to report
        
        Args:
            report_content: Nội dung báo cáo
            metrics: Metrics để thêm
        """
        # Basic metrics
        report_content.append(f"RMS Energy: {metrics.get('rms_energy', 0):.6f}")
        report_content.append(f"Peak Amplitude: {metrics.get('peak_amplitude', 0):.6f}")
        report_content.append(f"Dynamic Range: {metrics.get('dynamic_range_db', 0):.2f} dB")
        report_content.append(f"Estimated SNR: {metrics.get('estimated_snr_db', 0):.2f} dB")
        report_content.append(f"THD: {metrics.get('thd_percent', 0):.3f}%")
        report_content.append(f"Silence Ratio: {metrics.get('silence_ratio', 0):.3f}")
        
        # Spectral metrics
        report_content.append(f"Spectral Centroid: {metrics.get('spectral_centroid', 0):.2f} Hz")
        report_content.append(f"Spectral Rolloff: {metrics.get('spectral_rolloff', 0):.2f} Hz")
        report_content.append(f"Spectral Bandwidth: {metrics.get('spectral_bandwidth', 0):.2f} Hz")
        report_content.append(f"Zero Crossing Rate: {metrics.get('zero_crossing_rate', 0):.6f}")
        
        # Other metrics
        report_content.append(f"Estimated Loudness: {metrics.get('loudness_lufs', 0):.2f} LUFS")
        report_content.append(f"Tempo: {metrics.get('tempo_bpm', 0):.1f} BPM")
    
    def _add_comparison_to_report(self, report_content, comparison):
        """
        Thêm so sánh vào báo cáo
        Add comparison to report
        
        Args:
            report_content: Nội dung báo cáo
            comparison: Kết quả so sánh
        """
        snr_improvement = comparison.get('snr_improvement_db', 0)
        dr_change = comparison.get('dynamic_range_change_db', 0)
        thd_change = comparison.get('thd_change_percent', 0)
        rms_change = comparison.get('rms_change_db', 0)
        
        report_content.append(f"SNR Improvement: {snr_improvement:+.2f} dB")
        report_content.append(f"Dynamic Range Change: {dr_change:+.2f} dB")
        report_content.append(f"THD Change: {thd_change:+.3f}%")
        report_content.append(f"RMS Change: {rms_change:+.2f} dB")
        report_content.append(f"Spectral Centroid Change: {comparison.get('spectral_centroid_change_hz', 0):+.2f} Hz")
        report_content.append(f"Silence Ratio Change: {comparison.get('silence_ratio_change', 0):+.3f}")

if __name__ == "__main__":
    # Test code
    analyzer = QualityAnalyzer()
    
    # Test với audio giả
    test_audio = np.random.randn(22050)  # 1 second
    metrics = analyzer.calculate_metrics(test_audio, 22050)
    
    print("Test metrics calculated:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("QualityAnalyzer test completed!")
