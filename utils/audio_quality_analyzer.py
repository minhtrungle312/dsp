#!/usr/bin/env python3
"""
Audio Quality Analyzer
======================

Comprehensive audio quality analysis system for detecting and measuring:
- Clipping and distortion
- Audio artifacts 
- Dynamic range
- Signal-to-noise ratio
- Frequency balance
- Loudness and dynamics

Author: DSP Team
Date: 2025
"""

import numpy as np
import librosa
import scipy.signal
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
from pathlib import Path
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class AudioQualityAnalyzer:
    """Comprehensive audio quality analysis and reporting system."""
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the audio quality analyzer.
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.thresholds = {
            'clipping_percentage': 0.1,  # Max 0.1% clipping acceptable
            'dynamic_range_min': 10.0,   # Minimum 10 dB dynamic range
            'snr_min': 20.0,             # Minimum 20 dB SNR
            'thd_max': 5.0,              # Maximum 5% THD
            'spectral_centroid_min': 1000,  # Minimum spectral centroid
            'spectral_centroid_max': 8000,  # Maximum spectral centroid
            'zero_crossing_rate_max': 0.3,  # Maximum zero crossing rate
            'rms_energy_min': 0.001,    # Minimum RMS energy
            'peak_to_rms_ratio_max': 20.0,  # Maximum peak-to-RMS ratio (dB)
        }
    
    def analyze_audio(self, audio_data: np.ndarray, label: str = "Audio") -> Dict[str, Any]:
        """
        Perform comprehensive audio quality analysis.
        
        Args:
            audio_data: Audio signal as numpy array
            label: Label for this analysis (e.g., "Original", "Enhanced")
            
        Returns:
            Dictionary containing all quality metrics and analysis
        """
        if audio_data.ndim > 1:
            # Convert to mono if stereo
            audio_data = np.mean(audio_data, axis=0)
        
        # Normalize to prevent overflow in calculations
        if np.max(np.abs(audio_data)) > 0:
            audio_normalized = audio_data / np.max(np.abs(audio_data))
        else:
            audio_normalized = audio_data
        
        analysis = {
            'label': label,
            'basic_stats': self._analyze_basic_stats(audio_data),
            'clipping': self._analyze_clipping(audio_data),
            'dynamic_range': self._analyze_dynamic_range(audio_data),
            'noise_analysis': self._analyze_noise(audio_data),
            'distortion': self._analyze_distortion(audio_data),
            'frequency_analysis': self._analyze_frequency_content(audio_data),
            'temporal_analysis': self._analyze_temporal_features(audio_data),
            'artifacts': self._detect_artifacts(audio_data),
            'overall_quality': None  # Will be calculated at the end
        }
        
        # Calculate overall quality score
        analysis['overall_quality'] = self._calculate_overall_quality(analysis)
        
        return analysis
    
    def _analyze_basic_stats(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze basic statistical properties of the audio."""
        try:
            stats = {
                'duration': len(audio_data) / self.sample_rate,
                'sample_count': len(audio_data),
                'max_amplitude': float(np.max(np.abs(audio_data))),
                'rms_energy': float(np.sqrt(np.mean(audio_data ** 2))),
                'peak_amplitude': float(np.max(audio_data)),
                'min_amplitude': float(np.min(audio_data)),
                'mean': float(np.mean(audio_data)),
                'std_dev': float(np.std(audio_data)),
                'skewness': float(stats.skew(audio_data)),
                'kurtosis': float(stats.kurtosis(audio_data))
            }
            
            # Calculate peak-to-RMS ratio in dB
            if stats['rms_energy'] > 0:
                stats['peak_to_rms_db'] = 20 * np.log10(stats['max_amplitude'] / stats['rms_energy'])
            else:
                stats['peak_to_rms_db'] = 0.0
                
            return stats
        except Exception as e:
            self.logger.warning(f"Error in basic stats analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_clipping(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Detect and analyze audio clipping."""
        try:
            # Detect clipping (samples at or near maximum amplitude)
            threshold = 0.99  # 99% of maximum amplitude
            clipped_samples = np.sum(np.abs(audio_data) >= threshold)
            clipping_percentage = (clipped_samples / len(audio_data)) * 100
            
            # Detect consecutive clipped samples (more serious clipping)
            clipped_regions = []
            in_clip = False
            clip_start = 0
            
            for i, sample in enumerate(np.abs(audio_data)):
                if sample >= threshold:
                    if not in_clip:
                        in_clip = True
                        clip_start = i
                else:
                    if in_clip:
                        in_clip = False
                        clipped_regions.append(i - clip_start)
            
            return {
                'clipped_samples': int(clipped_samples),
                'clipping_percentage': float(clipping_percentage),
                'max_consecutive_clips': int(max(clipped_regions)) if clipped_regions else 0,
                'clipping_regions_count': len(clipped_regions),
                'severity': 'high' if clipping_percentage > 1.0 else 'medium' if clipping_percentage > 0.1 else 'low'
            }
        except Exception as e:
            self.logger.warning(f"Error in clipping analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_dynamic_range(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze dynamic range of the audio."""
        try:
            # Calculate RMS in overlapping windows
            window_size = int(0.1 * self.sample_rate)  # 100ms windows
            hop_size = window_size // 2
            
            rms_values = []
            for i in range(0, len(audio_data) - window_size, hop_size):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                if rms > 0:
                    rms_values.append(20 * np.log10(rms))  # Convert to dB
            
            if not rms_values:
                return {'error': 'No valid RMS values calculated'}
            
            rms_values = np.array(rms_values)
            
            # Calculate dynamic range metrics
            dynamic_range = float(np.max(rms_values) - np.min(rms_values))
            rms_mean = float(np.mean(rms_values))
            rms_std = float(np.std(rms_values))
            
            # Calculate crest factor (peak to average ratio)
            peak_db = 20 * np.log10(np.max(np.abs(audio_data))) if np.max(np.abs(audio_data)) > 0 else -np.inf
            crest_factor = peak_db - rms_mean
            
            return {
                'dynamic_range_db': dynamic_range,
                'rms_mean_db': rms_mean,
                'rms_std_db': rms_std,
                'crest_factor_db': float(crest_factor),
                'peak_db': float(peak_db),
                'loudness_range': float(np.percentile(rms_values, 90) - np.percentile(rms_values, 10))
            }
        except Exception as e:
            self.logger.warning(f"Error in dynamic range analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_noise(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze noise characteristics and estimate SNR."""
        try:
            # Estimate noise floor using quiet segments
            window_size = int(0.1 * self.sample_rate)
            hop_size = window_size // 2
            
            energy_values = []
            for i in range(0, len(audio_data) - window_size, hop_size):
                window = audio_data[i:i + window_size]
                energy = np.mean(window ** 2)
                energy_values.append(energy)
            
            # Assume bottom 10% of energy values represent noise
            noise_threshold = np.percentile(energy_values, 10)
            signal_energy = np.percentile(energy_values, 90)
            
            # Calculate SNR
            if noise_threshold > 0:
                snr_db = 10 * np.log10(signal_energy / noise_threshold)
            else:
                snr_db = np.inf
            
            # Estimate noise floor
            noise_floor_db = 10 * np.log10(noise_threshold) if noise_threshold > 0 else -np.inf
            
            # Analyze frequency domain noise
            fft = np.fft.fft(audio_data)
            magnitude_spectrum = np.abs(fft)
            
            # High frequency noise analysis (above 8kHz)
            nyquist = self.sample_rate // 2
            high_freq_start = int(8000 * len(fft) / (2 * nyquist))
            high_freq_noise = np.mean(magnitude_spectrum[high_freq_start:high_freq_start + len(fft)//4])
            
            return {
                'snr_db': float(snr_db),
                'noise_floor_db': float(noise_floor_db),
                'noise_threshold': float(noise_threshold),
                'signal_energy': float(signal_energy),
                'high_freq_noise_level': float(high_freq_noise),
                'noise_quality': 'good' if snr_db > 30 else 'acceptable' if snr_db > 20 else 'poor'
            }
        except Exception as e:
            self.logger.warning(f"Error in noise analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_distortion(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze harmonic distortion and other distortion metrics."""
        try:
            # Calculate Total Harmonic Distortion (THD) approximation
            # Using spectral analysis approach
            
            # Window the signal
            windowed = audio_data * np.hanning(len(audio_data))
            
            # FFT analysis
            fft = np.fft.fft(windowed)
            magnitude_spectrum = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
            
            # Find fundamental frequency (strongest component in reasonable vocal range)
            vocal_range = (freqs >= 80) & (freqs <= 1000)
            if np.any(vocal_range):
                fundamental_idx = np.argmax(magnitude_spectrum[vocal_range])
                fundamental_freq = freqs[vocal_range][fundamental_idx]
                fundamental_magnitude = magnitude_spectrum[vocal_range][fundamental_idx]
                
                # Calculate harmonics
                harmonic_energy = 0
                harmonic_count = 0
                
                for harmonic in range(2, 6):  # 2nd to 5th harmonics
                    harmonic_freq = fundamental_freq * harmonic
                    if harmonic_freq < freqs[-1]:
                        # Find closest frequency bin
                        harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                        harmonic_energy += magnitude_spectrum[harmonic_idx] ** 2
                        harmonic_count += 1
                
                # Calculate THD approximation
                if fundamental_magnitude > 0 and harmonic_count > 0:
                    thd_percent = 100 * np.sqrt(harmonic_energy) / fundamental_magnitude
                else:
                    thd_percent = 0.0
            else:
                thd_percent = 0.0
                fundamental_freq = 0.0
            
            # Calculate spectral distortion metrics
            spectral_flatness = self._calculate_spectral_flatness(magnitude_spectrum)
            spectral_entropy = self._calculate_spectral_entropy(magnitude_spectrum)
            
            return {
                'thd_percent': float(thd_percent),
                'fundamental_frequency': float(fundamental_freq),
                'spectral_flatness': float(spectral_flatness),
                'spectral_entropy': float(spectral_entropy),
                'distortion_level': 'high' if thd_percent > 5 else 'medium' if thd_percent > 2 else 'low'
            }
        except Exception as e:
            self.logger.warning(f"Error in distortion analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_frequency_content(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze frequency content and balance."""
        try:
            # Calculate spectral features using librosa
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # Calculate MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            
            # Analyze frequency bands
            fft = np.fft.fft(audio_data)
            magnitude_spectrum = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
            
            # Define frequency bands
            bands = {
                'sub_bass': (20, 60),
                'bass': (60, 250),
                'low_mid': (250, 500),
                'mid': (500, 2000),
                'high_mid': (2000, 4000),
                'presence': (4000, 6000),
                'brilliance': (6000, 20000)
            }
            
            band_energies = {}
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    band_energies[f'{band_name}_energy'] = float(np.sum(magnitude_spectrum[band_mask] ** 2))
                else:
                    band_energies[f'{band_name}_energy'] = 0.0
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'mfcc_mean': [float(x) for x in np.mean(mfccs, axis=1)],
                'mfcc_std': [float(x) for x in np.std(mfccs, axis=1)],
                **band_energies
            }
        except Exception as e:
            self.logger.warning(f"Error in frequency analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_temporal_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze temporal characteristics of the audio."""
        try:
            # Calculate onset detection
            onset_frames = librosa.onset.onset_detect(y=audio_data, sr=self.sample_rate)
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
            
            # Calculate tempo
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            
            # Calculate rhythm regularity
            if len(onset_times) > 1:
                inter_onset_intervals = np.diff(onset_times)
                rhythm_regularity = 1.0 / (1.0 + np.std(inter_onset_intervals))
            else:
                rhythm_regularity = 0.0
            
            # Calculate attack time (rise time)
            # Find the steepest rise in amplitude
            envelope = np.abs(audio_data)
            smoothed_envelope = scipy.signal.savgol_filter(envelope, 
                                                         window_length=min(101, len(envelope)//4*2+1), 
                                                         polyorder=3)
            
            # Find maximum gradient
            gradient = np.gradient(smoothed_envelope)
            max_gradient_idx = np.argmax(gradient)
            
            # Estimate attack time (simplified)
            attack_samples = max_gradient_idx if max_gradient_idx > 0 else 1
            attack_time_ms = (attack_samples / self.sample_rate) * 1000
            
            return {
                'onset_count': len(onset_times),
                'onset_rate': float(len(onset_times) / (len(audio_data) / self.sample_rate)),
                'tempo_bpm': float(tempo),
                'rhythm_regularity': float(rhythm_regularity),
                'attack_time_ms': float(attack_time_ms),
                'beat_count': len(beats)
            }
        except Exception as e:
            self.logger.warning(f"Error in temporal analysis: {e}")
            return {'error': str(e)}
    
    def _detect_artifacts(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Detect various audio artifacts."""
        try:
            artifacts = {
                'clicks_pops': self._detect_clicks_pops(audio_data),
                'dropouts': self._detect_dropouts(audio_data),
                'dc_offset': self._detect_dc_offset(audio_data),
                'phase_issues': self._detect_phase_issues(audio_data),
                'aliasing': self._detect_aliasing(audio_data)
            }
            
            # Overall artifact score
            artifact_count = sum([
                artifacts['clicks_pops']['detected'],
                artifacts['dropouts']['detected'],
                artifacts['dc_offset']['detected'],
                artifacts['phase_issues']['detected'],
                artifacts['aliasing']['detected']
            ])
            
            artifacts['overall_artifact_score'] = artifact_count
            artifacts['artifact_level'] = 'high' if artifact_count >= 3 else 'medium' if artifact_count >= 2 else 'low'
            
            return artifacts
        except Exception as e:
            self.logger.warning(f"Error in artifact detection: {e}")
            return {'error': str(e)}
    
    def _detect_clicks_pops(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Detect clicks and pops in the audio."""
        try:
            # Calculate first derivative to detect sudden changes
            diff = np.diff(audio_data)
            
            # Find sudden spikes in the derivative
            threshold = np.std(diff) * 5  # 5 standard deviations
            spike_indices = np.where(np.abs(diff) > threshold)[0]
            
            # Group nearby spikes as single events
            click_events = []
            if len(spike_indices) > 0:
                current_group = [spike_indices[0]]
                for idx in spike_indices[1:]:
                    if idx - current_group[-1] <= 10:  # Within 10 samples
                        current_group.append(idx)
                    else:
                        click_events.append(current_group)
                        current_group = [idx]
                click_events.append(current_group)
            
            return {
                'detected': len(click_events) > 0,
                'click_count': len(click_events),
                'severity': 'high' if len(click_events) > 10 else 'medium' if len(click_events) > 3 else 'low',
                'max_spike_amplitude': float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0
            }
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_dropouts(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Detect audio dropouts (periods of very low amplitude)."""
        try:
            # Calculate RMS in small windows
            window_size = int(0.01 * self.sample_rate)  # 10ms windows
            rms_values = []
            
            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                rms_values.append(rms)
            
            if not rms_values:
                return {'detected': False, 'dropout_count': 0}
            
            # Define dropout threshold (very low RMS)
            median_rms = np.median(rms_values)
            dropout_threshold = median_rms * 0.01  # 1% of median RMS
            
            # Find consecutive dropout windows
            dropouts = np.array(rms_values) < dropout_threshold
            dropout_regions = []
            
            in_dropout = False
            dropout_start = 0
            
            for i, is_dropout in enumerate(dropouts):
                if is_dropout and not in_dropout:
                    in_dropout = True
                    dropout_start = i
                elif not is_dropout and in_dropout:
                    in_dropout = False
                    dropout_regions.append(i - dropout_start)
            
            return {
                'detected': len(dropout_regions) > 0,
                'dropout_count': len(dropout_regions),
                'max_dropout_duration_ms': float(max(dropout_regions) * window_size / self.sample_rate * 1000) if dropout_regions else 0.0,
                'total_dropout_time_ms': float(sum(dropout_regions) * window_size / self.sample_rate * 1000)
            }
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_dc_offset(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Detect DC offset in the audio."""
        try:
            mean_value = np.mean(audio_data)
            dc_offset_percentage = abs(mean_value) * 100
            
            return {
                'detected': dc_offset_percentage > 1.0,  # More than 1% offset
                'dc_offset_value': float(mean_value),
                'dc_offset_percentage': float(dc_offset_percentage),
                'severity': 'high' if dc_offset_percentage > 5 else 'medium' if dc_offset_percentage > 2 else 'low'
            }
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_phase_issues(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Detect potential phase issues (simplified check)."""
        try:
            # For mono audio, check for unusual phase characteristics
            # Using autocorrelation to detect phase anomalies
            
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Normalize
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
            
            # Check for unusual autocorrelation pattern
            # Strong negative correlation at short delays indicates phase issues
            short_delay_corr = np.min(autocorr[1:min(100, len(autocorr))])
            
            return {
                'detected': short_delay_corr < -0.5,
                'min_short_delay_correlation': float(short_delay_corr),
                'severity': 'high' if short_delay_corr < -0.8 else 'medium' if short_delay_corr < -0.5 else 'low'
            }
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_aliasing(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Detect potential aliasing artifacts."""
        try:
            # FFT analysis to check for energy near Nyquist frequency
            fft = np.fft.fft(audio_data)
            magnitude_spectrum = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
            
            # Check energy in high frequency region (near Nyquist)
            nyquist = self.sample_rate / 2
            high_freq_start = int(0.9 * nyquist)
            high_freq_mask = freqs >= high_freq_start
            
            if np.any(high_freq_mask):
                high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask] ** 2)
                total_energy = np.sum(magnitude_spectrum ** 2)
                high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
                
                # Suspicious if more than 5% of energy is in the top 10% of frequency range
                aliasing_detected = high_freq_ratio > 0.05
                
                return {
                    'detected': aliasing_detected,
                    'high_freq_energy_ratio': float(high_freq_ratio),
                    'severity': 'high' if high_freq_ratio > 0.15 else 'medium' if high_freq_ratio > 0.05 else 'low'
                }
            else:
                return {'detected': False, 'high_freq_energy_ratio': 0.0}
                
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _calculate_spectral_flatness(self, magnitude_spectrum: np.ndarray) -> float:
        """Calculate spectral flatness (Wiener entropy)."""
        try:
            # Avoid log of zero
            magnitude_spectrum = magnitude_spectrum + 1e-10
            
            geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum)))
            arithmetic_mean = np.mean(magnitude_spectrum)
            
            return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_spectral_entropy(self, magnitude_spectrum: np.ndarray) -> float:
        """Calculate spectral entropy."""
        try:
            # Normalize to get probability distribution
            total_energy = np.sum(magnitude_spectrum ** 2)
            if total_energy == 0:
                return 0.0
            
            prob_dist = (magnitude_spectrum ** 2) / total_energy
            prob_dist = prob_dist + 1e-10  # Avoid log of zero
            
            entropy = -np.sum(prob_dist * np.log2(prob_dist))
            return entropy
        except:
            return 0.0
    
    def _calculate_overall_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality score based on all metrics."""
        try:
            score = 100.0  # Start with perfect score
            issues = []
            recommendations = []
            
            # Check clipping
            clipping = analysis.get('clipping', {})
            if isinstance(clipping, dict) and 'clipping_percentage' in clipping:
                if clipping['clipping_percentage'] > self.thresholds['clipping_percentage']:
                    penalty = min(30, clipping['clipping_percentage'] * 10)
                    score -= penalty
                    issues.append(f"Clipping detected ({clipping['clipping_percentage']:.2f}%)")
                    recommendations.append("Reduce input levels to prevent clipping")
            
            # Check dynamic range
            dynamic_range = analysis.get('dynamic_range', {})
            if isinstance(dynamic_range, dict) and 'dynamic_range_db' in dynamic_range:
                if dynamic_range['dynamic_range_db'] < self.thresholds['dynamic_range_min']:
                    penalty = (self.thresholds['dynamic_range_min'] - dynamic_range['dynamic_range_db']) * 2
                    score -= penalty
                    issues.append(f"Low dynamic range ({dynamic_range['dynamic_range_db']:.1f} dB)")
                    recommendations.append("Avoid over-compression to preserve dynamics")
            
            # Check SNR
            noise = analysis.get('noise_analysis', {})
            if isinstance(noise, dict) and 'snr_db' in noise:
                if noise['snr_db'] < self.thresholds['snr_min']:
                    penalty = (self.thresholds['snr_min'] - noise['snr_db']) * 1.5
                    score -= penalty
                    issues.append(f"Low signal-to-noise ratio ({noise['snr_db']:.1f} dB)")
                    recommendations.append("Apply noise reduction to improve SNR")
            
            # Check distortion
            distortion = analysis.get('distortion', {})
            if isinstance(distortion, dict) and 'thd_percent' in distortion:
                if distortion['thd_percent'] > self.thresholds['thd_max']:
                    penalty = distortion['thd_percent'] * 3
                    score -= penalty
                    issues.append(f"High harmonic distortion ({distortion['thd_percent']:.1f}%)")
                    recommendations.append("Check for equipment issues or input overload")
            
            # Check artifacts
            artifacts = analysis.get('artifacts', {})
            if isinstance(artifacts, dict) and 'overall_artifact_score' in artifacts:
                artifact_penalty = artifacts['overall_artifact_score'] * 10
                score -= artifact_penalty
                if artifacts['overall_artifact_score'] > 0:
                    issues.append(f"Audio artifacts detected (score: {artifacts['overall_artifact_score']})")
                    recommendations.append("Review audio processing chain for artifact sources")
            
            # Ensure score doesn't go below 0
            score = max(0.0, score)
            
            # Determine quality grade
            if score >= 90:
                grade = "Excellent"
            elif score >= 80:
                grade = "Good"
            elif score >= 70:
                grade = "Fair"
            elif score >= 60:
                grade = "Poor"
            else:
                grade = "Very Poor"
            
            return {
                'overall_score': float(score),
                'grade': grade,
                'issues': issues,
                'recommendations': recommendations,
                'analysis_timestamp': str(np.datetime64('now'))
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating overall quality: {e}")
            return {
                'overall_score': 0.0,
                'grade': "Unknown",
                'issues': [f"Analysis error: {e}"],
                'recommendations': ["Please check audio file and try again"],
                'error': str(e)
            }
    
    def compare_audio_quality(self, original_analysis: Dict[str, Any], 
                            enhanced_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare audio quality between original and enhanced versions.
        
        Args:
            original_analysis: Analysis results for original audio
            enhanced_analysis: Analysis results for enhanced audio
            
        Returns:
            Comparison report with improvements and degradations
        """
        try:
            comparison = {
                'improvement_summary': {},
                'degradation_summary': {},
                'overall_assessment': '',
                'recommendations': []
            }
            
            # Compare overall scores
            orig_score = original_analysis.get('overall_quality', {}).get('overall_score', 0)
            enh_score = enhanced_analysis.get('overall_quality', {}).get('overall_score', 0)
            score_improvement = enh_score - orig_score
            
            comparison['overall_score_change'] = {
                'original': orig_score,
                'enhanced': enh_score,
                'improvement': score_improvement,
                'percentage_change': (score_improvement / orig_score * 100) if orig_score > 0 else 0
            }
            
            # Compare specific metrics
            metrics_to_compare = [
                ('clipping', 'clipping_percentage', 'lower_is_better'),
                ('dynamic_range', 'dynamic_range_db', 'higher_is_better'),
                ('noise_analysis', 'snr_db', 'higher_is_better'),
                ('distortion', 'thd_percent', 'lower_is_better'),
                ('artifacts', 'overall_artifact_score', 'lower_is_better')
            ]
            
            for category, metric, direction in metrics_to_compare:
                orig_val = self._get_nested_value(original_analysis, [category, metric])
                enh_val = self._get_nested_value(enhanced_analysis, [category, metric])
                
                if orig_val is not None and enh_val is not None:
                    change = enh_val - orig_val
                    improvement = (change > 0) if direction == 'higher_is_better' else (change < 0)
                    
                    comparison_data = {
                        'original': orig_val,
                        'enhanced': enh_val,
                        'change': change,
                        'improved': improvement
                    }
                    
                    if improvement:
                        comparison['improvement_summary'][f'{category}_{metric}'] = comparison_data
                    elif change != 0:
                        comparison['degradation_summary'][f'{category}_{metric}'] = comparison_data
            
            # Generate overall assessment
            if score_improvement > 10:
                comparison['overall_assessment'] = "Significant improvement in audio quality"
            elif score_improvement > 5:
                comparison['overall_assessment'] = "Moderate improvement in audio quality"
            elif score_improvement > 0:
                comparison['overall_assessment'] = "Slight improvement in audio quality"
            elif score_improvement > -5:
                comparison['overall_assessment'] = "Minor changes in audio quality"
            else:
                comparison['overall_assessment'] = "Quality degradation detected"
            
            # Generate recommendations
            if score_improvement > 0:
                comparison['recommendations'].append("Enhancement process was successful")
            else:
                comparison['recommendations'].append("Consider adjusting enhancement parameters")
                
            if len(comparison['degradation_summary']) > 0:
                comparison['recommendations'].append("Review degraded metrics and adjust processing accordingly")
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"Error in quality comparison: {e}")
            return {'error': str(e)}
    
    def _get_nested_value(self, data: Dict, keys: List[str]) -> Any:
        """Safely get nested dictionary value."""
        try:
            for key in keys:
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    return None
            return data
        except:
            return None
    
    def generate_report(self, analysis: Dict[str, Any], 
                       comparison: Optional[Dict[str, Any]] = None,
                       output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive quality analysis report.
        
        Args:
            analysis: Analysis results
            comparison: Optional comparison results
            output_file: Optional file path to save report
            
        Returns:
            Report as formatted string
        """
        try:
            report_lines = []
            
            # Header
            report_lines.append("=" * 80)
            report_lines.append("AUDIO QUALITY ANALYSIS REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Analysis Label: {analysis.get('label', 'Unknown')}")
            report_lines.append(f"Generated: {analysis.get('overall_quality', {}).get('analysis_timestamp', 'Unknown')}")
            report_lines.append("")
            
            # Overall Quality Score
            overall = analysis.get('overall_quality', {})
            if overall:
                report_lines.append("OVERALL QUALITY ASSESSMENT")
                report_lines.append("-" * 30)
                report_lines.append(f"Quality Score: {overall.get('overall_score', 0):.1f}/100")
                report_lines.append(f"Quality Grade: {overall.get('grade', 'Unknown')}")
                report_lines.append("")
                
                if overall.get('issues'):
                    report_lines.append("Issues Detected:")
                    for issue in overall['issues']:
                        report_lines.append(f"  • {issue}")
                    report_lines.append("")
                
                if overall.get('recommendations'):
                    report_lines.append("Recommendations:")
                    for rec in overall['recommendations']:
                        report_lines.append(f"  • {rec}")
                    report_lines.append("")
            
            # Basic Statistics
            basic = analysis.get('basic_stats', {})
            if basic and 'error' not in basic:
                report_lines.append("BASIC AUDIO STATISTICS")
                report_lines.append("-" * 30)
                report_lines.append(f"Duration: {basic.get('duration', 0):.2f} seconds")
                report_lines.append(f"Peak Amplitude: {basic.get('peak_amplitude', 0):.3f}")
                report_lines.append(f"RMS Energy: {basic.get('rms_energy', 0):.3f}")
                report_lines.append(f"Peak-to-RMS Ratio: {basic.get('peak_to_rms_db', 0):.1f} dB")
                report_lines.append("")
            
            # Clipping Analysis
            clipping = analysis.get('clipping', {})
            if clipping and 'error' not in clipping:
                report_lines.append("CLIPPING ANALYSIS")
                report_lines.append("-" * 30)
                report_lines.append(f"Clipping Percentage: {clipping.get('clipping_percentage', 0):.3f}%")
                report_lines.append(f"Clipped Samples: {clipping.get('clipped_samples', 0)}")
                report_lines.append(f"Severity: {clipping.get('severity', 'Unknown').title()}")
                report_lines.append("")
            
            # Dynamic Range
            dynamic = analysis.get('dynamic_range', {})
            if dynamic and 'error' not in dynamic:
                report_lines.append("DYNAMIC RANGE ANALYSIS")
                report_lines.append("-" * 30)
                report_lines.append(f"Dynamic Range: {dynamic.get('dynamic_range_db', 0):.1f} dB")
                report_lines.append(f"Crest Factor: {dynamic.get('crest_factor_db', 0):.1f} dB")
                report_lines.append(f"RMS Mean: {dynamic.get('rms_mean_db', 0):.1f} dB")
                report_lines.append("")
            
            # Noise Analysis
            noise = analysis.get('noise_analysis', {})
            if noise and 'error' not in noise:
                report_lines.append("NOISE ANALYSIS")
                report_lines.append("-" * 30)
                report_lines.append(f"Signal-to-Noise Ratio: {noise.get('snr_db', 0):.1f} dB")
                report_lines.append(f"Noise Floor: {noise.get('noise_floor_db', 0):.1f} dB")
                report_lines.append(f"Noise Quality: {noise.get('noise_quality', 'Unknown').title()}")
                report_lines.append("")
            
            # Distortion Analysis
            distortion = analysis.get('distortion', {})
            if distortion and 'error' not in distortion:
                report_lines.append("DISTORTION ANALYSIS")
                report_lines.append("-" * 30)
                report_lines.append(f"Total Harmonic Distortion: {distortion.get('thd_percent', 0):.2f}%")
                report_lines.append(f"Fundamental Frequency: {distortion.get('fundamental_frequency', 0):.1f} Hz")
                report_lines.append(f"Distortion Level: {distortion.get('distortion_level', 'Unknown').title()}")
                report_lines.append("")
            
            # Artifacts Detection
            artifacts = analysis.get('artifacts', {})
            if artifacts and 'error' not in artifacts:
                report_lines.append("ARTIFACTS DETECTION")
                report_lines.append("-" * 30)
                report_lines.append(f"Overall Artifact Score: {artifacts.get('overall_artifact_score', 0)}")
                report_lines.append(f"Artifact Level: {artifacts.get('artifact_level', 'Unknown').title()}")
                
                artifact_details = []
                if artifacts.get('clicks_pops', {}).get('detected'):
                    artifact_details.append("Clicks/Pops detected")
                if artifacts.get('dropouts', {}).get('detected'):
                    artifact_details.append("Audio dropouts detected")
                if artifacts.get('dc_offset', {}).get('detected'):
                    artifact_details.append("DC offset detected")
                if artifacts.get('phase_issues', {}).get('detected'):
                    artifact_details.append("Phase issues detected")
                if artifacts.get('aliasing', {}).get('detected'):
                    artifact_details.append("Aliasing detected")
                
                if artifact_details:
                    report_lines.append("Detected Artifacts:")
                    for detail in artifact_details:
                        report_lines.append(f"  • {detail}")
                else:
                    report_lines.append("No significant artifacts detected")
                report_lines.append("")
            
            # Comparison Report
            if comparison:
                report_lines.append("QUALITY COMPARISON")
                report_lines.append("-" * 30)
                score_change = comparison.get('overall_score_change', {})
                report_lines.append(f"Overall Assessment: {comparison.get('overall_assessment', 'Unknown')}")
                report_lines.append(f"Score Change: {score_change.get('improvement', 0):.1f} points ({score_change.get('percentage_change', 0):.1f}%)")
                
                improvements = comparison.get('improvement_summary', {})
                if improvements:
                    report_lines.append("\nImprovements:")
                    for metric, data in improvements.items():
                        report_lines.append(f"  • {metric}: {data['original']:.3f} → {data['enhanced']:.3f}")
                
                degradations = comparison.get('degradation_summary', {})
                if degradations:
                    report_lines.append("\nDegradations:")
                    for metric, data in degradations.items():
                        report_lines.append(f"  • {metric}: {data['original']:.3f} → {data['enhanced']:.3f}")
                
                comp_recs = comparison.get('recommendations', [])
                if comp_recs:
                    report_lines.append("\nComparison Recommendations:")
                    for rec in comp_recs:
                        report_lines.append(f"  • {rec}")
                report_lines.append("")
            
            report_lines.append("=" * 80)
            
            # Join all lines
            report = "\n".join(report_lines)
            
            # Save to file if requested
            if output_file:
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(report)
                    self.logger.info(f"Report saved to: {output_file}")
                except Exception as e:
                    self.logger.warning(f"Could not save report to file: {e}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"
    
    def save_analysis_json(self, analysis: Dict[str, Any], output_file: str) -> bool:
        """
        Save analysis results to JSON file.
        
        Args:
            analysis: Analysis results dictionary
            output_file: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Analysis saved to JSON: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving analysis to JSON: {e}")
            return False


# Convenience function for quick analysis
def analyze_audio_file(file_path: str, sample_rate: int = 44100) -> Dict[str, Any]:
    """
    Quick function to analyze an audio file.
    
    Args:
        file_path: Path to audio file
        sample_rate: Sample rate for analysis
        
    Returns:
        Analysis results dictionary
    """
    try:
        import librosa
        audio_data, sr = librosa.load(file_path, sr=sample_rate)
        
        analyzer = AudioQualityAnalyzer(sample_rate=sr)
        analysis = analyzer.analyze_audio(audio_data, label=Path(file_path).stem)
        
        return analysis
    except Exception as e:
        return {'error': f"Failed to analyze audio file: {e}"}


if __name__ == "__main__":
    # Example usage
    print("Audio Quality Analyzer")
    print("This module provides comprehensive audio quality analysis capabilities.")
    print("Use the AudioQualityAnalyzer class to analyze audio data.")
