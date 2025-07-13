"""
Core DSP Processing Module - Module xử lý DSP cốt lõi
================================================

Module này chứa các lớp xử lý DSP cốt lõi:
- SpectralProcessor: Xử lý phổ tín hiệu
- HarmonicEnhancer: Cải thiện âm hài  
- NoiseGateProcessor: Xử lý cổng nhiễu
- DSPProcessor: Bộ xử lý DSP chính

Author: DSP Team
Date: 2025
"""

# Import core classes when they're available
try:
    from .spectral_processing import SpectralProcessor
    from .harmonic_enhancement import HarmonicEnhancer
    from .noise_gate import NoiseGateProcessor
    from .dsp_processor import AdvancedDSPProcessor
    
    __all__ = [
        'SpectralProcessor',
        'HarmonicEnhancer', 
        'NoiseGateProcessor',
        'AdvancedDSPProcessor'
    ]
except ImportError:
    # Handle case where modules are not yet created
    __all__ = []

# Phiên bản module
__version__ = "1.0.0"
