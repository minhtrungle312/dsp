"""
Noise Reduction Module - Module giảm nhiễu
=========================================

Module này chứa lớp chính cho hệ thống giảm nhiễu fancam:
- EnhancedFancamNoiseReduction: Lớp chính điều phối toàn bộ hệ thống

Author: DSP Team
Date: 2025
"""

# Import noise reduction classes when they're available
try:
    from .fancam_processor import EnhancedFancamNoiseReduction
    
    __all__ = [
        'EnhancedFancamNoiseReduction'
    ]
except ImportError:
    # Handle case where modules are not yet created
    __all__ = []

# Phiên bản module
__version__ = "1.0.0"
