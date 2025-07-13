"""
Utilities Module - Module tiện ích
=================================

Module này chứa các lớp tiện ích:
- AudioUtils: Tiện ích xử lý âm thanh
- QualityAnalyzer: Phân tích chất lượng âm thanh

Author: DSP Team
Date: 2025
"""

# Import utility classes when they're available
try:
    from .audio_utils import AudioUtils
    from .quality_analyzer import QualityAnalyzer
    
    __all__ = [
        'AudioUtils',
        'QualityAnalyzer'
    ]
except ImportError:
    # Handle case where modules are not yet created
    __all__ = []

# Phiên bản module
__version__ = "1.0.0"
