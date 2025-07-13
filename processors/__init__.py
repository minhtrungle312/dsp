"""
Processors Module - Module các bộ xử lý
=====================================

Module này chứa các bộ xử lý chuyên biệt:
- SpleeterProcessor: Xử lý tách nguồn âm thanh AI
- EnhancedPostProcessor: Hậu xử lý cải tiến

Author: DSP Team
Date: 2025
"""

# Import processor classes when they're available
try:
    from .spleeter_processor import SpleeterProcessor
    from .post_processor import EnhancedPostProcessor
    
    __all__ = [
        'SpleeterProcessor',
        'EnhancedPostProcessor'
    ]
except ImportError:
    # Handle case where modules are not yet created
    __all__ = []

# Phiên bản module
__version__ = "1.0.0"
