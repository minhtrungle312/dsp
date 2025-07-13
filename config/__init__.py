"""
Configuration module for DSP processing
Contains all configuration classes and parameters for the noise reduction system
"""

# Import main configuration class when it's available
try:
    from .dsp_config import DSPConfiguration
    __all__ = ['DSPConfiguration']
except ImportError:
    # Handle case where module is not yet created
    __all__ = []
