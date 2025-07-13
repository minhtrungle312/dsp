#!/usr/bin/env python3
"""
Comprehensive Warning Suppression for Fancam Voice Enhancement System
=====================================================================

This module provides comprehensive warning suppression for:
- Python warnings
- TensorFlow warnings (Python level)
- TensorFlow warnings (C++ level)
- Spleeter warnings
- Deprecation warnings

Usage:
    import sys
    sys.path.insert(0, '.')
    from utils.warning_suppressor import suppress_all_warnings
    suppress_all_warnings()
"""

import os
import sys
import warnings
import logging


def suppress_tensorflow_warnings():
    """Suppress TensorFlow specific warnings."""
    
    # Environment variables must be set BEFORE importing tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization warnings
    
    # Suppress TensorFlow Python logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('tensorflow.python').setLevel(logging.ERROR)
    
    # Try to suppress TensorFlow internal warnings
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        # Suppress TensorFlow v1 compatibility warnings
        if hasattr(tf.compat.v1, 'logging'):
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            
        # Disable specific TensorFlow warnings
        tf.compat.v1.disable_eager_execution()
        
    except ImportError:
        # TensorFlow not installed or not available
        pass
    except Exception:
        # Any other TensorFlow related errors
        pass


def suppress_spleeter_warnings():
    """Suppress Spleeter specific warnings."""
    
    # Spleeter often imports TensorFlow, so ensure TF warnings are suppressed
    suppress_tensorflow_warnings()
    
    # Suppress Spleeter model loading warnings
    logging.getLogger('spleeter').setLevel(logging.ERROR)
    
    # Suppress specific deprecation warnings from Spleeter
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='spleeter')
    warnings.filterwarnings('ignore', category=FutureWarning, module='spleeter')


def suppress_numpy_warnings():
    """Suppress NumPy warnings."""
    
    # Suppress NumPy warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
    
    # Suppress specific NumPy warnings
    import numpy as np
    np.seterr(all='ignore')


def suppress_librosa_warnings():
    """Suppress Librosa audio processing warnings."""
    
    warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
    warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')


def suppress_matplotlib_warnings():
    """Suppress Matplotlib warnings (if used)."""
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    except ImportError:
        pass


def suppress_all_warnings(verbose=False):
    """
    Comprehensive warning suppression for the entire system.
    
    Args:
        verbose: If True, print suppression status messages
    
    Call this function at the very beginning of your application
    to suppress all types of warnings.
    """
    
    if verbose:
        print("🔇 Suppressing all warnings...")
    
    # General Python warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    
    # Specific warning categories
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=ImportWarning)
    
    # Library specific warnings
    suppress_tensorflow_warnings()
    suppress_spleeter_warnings()
    suppress_numpy_warnings()
    suppress_librosa_warnings()
    suppress_matplotlib_warnings()
    
    # Standard library logging
    logging.getLogger().setLevel(logging.ERROR)
    
    if verbose:
        print("✓ All warnings suppressed")


def enable_warnings():
    """Re-enable warnings (for debugging purposes)."""
    
    warnings.resetwarnings()
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    
    print("✓ Warnings re-enabled")


class SilentContext:
    """
    Context manager to temporarily suppress all output.
    
    Usage:
        with SilentContext():
            # Code that produces unwanted output
            pass
    """
    
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.old_stdout = None
        self.old_stderr = None
    
    def __enter__(self):
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        
        if self.suppress_stderr:
            self.old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_stdout:
            sys.stdout.close()
            sys.stdout = self.old_stdout
        
        if self.old_stderr:
            sys.stderr.close()
            sys.stderr = self.old_stderr


if __name__ == "__main__":
    # Test the warning suppression
    print("Testing warning suppression...")
    
    suppress_all_warnings()
    
    # Test that warnings are suppressed
    import warnings
    warnings.warn("This warning should be suppressed")
    
    print("✓ Warning suppression test completed")
