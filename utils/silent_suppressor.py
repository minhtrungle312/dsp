#!/usr/bin/env python3
"""
Silent Warning Suppression - Minimal and Silent
================================================

This provides the same warning suppression as before, but completely silent.
No print statements, no output - just pure suppression.
"""

import os
import warnings
import logging


def silent_suppress_all():
    """
    Silent warning suppression - no output, just suppression.
    
    This replicates the original behavior where warnings were
    suppressed without any console output.
    """
    
    # TensorFlow environment variables (must be set before import)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Python warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    
    # All warning categories
    for category in [DeprecationWarning, FutureWarning, UserWarning, 
                    RuntimeWarning, ImportWarning]:
        warnings.filterwarnings('ignore', category=category)
    
    # Logging
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('spleeter').setLevel(logging.ERROR)
    
    # TensorFlow specific (if available)
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        if hasattr(tf.compat.v1, 'logging'):
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except ImportError:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    silent_suppress_all()
    print("Silent suppression applied - no warnings should appear")
