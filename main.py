#!/usr/bin/env python3
"""
Fancam Voice Enhancement System - Main Entry Point
=================================================

This system processes MP4 fancam videos to enhance vocal quality using:
- Advanced DSP noise reduction
- Spleeter AI-powered vocal separation
- Video reconstruction with enhanced audio

Input: MP4 video file
Output: MP4 video file with enhanced vocals

Author: DSP Team
Date: 2025
"""

import os
import sys
import argparse
import logging
import subprocess
import warnings
from pathlib import Path

# Early TensorFlow warning suppression (before any TF imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops warnings
warnings.filterwarnings('ignore')

# Import our components
from utils.warning_suppressor import suppress_all_warnings
from config.dsp_config import DSPConfiguration
from noise_reduction.fancam_processor import FancamVoiceEnhancer
import warnings

# Suppress all warnings silently (no print messages)
suppress_all_warnings(verbose=False)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fancam_enhancement.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_input_file(file_path: str) -> bool:
    """
    Validate that the input file exists and is a supported video format.
    
    Args:
        file_path: Path to the input video file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"Error: Input file '{file_path}' does not exist.")
        return False
    
    # Supported video formats for fancam processing
    video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in video_formats:
        print(f"Error: Unsupported video format '{file_ext}'.")
        print(f"Supported formats: {video_formats}")
        return False
    
    return True


def check_ffmpeg_availability() -> bool:
    """Check if FFmpeg is available in the system."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def process_fancam_video(input_path: str, output_path: str, config: DSPConfiguration) -> bool:
    """
    Process a fancam video to enhance vocal quality.
    
    Args:
        input_path: Path to input video file
        output_path: Path for output enhanced video
        config: DSP configuration object
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"FANCAM VOICE ENHANCEMENT PROCESSING")
        print(f"{'='*60}")
        print(f"Input video: {input_path}")
        print(f"Output video: {output_path}")
        print(f"{'='*60}")
        
        # Initialize the voice enhancer
        enhancer = FancamVoiceEnhancer(config)
        
        # Process the video
        result = enhancer.process_video(input_path, output_path)
        
        if result['success']:
            print(f"\n✓ PROCESSING COMPLETED SUCCESSFULLY!")
            print(f"  Enhanced video saved to: {output_path}")
            print(f"  Total processing time: {result['processing_time']:.2f} seconds")
            
            # Show processing details
            if 'details' in result:
                details = result['details']
                print(f"\nProcessing Details:")
                print(f"  - Original video duration: {details.get('duration', 'N/A'):.2f}s")
                print(f"  - Audio extraction time: {details.get('extraction_time', 'N/A'):.2f}s")
                print(f"  - Voice enhancement time: {details.get('enhancement_time', 'N/A'):.2f}s")
                print(f"  - Video reconstruction time: {details.get('reconstruction_time', 'N/A'):.2f}s")
            
            return True
        else:
            print(f"\n✗ PROCESSING FAILED")
            print(f"  Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {str(e)}")
        logging.error(f"Processing error: {str(e)}", exc_info=True)
        return False


def interactive_mode():
    """Run the system in interactive mode."""
    print(f"\n{'='*60}")
    print(f"  FANCAM VOICE ENHANCEMENT SYSTEM")
    print(f"{'='*60}")
    print(f"  Transform your fancam videos with AI-powered vocal enhancement!")
    print(f"  Input: MP4 video → Output: MP4 video with enhanced vocals")
    print(f"{'='*60}")
    
    # Check FFmpeg availability
    if not check_ffmpeg_availability():
        print("\n⚠️  WARNING: FFmpeg not found in system PATH")
        print("   FFmpeg is required for video processing.")
        print("   Please install FFmpeg and add it to your PATH.")
        print("   Download from: https://ffmpeg.org/download.html")
        return
    
    try:
        while True:
            print(f"\nOptions:")
            print(f"1. Process fancam video")
            print(f"2. Configure enhancement settings")
            print(f"3. Show current configuration")
            print(f"4. Exit")
            print(f"{'-'*30}")
            
            choice = input("Select option (1-4): ").strip()
            
            if choice == "1":
                # Process video
                print(f"\nSupported video formats: .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv, .m4v")
                
                input_path = input("Enter input video path: ").strip()
                if not validate_input_file(input_path):
                    continue
                
                # Auto-generate output path in output folder
                input_name = Path(input_path).stem
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                os.makedirs(output_dir, exist_ok=True)  # Create output folder if it doesn't exist
                output_path = os.path.join(output_dir, f"{input_name}_enhanced.mp4")
                
                print(f"Output will be saved to: {output_path}")
                
                # Load configuration
                config = DSPConfiguration()
                
                # Process the video
                success = process_fancam_video(input_path, output_path, config)
                
                if success:
                    print(f"\n🎉 Success! Your enhanced fancam video is ready!")
                    print(f"   Output: {output_path}")
                else:
                    print(f"\n❌ Processing failed. Check the log for details.")
            
            elif choice == "2":
                # Configure settings
                print(f"\n=== Enhancement Configuration ===")
                print(f"Choose enhancement quality:")
                print(f"1. Fast (lower quality, faster processing)")
                print(f"2. Balanced (recommended)")
                print(f"3. High Quality (slower processing)")
                
                quality_choice = input("Enter choice (1-3) [2]: ").strip() or "2"
                
                if quality_choice == "1":
                    config = DSPConfiguration(
                        sample_rate=44100,
                        use_ai_separation=False,
                        noise_reduction_strength=0.7
                    )
                elif quality_choice == "3":
                    config = DSPConfiguration(
                        sample_rate=48000,
                        use_ai_separation=True,
                        noise_reduction_strength=0.9
                    )
                else:
                    config = DSPConfiguration()
                
                print(f"\nConfiguration updated!")
                print(config.get_summary())
            
            elif choice == "3":
                # Show configuration
                config = DSPConfiguration()
                print(f"\nCurrent Configuration:")
                print(config.get_summary())
            
            elif choice == "4":
                print(f"\nGoodbye! 👋")
                break
            
            else:
                print(f"Invalid choice. Please select 1-4.")
            
            input(f"\nPress Enter to continue...")
        
    except KeyboardInterrupt:
        print(f"\n\nExiting...")
        sys.exit(0)


def main():
    """Main entry point of the application."""
    
    # Comprehensive warning suppression for TensorFlow and Python
    warnings.filterwarnings('ignore')
    
    # TensorFlow specific warning suppression
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
    
    # Suppress specific TensorFlow deprecation warnings
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    # Try to suppress TensorFlow warnings at import time
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except ImportError:
        pass  # TensorFlow not directly imported here
    
    parser = argparse.ArgumentParser(
        description="Fancam Voice Enhancement System - AI-powered vocal enhancement for fancam videos"
    )
    parser.add_argument(
        "-i", "--input",
        help="Input video file path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output video file path (optional - defaults to ./output/[filename]_enhanced.mp4)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (JSON)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = DSPConfiguration.from_json(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = DSPConfiguration()
    
    # Command line processing
    if args.input and not args.interactive:
        # Direct command line processing
        if not validate_input_file(args.input):
            sys.exit(1)
        
        if not check_ffmpeg_availability():
            print("Error: FFmpeg not found. Please install FFmpeg and add it to your PATH.")
            sys.exit(1)
        
        # Auto-generate output path if not provided
        if args.output:
            output_path = args.output
        else:
            input_name = Path(args.input).stem
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{input_name}_enhanced.mp4")
            print(f"No output path specified. Using: {output_path}")
        
        success = process_fancam_video(args.input, output_path, config)
        sys.exit(0 if success else 1)
    
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
