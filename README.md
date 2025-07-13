# Fancam Voice Enhancement System

🎤 **AI-Powered Vocal Enhancement for Fancam Videos**

Transform your fancam videos with crystal-clear vocals using advanced AI and DSP technology.

## ✨ Features

- **AI-Powered Source Separation**: Uses Spleeter to isolate vocals from background music
- **Advanced DSP Processing**: Applies sophisticated digital signal processing to enhance vocal clarity
- **Video Processing**: Input MP4 video → Output MP4 video with enhanced vocals
- **Intelligent Audio Extraction**: Automatically extracts and processes audio from video files
- **Professional Quality**: Combines AI separation with traditional DSP for optimal results

## 🎯 How It Works

1. **Audio Extraction**: Extracts audio track from input video
2. **AI Separation**: Uses Spleeter AI to separate vocals from accompaniment
3. **Vocal Enhancement**: Applies advanced DSP processing to improve vocal clarity:
   - Noise reduction
   - Spectral enhancement
   - Dynamic range optimization
   - Vocal frequency boosting
4. **Audio Mixing**: Intelligently mixes enhanced vocals with original accompaniment
5. **Video Integration**: Merges enhanced audio back with original video

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **FFmpeg**: Must be installed and available in PATH
- **OS**: Windows, macOS, or Linux

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `tensorflow` - For Spleeter AI models
- `spleeter` - AI-powered audio source separation
- `librosa` - Audio analysis and processing
- `soundfile` - Audio I/O operations
- `numpy`, `scipy` - Scientific computing

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fancam_noise_reduction
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg**
   - **Windows**: Download from https://ffmpeg.org/ and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian)

4. **Verify installation**
   ```bash
   python main.py --help
   ```

## 💻 Usage

### Basic Usage
```bash
python main.py -i input_video.mp4 -o enhanced_video.mp4
```

### Command Line Options
```bash
python main.py --help

options:
  -i, --input     INPUT     Input video file path (required)
  -o, --output    OUTPUT    Output video file path (required)
  --log-level     LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Example
```bash
# Enhance vocals in a fancam video
python main.py -i "concert_fancam.mp4" -o "enhanced_fancam.mp4"

# With debug logging
python main.py -i "input.mp4" -o "output.mp4" --log-level DEBUG
```

## 🎵 Supported Formats

### Input Video Formats
- `.mp4` (recommended)
- `.avi`
- `.mov` 
- `.mkv`
- `.webm`
- `.flv`
- `.wmv`
- `.m4v`

### Output Format
- `.mp4` with enhanced audio (AAC encoding)

## 🔧 Configuration

The system uses optimized default settings for fancam vocal enhancement:

- **Sample Rate**: 22kHz (optimized for vocal processing)
- **AI Model**: Spleeter 2-stems (vocals + accompaniment)
- **DSP Processing**: Vocal-specific frequency enhancement
- **Output Quality**: High-quality AAC audio encoding

## 📊 Processing Pipeline

```
Input Video (MP4)
    ↓
Audio Extraction (FFmpeg)
    ↓
AI Source Separation (Spleeter)
    ↓ 
Vocal Enhancement (Advanced DSP)
    ↓
Audio Mixing (Enhanced vocals + accompaniment)
    ↓
Video Integration (FFmpeg)
    ↓
Output Video (MP4) with Enhanced Vocals
```

## 🎯 Optimization Tips

1. **Input Quality**: Use highest quality source videos for best results
2. **Audio Content**: Works best with videos containing clear vocal content
3. **Processing Time**: Expect 2-5x real-time processing depending on hardware
4. **Hardware**: GPU acceleration (if available) will speed up AI processing

## 🐛 Troubleshooting

### Common Issues

**FFmpeg not found**
```bash
# Verify FFmpeg installation
ffmpeg -version
```

**Spleeter model download**
- First run will download AI models (~100MB)
- Ensure stable internet connection

**Audio extraction fails**
- Check input video format
- Verify FFmpeg installation
- Try with different input file

**Memory issues**
- Close other applications
- Use shorter video clips for testing
- Monitor system memory usage

### Debug Mode
```bash
python main.py -i input.mp4 -o output.mp4 --log-level DEBUG
```

## 📁 Project Structure

```
fancam_noise_reduction/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── config/
│   └── dsp_config.py      # DSP configuration settings
├── core/                  # Core DSP processing modules
├── noise_reduction/
│   └── fancam_processor.py # Main audio processing logic
├── processors/            # Specialized audio processors
├── utils/                 # Utility modules
└── pretrained_models/     # AI model storage (auto-created)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Spleeter** by Deezer for AI-powered source separation
- **Librosa** team for audio analysis tools
- **FFmpeg** project for multimedia processing

---

🎵 **Enhance your fancam videos with professional-grade vocal clarity!** 🎵
