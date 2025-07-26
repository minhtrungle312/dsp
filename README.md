# Hệ thống Nâng cao Chất lượng Giọng hát Fancam (Fancam Voice Enhancement System)

**Nâng cao Giọng hát trong Video Fancam bằng AI và DSP**

Dự án này là một hệ thống tự động xử lý video fancam để cải thiện và nâng cao chất lượng giọng hát của nghệ sĩ. Hệ thống sử dụng kết hợp các kỹ thuật Xử lý Tín hiệu số (DSP) tiên tiến và mô hình AI (Spleeter) để tách và làm rõ giọng hát từ môi trường có nhiều tiếng ồn.

## Tính năng chính

- **Tách nguồn âm thanh bằng AI**: Sử dụng model Spleeter (2-stems-16kHz) để tách riêng giọng hát (vocals) và nhạc nền (accompaniment)
- **Xử lý DSP tiên tiến**: Áp dụng pipeline DSP 5 bước gồm:
  - Spectral Subtraction (Trừ phổ) để khử nhiễu cơ bản
  - Wiener Filtering (Lọc Wiener) thích ứng để giảm nhiễu thông minh
  - Harmonic Enhancement (Tăng cường hài âm) dựa trên F0 tracking
  - Advanced Noise Gate để loại bỏ tín hiệu yếu
  - RMS Normalization và Soft Clipping để chống méo
- **Xử lý Video hoàn chỉnh**: Input MP4 → Output MP4 với âm thanh đã được cải thiện
- **Giao diện linh hoạt**: Hỗ trợ cả chế độ dòng lệnh và chế độ tương tác (interactive mode)
- **Debug và Phân tích**: Xuất tất cả các bước trung gian để phân tích và gỡ lỗi
- **Tự động tối ưu**: Tự động điều chỉnh các tham số dựa trên đặc tính của từng file âm thanh

## Cách thức hoạt động

1. **Trích xuất Âm thanh**: Tách luồng âm thanh từ video đầu vào bằng FFmpeg
2. **Tiền xử lý DSP**: Áp dụng pipeline xử lý tín hiệu số 5 bước:
   - **Phân tích phổ**: STFT để chuyển đổi từ miền thời gian sang miền tần số
   - **Trừ phổ nâng cao**: Ước tính và trừ bỏ nhiễu sử dụng HPSS
   - **Lọc Wiener thích ứng**: Giảm nhiễu dựa trên tỷ số tín hiệu/nhiễu
   - **Tăng cường hài âm**: Khuếch đại các tần số hài dựa trên F0 tracking
   - **Noise Gate nâng cao**: Loại bỏ tín hiệu dưới ngưỡng động
3. **Tách nguồn bằng AI**: Sử dụng Spleeter để tách vocals và accompaniment
4. **Tái tạo và Tối ưu**: 
   - Đồng bộ độ dài các track âm thanh
   - Áp dụng noise gate thông minh trên vocals
   - Trộn âm thanh với tỷ lệ tối ưu (vocals 90%, accompaniment 10%)
   - Chuẩn hóa RMS và chống clipping
5. **Tích hợp Video**: Ghép âm thanh đã cải thiện vào video gốc

## Yêu cầu hệ thống

### Yêu cầu hệ thống
- **Python**: 3.8 trở lên
- **FFmpeg**: Bắt buộc phải cài đặt và có trong PATH của hệ thống
- **OS**: Windows, macOS, hoặc Linux
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **Dung lượng**: ~2GB cho models và dependencies

### Thư viện Python chính
```bash
pip install -r requirements.txt
```

Các thư viện quan trọng:
- `tensorflow>=2.0` - Cho Spleeter AI models
- `spleeter` - Tách nguồn âm thanh bằng AI
- `librosa>=0.8.0` - Phân tích và xử lý âm thanh
- `soundfile` - Đọc/ghi file âm thanh
- `numpy`, `scipy` - Tính toán khoa học
- `ffmpeg-python` - Interface với FFmpeg

## Cài đặt

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd fancam_noise_reduction
   ```

2. **Tạo môi trường ảo (khuyến nghị)**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux  
   source .venv/bin/activate
   ```

3. **Cài đặt thư viện Python**
   ```bash
   pip install -r requirements.txt
   ```

4. **Cài đặt FFmpeg**
   - **Windows**: Tải từ https://ffmpeg.org/ và thêm vào PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian)

5. **Kiểm tra cài đặt**
   ```bash
   python main.py --help
   ffmpeg -version
   ```

## Cách sử dụng

### 1. Chế độ Tương tác (Interactive Mode) - Khuyến nghị cho người mới

```bash
python main.py
```

Hệ thống sẽ hiển thị menu tương tác, cho phép bạn:
- Chọn file video đầu vào
- Tự động tạo đường dẫn đầu ra trong thư mục `output/`
- Theo dõi tiến trình xử lý

### 2. Chế độ Dòng lệnh (Command-Line Mode)

```bash
python main.py -i input_video.mp4 -o enhanced_video.mp4
```

### Các tùy chọn dòng lệnh
```bash
python main.py --help

options:
  -i, --input     INPUT     Đường dẫn file video đầu vào (bắt buộc)
  -o, --output    OUTPUT    Đường dẫn file video đầu ra (tùy chọn)
  --config        CONFIG    Đường dẫn file cấu hình JSON (tùy chọn)  
  --log-level     LEVEL     Mức độ log (DEBUG, INFO, WARNING, ERROR)
  --interactive             Chạy ở chế độ tương tác
```

### Ví dụ sử dụng
```bash
# Cách đơn giản nhất (output tự động)
python main.py -i "jisoo.mp4"

# Chỉ định output cụ thể
python main.py -i "concert_fancam.mp4" -o "enhanced_fancam.mp4"

# Với debug logging để theo dõi chi tiết
python main.py -i "input.mp4" -o "output.mp4" --log-level DEBUG
```

## Định dạng được hỗ trợ

### Định dạng Video đầu vào
- `.mp4` (khuyến nghị - tối ưu nhất)
- `.avi`
- `.mov` 
- `.mkv`
- `.webm`
- `.flv`
- `.wmv`
- `.m4v`

### Định dạng đầu ra
- `.mp4` với âm thanh AAC 192kbps (chất lượng cao)

## Cấu hình

Hệ thống sử dụng các thiết lập được tối ưu cho fancam vocal enhancement:

- **Sample Rate**: 16kHz (tối ưu cho Spleeter model và giọng người)
- **AI Model**: Spleeter 2-stems-16kHz (vocals + accompaniment)  
- **DSP Parameters**: 
  - FFT size: 2048 (cân bằng độ phân giải tần số/thời gian)
  - Hop length: 512 (75% overlap cho độ mượt cao)
  - Window: Hann (giảm thiểu leakage)
- **Audio Mixing**: Vocals 90%, Accompaniment 10% (tối ưu cho fancam)
- **Output Quality**: AAC 192kbps (cân bằng chất lượng/kích thước file)

Tất cả các tham số có thể được điều chỉnh trong `config/dsp_config.py`.

## Pipeline Xử lý Chi tiết

```
Input Video (MP4)
    ↓
Audio Extraction (FFmpeg → WAV 16kHz mono)
    ↓
DSP Pre-processing (5 bước):
  ├── STFT Analysis (2048 FFT, Hann window)
  ├── HPSS Noise Profiling (tách harmonic/percussive)
  ├── Spectral Subtraction (α=1.0, β=0.15)
  ├── Adaptive Wiener Filtering (PSD-based)
  └── Harmonic Enhancement (F0 tracking + harmonic boost)
    ↓
AI Source Separation (Spleeter 2-stems-16kHz)
  ├── Vocals stem
  └── Accompaniment stem
    ↓
Audio Reconstruction:
  ├── Length Synchronization (padding/trimming)
  ├── Smart Noise Gate (RMS-based dynamic threshold)
  ├── Audio Mixing (vocals×0.9 + accompaniment×0.1)
  ├── RMS Normalization (match original energy)
  └── Soft Clipping Prevention (tanh limiting)
    ↓
Video Integration (FFmpeg: copy video stream + new audio)
    ↓
Output Video (MP4) với Enhanced Vocals
```

### Debug Output Files
Hệ thống tự động xuất các file debug vào `output/dsp_steps/`:
- `00_original_extracted_from_video.wav`
- `02_dsp_spectral_subtracted.wav`  
- `03_dsp_wiener_filtered.wav`
- `04_dsp_harmonic_enhanced.wav`
- `05_dsp_noise_gated.wav`  
- `05_preprocessed_for_ai.wav`
- `02a_ai_vocals_separated.wav`
- `02b_ai_accompaniment_separated.wav`
- `reconstruct_xx_*.wav` (các bước tái tạo)

## Mẹo Tối ưu và Lưu ý

1. **Chất lượng Input**: Sử dụng video có chất lượng cao nhất để có kết quả tốt nhất
2. **Nội dung Âm thanh**: Hiệu quả tốt nhất với video có giọng hát rõ ràng và ít echo
3. **Thời gian Xử lý**: Kỳ vọng thời gian xử lý 2-5x thời lượng video tùy theo hardware
4. **Hardware**: GPU sẽ tăng tốc quá trình AI processing (nếu có)
5. **File Size**: Video đầu ra có thể nhỏ hơn do tái mã hóa audio AAC
6. **Memory Usage**: Giới hạn độ dài video nếu gặp vấn đề về RAM (khuyến nghị <10 phút mỗi lần)

### Các trường hợp sử dụng tối ưu
- Fancam concert với giọng hát rõ ràng
- Video live performance có nhiễu nền vừa phải  
- Recording từ điện thoại với âm thanh trực tiếp
- Video có quá nhiều reverb/echo
- Audio đã bị nén quá mạnh hoặc có artifact

## Khắc phục Sự cố

### Các vấn đề thường gặp

**Lỗi "FFmpeg not found"**
```bash
# Kiểm tra FFmpeg installation
ffmpeg -version
# Nếu không có: tải và cài đặt FFmpeg, thêm vào PATH
```

**Lỗi "Spleeter model download"**
- Lần chạy đầu tiên sẽ tải models (~100MB)
- Đảm bảo kết nối internet ổn định
- Models được lưu trong `pretrained_models/2stems/`

**Lỗi "Audio extraction fails"**
- Kiểm tra định dạng video đầu vào có được hỗ trợ
- Xác nhận FFmpeg hoạt động bình thường
- Thử với file video khác để test

**Vấn đề về Memory**
- Đóng các ứng dụng khác
- Sử dụng video ngắn hơn để test
- Theo dõi RAM usage trong Task Manager
- Xem xét nâng cấp RAM nếu xử lý video dài

**Lỗi TensorFlow/AI model**
- Đảm bảo Python version 3.8+
- Cài đặt lại: `pip install --upgrade tensorflow spleeter`
- Kiểm tra CUDA drivers nếu dùng GPU

### Chế độ Debug
```bash
python main.py -i input.mp4 --log-level DEBUG
```

Chế độ này sẽ:
- Hiển thị chi tiết từng bước xử lý
- Xuất tất cả file debug vào `output/dsp_steps/`
- Ghi chi tiết lỗi vào `fancam_enhancement.log`

## Cấu trúc Dự án Chi tiết

```
fancam_noise_reduction/
├── main.py                     # Entry point chính, điều phối toàn bộ pipeline
├── requirements.txt            # Danh sách dependencies
├── fancam_enhancement.log      # Log file (tự động tạo)
├── test_config.ipynb          # Jupyter notebook để test cấu hình
│
├── config/                     # Module cấu hình DSP
│   ├── __init__.py
│   └── dsp_config.py          # DSPConfiguration class - tất cả tham số DSP
│
├── core/                       # Module xử lý DSP cốt lõi
│   ├── __init__.py
│   ├── dsp_processor.py       # AdvancedDSPProcessor - pipeline DSP chính
│   ├── harmonic_enhancement.py # Tăng cường hài âm, F0 tracking
│   ├── noise_gate.py          # Advanced noise gate processing
│   └── spectral_processing.py # Xử lý phổ tần số
│
├── noise_reduction/            # Module chính điều phối
│   ├── __init__.py
│   └── fancam_processor.py    # FancamVoiceEnhancer - orchestrator chính
│
├── processors/                 # Module tích hợp công cụ bên ngoài
│   ├── spleeter_processor.py  # SpleeterProcessor - interface với Spleeter AI
│   └── __pycache__/
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── audio_utils.py         # AudioUtils - xử lý audio/video I/O
│   ├── silent_suppressor.py   # Tắt warnings/debug messages
│   └── warning_suppressor.py  # TensorFlow warning suppression
│
├── output/                     # Thư mục output (tự động tạo)
│   ├── [tên_video]_enhanced.mp4  # Video đã xử lý
│   └── dsp_steps/             # Các file debug từng bước
│       ├── [timestamp]_00_original_extracted_from_video.wav
│       ├── [timestamp]_02_dsp_spectral_subtracted.wav
│       ├── [timestamp]_03_dsp_wiener_filtered.wav
│       ├── [timestamp]_04_dsp_harmonic_enhanced.wav
│       ├── [timestamp]_05_dsp_noise_gated.wav
│       ├── [timestamp]_05_preprocessed_for_ai.wav
│       ├── [timestamp]_02a_ai_vocals_separated.wav
│       ├── [timestamp]_02b_ai_accompaniment_separated.wav
│       └── [timestamp]_reconstruct_xx_*.wav
│
├── pretrained_models/          # AI models (tự động tải về)
│   └── 2stems/                # Spleeter 2-stems model files
│
└── spleeter_output/           # Thư mục tạm cho Spleeter (tự động tạo)
    ├── vocals.wav
    └── accompaniment.wav
```

### Các module chính và chức năng

1. **main.py**: Entry point, xử lý arguments, orchestrate toàn bộ pipeline
2. **FancamVoiceEnhancer**: Class chính điều phối các bước xử lý
3. **AdvancedDSPProcessor**: Thực hiện 5-step DSP preprocessing pipeline  
4. **SpleeterProcessor**: Interface với Spleeter AI cho source separation
5. **AudioUtils**: Các utility cho audio/video I/O operations

## Đóng góp

Chúng tôi hoan nghênh các đóng góp! Vui lòng tạo pull requests hoặc mở issues cho bugs và feature requests.

### Các lĩnh vực cần cải thiện
- Tối ưu hiệu năng cho video dài
- Thêm hỗ trợ GPU acceleration  
- Cải thiện thuật toán F0 tracking
- Phát triển GUI interface
- Thêm metrics để đánh giá chất lượng tự động

## Ghi nhận

- **Spleeter** by Deezer cho AI-powered source separation
- **Librosa** team cho audio analysis tools
- **FFmpeg** project cho multimedia processing
- **TensorFlow** team cho deep learning framework

