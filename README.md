# Subtitle AI Suite

Professional AI-powered subtitle generation and processing toolkit with advanced speaker diarization and colorization.

## Features

- 🎯 **Advanced Speaker Diarization**: Identify and separate multiple speakers
- 🎨 **Speaker Colorization**: Assign distinct colors to primary speakers  
- 🎙️ **High-Quality Transcription**: Whisper large-v2 integration with word-level timestamps
- 📺 **Multiple Formats**: Export to SRT, ASS, VTT, and professional video editing formats
- 🔄 **Resume Processing**: Checkpoint system for interrupted processing
- 🚀 **GPU Acceleration**: Automatic GPU detection and optimization
- 📱 **Multiple Input Sources**: YouTube URLs, local files, playlists

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/subtitle-ai-suite.git
cd subtitle-ai-suite

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment and download models
python scripts/setup_environment.py
python scripts/download_models.py
```

### 2. Basic Usage

```bash
# Process a local video file
python src/main.py path/to/your/video.mp4

# Process with speaker colorization
python src/main.py path/to/your/video.mp4 --enable-colorization

# Process YouTube video
python src/main.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify output directory
python src/main.py video.mp4 --output-dir ./my_subtitles/
```

### 3. Advanced Options

```bash
# Specify language
python src/main.py video.mp4 --language en

# Use different Whisper model
python src/main.py video.mp4 --whisper-model medium

# Adjust speaker threshold
python src/main.py video.mp4 --enable-colorization --color-threshold 0.90
```

## Configuration

Create a `.env` file to configure API keys and settings:

```env
# Optional API Keys
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Processing Settings
DEFAULT_WHISPER_MODEL=large-v2
TEMP_DIR=./temp
OUTPUT_DIR=./output
MAX_CONCURRENT_JOBS=2

# Feature Flags
ENABLE_GPU=true
ENABLE_SPEAKER_DIARIZATION=true
```

## Output Files

The suite generates several output files:

- `subtitles.srt` - Standard subtitle format
- `subtitles.ass` - Advanced SubStation Alpha with speaker colors
- `speakers.json` - Speaker analysis and metadata
- `processing_metadata.json` - Complete processing information

## System Requirements

- Python 3.8+
- FFmpeg (for audio/video processing)
- CUDA-compatible GPU (optional, for acceleration)
- 4GB+ RAM (8GB+ recommended)

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/

# Run specific test
pytest tests/test_main.py
```

## Architecture

```
subtitle-ai-suite/
├── src/
│   ├── core/                 # Core processing modules
│   ├── utils/                # Utility functions
│   └── main.py              # Main application entry
├── tests/                   # Test suite
├── scripts/                 # Setup and utility scripts
├── data/                    # AI models and templates
└── docs/                    # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: Report bugs and request features
- Documentation: See `docs/` directory
- Examples: See `examples/` directory