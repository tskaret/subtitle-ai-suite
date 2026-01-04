# Subtitle AI Suite

Professional AI-powered subtitle generation and processing toolkit with advanced speaker diarization, emotion detection, and multilingual support.

## Features

- 🎯 **Advanced Speaker Diarization**: Identify and separate multiple speakers in audio.
- 🎨 **Speaker Colorization**: Assign distinct colors to primary speakers in generated subtitles.
- 🎙️ **High-Quality Transcription**: Integrates Whisper large-v2 for accurate speech-to-text with word-level timestamps.
- 🌐 **Multilingual Support & Translation**: Transcribe in multiple languages and translate subtitles using Hugging Face models.
- 🗣️ **Emotion Detection**: Optionally detect emotions within audio segments.
- 📦 **Batch Processing**: Efficiently process multiple video/audio files or entire YouTube playlists.
- 🔄 **Resume Processing**: Checkpoint system for interrupted processing (planned, partially implemented).
- 🚀 **GPU Acceleration**: Automatic GPU detection and optimization for faster processing.
- 📺 **Multiple Output Formats**: Export to SRT, ASS (Advanced SubStation Alpha), JSON, and potentially other professional video editing formats.
- 📱 **Multiple Input Sources**: Supports YouTube URLs, local video/audio files, and YouTube playlists.

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

# Setup environment (creates necessary directories)
# python scripts/setup_environment.py # This script is for future use; directories are created on demand.

# Download AI models (e.g., Whisper, Pyannote models)
# Some models (like Pyannote) might require a HuggingFace token set as an environment variable (HF_TOKEN)
# See SETUP_HF_TOKEN.md for more details.
# python scripts/download_models.py # This script is for future use; models download on first use.
```

### 2. Basic Usage (via CLI)

The main command-line interface is located at `src/interfaces/cli/main_cli.py`.

```bash
# Process a local video file (generates SRT and ASS by default)
python src/interfaces/cli/main_cli.py path/to/your/video.mp4

# Process with speaker colorization (requires pyannote models to be downloaded on first run)
python src/interfaces/cli/main_cli.py path/to/your/video.mp4 --colorize

# Process YouTube video
python src/interfaces/cli/main_cli.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify output directory and output formats
python src/interfaces/cli/main_cli.py video.mp4 --output-dir ./my_subtitles/ --format srt json

# Process an entire directory of media files
python src/interfaces/cli/main_cli.py --batch ./path/to/media_folder/ --output-dir ./batch_output/

# Process a YouTube playlist
python src/interfaces/cli/main_cli.py --playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID" --output-dir ./playlist_output/
```

### 3. Advanced Options

```bash
# Specify input language (e.g., Spanish)
python src/interfaces/cli/main_cli.py video.mp4 --language es

# Use a different Whisper model size
python src/interfaces/cli/main_cli.py video.mp4 --whisper-model medium

# Enable translation from detected/specified language to French
python src/interfaces/cli/main_cli.py video.mp4 --language en --translate fr

# Enable emotion detection (requires SpeechBrain emotion model download on first run)
python src/interfaces/cli/main_cli.py video.mp4 --enable-emotion-detection

# Adjust speaker colorization threshold (e.g., top 80% of speech time get colors)
python src/interfaces/cli/main_cli.py video.mp4 --colorize --speaker-threshold 0.80

# Keep temporary files for debugging
python src/interfaces/cli/main_cli.py video.mp4 --keep-temp

# Show system information and detected devices
python src/interfaces/cli/main_cli.py --info
```

## Configuration

Create a `.env` file in the project root to configure API keys and settings. A sample `.env.example` is provided.

```env
# API Keys (required for some models/features)
# Hugging Face token is crucial for pyannote.audio and some other models
HF_TOKEN=your_huggingface_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
XAI_API_KEY=your_xai_api_key_here # For Grok integration if supported

# Processing Settings
DEFAULT_WHISPER_MODEL=large-v2
TEMP_DIR=./temp
OUTPUT_DIR=./output
MAX_CONCURRENT_JOBS=2 # For batch processing

# Feature Flags
ENABLE_COLLABORATION=false # Set to true to enable collaboration features (if implemented)
ENABLE_ANALYTICS=false     # Set to true to enable analytics (if implemented)
```

## Output Files

The suite generates various output files based on your configuration and requested formats:

- `*.srt` - Standard SubRip subtitle format.
- `*.ass` - Advanced SubStation Alpha format with speaker colorization (if enabled).
- `*.json` - Full transcription segments (including speaker labels, timestamps, and potential emotion data).
- `speakers_report.json` - Detailed speaker analysis (planned).
- `emotion_report.json` - Comprehensive emotion analysis (planned).

## System Requirements

- Python 3.8+
- FFmpeg (must be installed and accessible in system's PATH, for audio/video processing)
- CUDA-compatible GPU (optional, for acceleration with PyTorch, highly recommended for performance)
- 8GB+ RAM (16GB+ recommended for `large` Whisper models and complex diarization)
- Sufficient disk space for models (several GBs) and temporary files.

## Testing

The project utilizes `pytest` for unit, integration, and end-to-end tests.

```bash
# Run all tests
pytest tests/

# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run end-to-end tests (may download models and take longer)
pytest tests/e2e/

# Run with coverage report
pytest tests/ --cov=src/
```

## Architecture

```
subtitle-ai-suite/
├── config/                   # Global configuration files
│   ├── settings.py           # Core application settings
│   └── ...                   # Other config files (language, AI models, etc.)
├── src/
│   ├── subtitle_suite/       # Main application package
│   │   ├── main.py           # Main entry point (can also be cli or gui)
│   │   ├── core/             # Core processing modules (input, audio, transcription, diarization, sync, translation)
│   │   ├── ai/               # AI and ML modules (emotion detection, content analysis)
│   │   ├── processing/       # Processing pipeline (batch, pipeline management, queues)
│   │   ├── formats/          # Subtitle format handlers (SRT, ASS, VTT)
│   │   ├── collaboration/    # Team features (cloud sync, real-time editing)
│   │   ├── interfaces/       # User interfaces (CLI, GUI, API)
│   │   │   ├── cli/          # Command-line interface modules
│   │   │   └── gui/          # Graphical user interface modules
│   │   │   └── api/          # RESTful API modules
│   │   └── analytics/        # Analytics & reporting
│   │   └── utils/            # Utility functions (file management, logging, error handling, security)
│   └── ...                   # Other top-level src modules
├── data/                     # AI models, templates, language data, presets
│   ├── models/               # Pre-trained AI models (Whisper, SpeechBrain)
│   ├── templates/            # Export templates (ASS styles)
│   └── ...                   # Other data (languages, presets)
├── projects/                 # User project workspaces and checkpoints
├── temp/                     # System temporary files
├── output/                   # Final output directories (subtitles, videos, reports)
├── tests/                    # Test suite (unit, integration, e2e)
│   ├── unit/                 # Unit tests for individual components
│   ├── integration/          # Integration tests for combined components
│   └── e2e/                  # End-to-End tests for full application flows
├── scripts/                  # Utility scripts (setup, model download, migration)
├── deployment/               # Deployment configurations (Docker, Kubernetes, Terraform)
├── .github/                  # GitHub workflows (CI/CD, security)
├── .env.example              # Example environment variables
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup script
├── pyproject.toml            # Project build configuration
└── ...                       # Other project files (.gitignore, LICENSE, etc.)
```

## Contributing

We welcome contributions! Please see our `CONTRIBUTING.md` (planned) for guidelines.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Support

- GitHub Issues: Report bugs and request features (https://github.com/yourusername/subtitle-ai-suite/issues)
- Documentation: See the `docs/` directory for detailed guides.

---
**Note**: This `README.md` reflects the current and planned state of the project. Features marked "planned" or "partially implemented" will be developed in future iterations.
