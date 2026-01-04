import os
from pathlib import Path

# Project Configuration
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / 'src'
TEMP_DIR = PROJECT_ROOT / 'temp'
OUTPUT_DIR = PROJECT_ROOT / 'output'

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration Settings
CONFIG = {
    'input': {
        'youtube_url_support': True,
        'local_file_support': True,
        'supported_formats': ['.mp4', '.mkv', '.avi', '.mov']
    },
    'audio': {
        'sample_rate': 16000,
        'channels': 1,
        'quality': 'high'
    },
    'checkpoints': {
        'enabled': True,
        'save_interval': 5  # Save checkpoint every 5 steps
    }
}