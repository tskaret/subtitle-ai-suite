import pytest
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from main import SubtitleAISuite, parse_arguments

class TestSubtitleAISuite:
    """Test suite for main application"""
    
    def test_initialization(self):
        """Test application initialization"""
        config = {
            'temp_dir': './test_temp',
            'output_dir': './test_output'
        }
        
        app = SubtitleAISuite(config)
        assert app.config['temp_dir'] == './test_temp'
        assert app.config['output_dir'] == './test_output'
    
    def test_default_configuration(self):
        """Test default configuration"""
        app = SubtitleAISuite()
        assert 'temp_dir' in app.config
        assert 'output_dir' in app.config
        assert 'whisper_model' in app.config

def test_argument_parsing():
    """Test command line argument parsing"""
    # Mock sys.argv for testing
    import sys
    original_argv = sys.argv
    
    try:
        sys.argv = ['main.py', 'test.wav', '--output-dir', './test_output']
        args = parse_arguments()
        assert args.input == 'test.wav'
        assert args.output_dir == './test_output'
    finally:
        sys.argv = original_argv