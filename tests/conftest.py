import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture(scope='session')
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(scope='session')
def config():
    """Provide test configuration"""
    return {
        'temp_dir': './test_temp',
        'output_dir': './test_output',
        'whisper_model': 'tiny',  # Use smallest model for testing
    }

@pytest.fixture(scope='session')
def sample_audio_path():
    """Create a mock audio file path for testing"""
    # For now, return a placeholder path
    # In a real test, this would create an actual audio file
    return './tests/fixtures/sample_audio.wav'

@pytest.fixture(autouse=True)
def setup_test_directories():
    """Setup and cleanup test directories"""
    test_dirs = ['./test_temp', './test_output', './tests/fixtures']
    
    # Create directories
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup after tests
    import shutil
    for dir_path in test_dirs:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path, ignore_errors=True)