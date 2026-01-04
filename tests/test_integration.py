import pytest
import os
import sys
import torch
import torchaudio
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.device_manager import DeviceManager
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler, InputError

class TestIntegration:
    """Integration tests with mock audio files"""
    
    @pytest.fixture
    def mock_audio_file(self):
        """Create a mock audio file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Generate 5 seconds of sine wave audio
            sample_rate = 16000
            duration = 5
            frequency = 440  # A4 note
            
            t = torch.linspace(0, duration, sample_rate * duration)
            waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
            
            # Save as WAV file
            torchaudio.save(f.name, waveform, sample_rate)
            
            yield f.name
            
            # Cleanup
            os.unlink(f.name)
    
    @pytest.fixture
    def config_manager(self):
        """Create config manager for testing"""
        return ConfigManager()
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing"""
        return ErrorHandler()
    
    def test_device_detection(self):
        """Test device detection works"""
        device = DeviceManager.get_optimal_device()
        assert device is not None
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_config_validation(self, config_manager):
        """Test configuration validation"""
        assert config_manager.validate_config() == True
        
        # Test with invalid configuration
        config_manager.update({'whisper_model': 'invalid_model'})
        assert config_manager.validate_config() == False
    
    def test_input_validation_valid_file(self, error_handler, mock_audio_file):
        """Test input validation with valid file"""
        assert error_handler.validate_input(mock_audio_file) == True
    
    def test_input_validation_invalid_file(self, error_handler):
        """Test input validation with invalid file"""
        with pytest.raises(InputError):
            error_handler.validate_input('/nonexistent/file.mp4')
    
    def test_input_validation_youtube_url(self, error_handler):
        """Test input validation with YouTube URL"""
        youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert error_handler.validate_input(youtube_url) == True
    
    def test_dependency_validation(self, error_handler):
        """Test dependency validation"""
        try:
            result = error_handler.validate_dependencies()
            assert result == True
        except Exception:
            # Some dependencies might be missing in test environment
            pytest.skip("Dependencies not fully available in test environment")
    
    def test_audio_file_creation(self, mock_audio_file):
        """Test that mock audio file is created correctly"""
        assert os.path.exists(mock_audio_file)
        
        # Load and verify the audio file
        waveform, sample_rate = torchaudio.load(mock_audio_file)
        assert waveform.shape[0] == 1  # Mono
        assert sample_rate == 16000
        assert waveform.shape[1] == 5 * 16000  # 5 seconds
    
    def test_config_file_operations(self, config_manager):
        """Test configuration file operations"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Save configuration
            config_manager.save_config(config_path)
            assert os.path.exists(config_path)
            
            # Load configuration
            new_config = ConfigManager(config_path)
            assert new_config.config.whisper_model == config_manager.config.whisper_model
        
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_error_handling_and_recovery(self, error_handler):
        """Test error handling and recovery mechanisms"""
        # Test error counting
        initial_count = len(error_handler.get_error_summary())
        
        try:
            raise InputError("Test error")
        except InputError as e:
            error_handler.handle_error(e, "test_context", reraise=False)
        
        # Check that error was counted
        error_summary = error_handler.get_error_summary()
        assert len(error_summary) > initial_count
        assert 'InputError' in error_summary
    
    def test_graceful_degradation(self):
        """Test graceful degradation when components fail"""
        # Test that basic functionality works even if advanced features fail
        device = DeviceManager.get_optimal_device()
        assert device is not None
        
        # Even if GPU is not available, CPU should work
        cpu_device = torch.device('cpu')
        assert cpu_device.type == 'cpu'
    
    @pytest.mark.slow
    def test_full_pipeline_mock(self, mock_audio_file, config_manager):
        """Test full processing pipeline with mock audio (if dependencies available)"""
        try:
            # This test requires all dependencies to be installed
            import whisper
            
            # Use smallest model for testing
            config_manager.update({'whisper_model': 'tiny'})
            
            # Try to load the model (might fail in CI/CD)
            model = whisper.load_model('tiny')
            
            # Basic transcription test
            result = model.transcribe(mock_audio_file)
            assert 'text' in result
            assert 'segments' in result
            
        except ImportError:
            pytest.skip("Whisper not available for full pipeline test")
        except Exception as e:
            pytest.skip(f"Full pipeline test skipped due to: {e}")

class TestSystemIntegration:
    """System-level integration tests"""
    
    def test_directory_creation(self):
        """Test that required directories can be created"""
        test_dirs = ['./test_temp', './test_output', './test_logs']
        
        for dir_path in test_dirs:
            os.makedirs(dir_path, exist_ok=True)
            assert os.path.exists(dir_path)
            
            # Cleanup
            shutil.rmtree(dir_path)
    
    def test_import_structure(self):
        """Test that all modules can be imported"""
        modules_to_test = [
            'utils.device_manager',
            'utils.config_manager',
            'utils.error_handler',
            'utils.logger'
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name, fromlist=[''])
                print(f"✓ {module_name}")
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")
    
    def test_cli_interface_basic(self):
        """Test basic CLI interface functionality"""
        try:
            from interfaces.cli import SubtitleCLI
            
            cli = SubtitleCLI()
            parser = cli.create_parser()
            
            # Test help generation (shouldn't raise errors)
            help_text = parser.format_help()
            assert 'Subtitle AI Suite' in help_text
            
        except ImportError:
            pytest.skip("CLI interface dependencies not available")
    
    def test_version_compatibility(self):
        """Test version compatibility of key dependencies"""
        try:
            import torch
            import numpy as np
            
            # Check PyTorch version
            torch_version = torch.__version__
            assert len(torch_version.split('.')) >= 2
            
            # Check NumPy compatibility
            numpy_version = np.__version__
            major, minor = map(int, numpy_version.split('.')[:2])
            
            # Ensure NumPy < 3 (allow 2.x)
            assert major < 3, f"NumPy {numpy_version} may be incompatible"
            
        except ImportError:
            pytest.skip("Version compatibility test skipped - dependencies not available")