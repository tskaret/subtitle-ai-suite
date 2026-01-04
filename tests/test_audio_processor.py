import pytest
import numpy as np
import soundfile as sf
import os
import shutil
from src.core.audio_processor import AudioProcessor

class TestAudioProcessor:
    
    @pytest.fixture
    def audio_processor(self):
        config = {
            'sample_rate': 16000, # Use lower SR for faster test
            'enable_noise_reduction': True,
            'enable_normalization': True
        }
        return AudioProcessor(config)

    @pytest.fixture
    def noisy_audio_file(self, tmp_path):
        """Create a synthetic noisy audio file"""
        sr = 16000
        duration = 2.0 # seconds
        t = np.linspace(0, duration, int(sr * duration))
        
        # Signal: Sine wave
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Noise: Random noise
        noise = 0.1 * np.random.normal(0, 1, len(t))
        
        # Mixed
        audio = signal + noise
        
        file_path = tmp_path / "noisy_test.wav"
        sf.write(file_path, audio, sr)
        return str(file_path)

    def test_process_creates_file(self, audio_processor, noisy_audio_file, tmp_path):
        output_path = tmp_path / "processed.wav"
        result_path = audio_processor.process(noisy_audio_file, str(output_path))
        
        assert os.path.exists(result_path)
        assert result_path == str(output_path)
        
        # Verify it's a valid audio file
        info = audio_processor.get_audio_info(result_path)
        assert info['sample_rate'] == 16000
        assert info['duration'] >= 2.0

    def test_normalization(self, audio_processor, tmp_path):
        # Create a quiet file
        sr = 16000
        audio = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
        input_path = tmp_path / "quiet.wav"
        sf.write(input_path, audio, sr)
        
        output_path = tmp_path / "normalized.wav"
        audio_processor.process(str(input_path), str(output_path))
        
        # Read back and check max amplitude
        y, _ = sf.read(output_path)
        max_val = np.max(np.abs(y))
        
        # Should be close to 1.0 (normalization target)
        # Note: soundfile writes as PCM_16, so there might be slight quantization error
        assert max_val > 0.9

    def test_noise_reduction_runs(self, audio_processor, noisy_audio_file, tmp_path):
        # Just verifying it runs without error and produces output
        output_path = tmp_path / "denoised.wav"
        audio_processor.process(noisy_audio_file, str(output_path))
        assert os.path.exists(output_path)

