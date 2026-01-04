import pytest
import numpy as np
import soundfile as sf
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# TEMPORARY: Adjusting sys.path for direct execution during development
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2] # Adjust if needed
sys.path.insert(0, str(project_root))

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

    @patch('moviepy.editor.VideoFileClip')
    def test_extract_audio_from_video(self, mock_videofileclip, audio_processor, tmp_path):
        mock_audio_clip = MagicMock()
        mock_videofileclip.return_value.__enter__.return_value.audio = mock_audio_clip
        
        video_path = str(tmp_path / "test_video.mp4")
        output_audio_path = str(tmp_path / "extracted_audio.wav")
        
        extracted_path = audio_processor.extract_audio_from_video(video_path, output_audio_path)
        
        mock_videofileclip.assert_called_once_with(video_path)
        mock_audio_clip.write_audiofile.assert_called_once_with(
            output_audio_path,
            fps=audio_processor.target_sr,
            nbytes=2,
            nchannels=1
        )
        assert extracted_path == output_audio_path

    @patch('src.core.audio_processor.AudioProcessor.extract_audio_from_video')
    @patch('librosa.load', return_value=(np.array([0.1, -0.1]), 16000))
    @patch('soundfile.write')
    def test_process_with_video_input(self, mock_sf_write, mock_librosa_load, mock_extract_audio, audio_processor, tmp_path):
        video_input_path = str(tmp_path / "input.mp4")
        extracted_audio_temp_path = str(tmp_path / "audio_processing" / "input_extracted.wav")
        mock_extract_audio.return_value = extracted_audio_temp_path
        
        # Create the directory that audio_processor.process expects for its temporary extracted audio
        (Path(audio_processor.output_dir) / "audio_processing").mkdir(parents=True, exist_ok=True)

        processed_output_path = audio_processor.process(video_input_path)
        
        mock_extract_audio.assert_called_once_with(video_input_path, extracted_audio_temp_path)
        mock_librosa_load.assert_called_once_with(extracted_audio_temp_path, sr=audio_processor.target_sr, mono=True)
        mock_sf_write.assert_called_once()
        assert Path(processed_output_path).name == "input_extracted_processed.wav"