import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import shutil
import torch
import torchaudio
import numpy as np

# TEMPORARY: Adjusting sys.path for direct execution during development
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.insert(0, str(project_root))

from src.core.speaker_analyzer import ModernSpeakerDiarization, SpeakerProfile

@pytest.fixture
def speaker_diarization_instance():
    with patch('pyannote.audio.Pipeline.from_pretrained') as mock_pipeline_from_pretrained:
        mock_pipeline = MagicMock()
        mock_pipeline_from_pretrained.return_value = mock_pipeline
        diarizer = ModernSpeakerDiarization()
        diarizer.pipeline = mock_pipeline # Ensure the instance uses our mock
        yield diarizer

@pytest.fixture
def dummy_audio_file(tmp_path):
    file_path = tmp_path / "dummy_audio.wav"
    # Create a dummy silent WAV file
    sf.write(file_path, np.zeros(16000 * 5, dtype=np.float32), 16000)
    return str(file_path)

def test_init_loads_pipeline(speaker_diarization_instance):
    assert speaker_diarization_instance.pipeline is not None
    speaker_diarization_instance.pipeline.from_pretrained.assert_called_once()

@patch.dict(os.environ, {'HF_TOKEN': 'test_token'}, clear=True)
@patch('huggingface_hub.login')
def test_setup_huggingface_auth_with_token(mock_login):
    diarizer = ModernSpeakerDiarization()
    diarizer._setup_huggingface_auth() # Call directly as it runs in init
    mock_login.assert_called_once_with(token='test_token', add_to_git_credential=False)

@patch.dict(os.environ, {}, clear=True) # Ensure no HF_TOKEN
@patch('huggingface_hub.login')
def test_setup_huggingface_auth_no_token(mock_login):
    diarizer = ModernSpeakerDiarization()
    diarizer._setup_huggingface_auth() # Call directly as it runs in init
    mock_login.assert_not_called()

def test_convert_diarization_results(speaker_diarization_instance):
    mock_annotation = MagicMock()
    # Simulate pyannote Annotation object
    mock_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=2.0), None, 'SPEAKER_00'),
        (MagicMock(start=2.5, end=4.0), None, 'SPEAKER_01'),
        (MagicMock(start=4.5, end=6.0), None, 'SPEAKER_00'),
    ]

    speaker_profiles = speaker_diarization_instance._convert_diarization_results(mock_annotation)

    assert len(speaker_profiles) == 2
    speaker_profiles.sort(key=lambda x: x.id) # Sort for consistent assertion
    
    assert speaker_profiles[0].id == "speaker_SPEAKER_00"
    assert speaker_profiles[0].total_speech_time == pytest.approx(3.5)
    assert speaker_profiles[0].segment_times == [(0.0, 2.0), (4.5, 6.0)]

    assert speaker_profiles[1].id == "speaker_SPEAKER_01"
    assert speaker_profiles[1].total_speech_time == pytest.approx(1.5)
    assert speaker_profiles[1].segment_times == [(2.5, 4.0)]

@patch('torchaudio.load', return_value=(torch.tensor([0.0, 0.0, 0.0]), 16000))
@patch('pyannote.audio.Pipeline.__call__') # Mock the pipeline call directly
def test_process_audio_success(mock_pipeline_call, mock_torchaudio_load, speaker_diarization_instance, dummy_audio_file):
    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (MagicMock(start=0.0, end=2.0), None, 'SPEAKER_00'),
        (MagicMock(start=2.5, end=4.0), None, 'SPEAKER_01'),
    ]
    mock_pipeline_call.return_value = mock_annotation

    with patch.object(speaker_diarization_instance, '_convert_diarization_results', wraps=speaker_diarization_instance._convert_diarization_results) as mock_convert:
        with patch.object(speaker_diarization_instance, '_detect_genders') as mock_detect_genders:
            profiles = speaker_diarization_instance.process_audio(dummy_audio_file)
            
            assert len(profiles) == 2
            mock_pipeline_call.assert_called_once()
            mock_convert.assert_called_once_with(mock_annotation)
            mock_detect_genders.assert_called_once_with(dummy_audio_file, profiles)

@patch('torchaudio.load', side_effect=Exception("Audio load error"))
def test_process_audio_pipeline_failure_fallback(mock_torchaudio_load, speaker_diarization_instance, dummy_audio_file):
    # Simulate pipeline failure (e.g., model loading error, or processing error)
    speaker_diarization_instance.pipeline = None # Force fallback path

    with patch.object(speaker_diarization_instance, '_fallback_speaker_detection', wraps=speaker_diarization_instance._fallback_speaker_detection) as mock_fallback:
        profiles = speaker_diarization_instance.process_audio(dummy_audio_file)
        assert len(profiles) == 2 # Expect fallback to return 2 speakers
        mock_fallback.assert_called_once_with(dummy_audio_file)

@patch('librosa.load', return_value=(np.zeros(16000 * 5, dtype=np.float32), 16000))
@patch('librosa.pyin', return_value=(np.array([100.0, 110.0]), np.array([True, True]), None)) # Male F0
def test_detect_genders_male(mock_pyin, mock_librosa_load, speaker_diarization_instance, dummy_audio_file):
    profiles = [
        SpeakerProfile(id="speaker_0", total_speech_time=10.0, segment_times=[(0.0, 5.0)]),
    ]
    speaker_diarization_instance._detect_genders(dummy_audio_file, profiles)
    assert profiles[0].gender == 'male'

@patch('librosa.load', return_value=(np.zeros(16000 * 5, dtype=np.float32), 16000))
@patch('librosa.pyin', return_value=(np.array([200.0, 210.0]), np.array([True, True]), None)) # Female F0
def test_detect_genders_female(mock_pyin, mock_librosa_load, speaker_diarization_instance, dummy_audio_file):
    profiles = [
        SpeakerProfile(id="speaker_0", total_speech_time=10.0, segment_times=[(0.0, 5.0)]),
    ]
    speaker_diarization_instance._detect_genders(dummy_audio_file, profiles)
    assert profiles[0].gender == 'female'
