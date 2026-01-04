import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import shutil

# TEMPORARY: Adjusting sys.path for direct execution during development
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.insert(0, str(project_root))

from src.core.transcription import TranscriptionProcessor

@pytest.fixture
def transcription_processor():
    # Patch load_model to avoid actual model download during test setup
    with patch('whisper.load_model') as mock_load_model:
        mock_model = MagicMock()
        # Configure the mock model to have a 'model_name' attribute for checks
        mock_model.model_name = "mock_model"
        mock_load_model.return_value = mock_model
        processor = TranscriptionProcessor(model_name="mock_model", device="cpu")
        yield processor

@pytest.fixture
def dummy_audio_file(tmp_path):
    file_path = tmp_path / "test_audio.wav"
    file_path.touch() # Create an empty file
    return str(file_path)

def test_init_loads_model(transcription_processor):
    # Model is loaded during init, so just check if it's not None
    assert transcription_processor.model is not None
    assert transcription_processor.model_name == "mock_model"

@patch('whisper.load_model', side_effect=Exception("Failed to load"))
def test_init_fails_to_load_model(mock_load_model):
    with pytest.raises(Exception, match="Failed to load Whisper model"):
        TranscriptionProcessor(model_name="fail_model", device="cpu")

@patch('whisper.load_model')
def test_load_model_reloads_if_needed(mock_load_model):
    # Initial load during __init__
    mock_model_initial = MagicMock()
    mock_model_initial.model_name = "initial_model"
    mock_load_model.return_value = mock_model_initial
    
    processor = TranscriptionProcessor(model_name="initial_model", device="cpu")
    mock_load_model.assert_called_once()
    mock_load_model.reset_mock()

    # Change model_name, should trigger reload
    processor.model_name = "new_model"
    mock_model_new = MagicMock()
    mock_model_new.model_name = "new_model"
    mock_load_model.return_value = mock_model_new

    processor._load_model()
    mock_load_model.assert_called_once_with("new_model", device="cpu")
    assert processor.model.model_name == "new_model"


@patch('whisper.load_model')
def test_load_model_does_not_reload_if_same(mock_load_model):
    # Initial load during __init__
    mock_model_initial = MagicMock()
    mock_model_initial.model_name = "same_model"
    mock_load_model.return_value = mock_model_initial
    
    processor = TranscriptionProcessor(model_name="same_model", device="cpu")
    mock_load_model.assert_called_once()
    mock_load_model.reset_mock()

    # Call _load_model with same params, should not reload
    processor._load_model()
    mock_load_model.assert_not_called()


@patch('whisper.load_model')
def test_transcribe_audio_success(mock_load_model, transcription_processor, dummy_audio_file):
    mock_transcribe_result = {
        'segments': [{'start': 0.0, 'end': 2.0, 'text': 'Hello world.'}],
        'language': 'en'
    }
    transcription_processor.model.transcribe.return_value = mock_transcribe_result

    result = transcription_processor.transcribe_audio(dummy_audio_file, language="en")
    
    transcription_processor.model.transcribe.assert_called_once_with(
        dummy_audio_file,
        word_timestamps=True,
        language="en",
        condition_on_previous_text=False,
        fp16=False
    )
    assert result == mock_transcribe_result

def test_transcribe_audio_file_not_found(transcription_processor):
    with pytest.raises(FileNotFoundError):
        transcription_processor.transcribe_audio("non_existent_audio.wav")

@patch('whisper.load_model')
def test_transcribe_audio_language_detection(mock_load_model, transcription_processor, dummy_audio_file):
    mock_transcribe_result = {
        'segments': [{'start': 0.0, 'end': 2.0, 'text': 'Hola mundo.'}],
        'language': 'es'
    }
    transcription_processor.model.transcribe.return_value = mock_transcribe_result

    result = transcription_processor.transcribe_audio(dummy_audio_file) # No language specified
    
    transcription_processor.model.transcribe.assert_called_once_with(
        dummy_audio_file,
        word_timestamps=True,
        language=None, # Should pass None for auto-detection
        condition_on_previous_text=False,
        fp16=False
    )
    assert result['language'] == 'es'

