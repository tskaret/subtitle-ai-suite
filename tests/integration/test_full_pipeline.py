import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import shutil
import argparse
import numpy as np
import soundfile as sf

# TEMPORARY: Adjusting sys.path for direct execution during development
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.insert(0, str(project_root))

from src.processing.pipeline_manager import SubtitlePipelineManager
from src.core.speaker_analyzer import SpeakerProfile


@pytest.fixture
def dummy_audio_input(tmp_path):
    # Create a dummy audio file
    audio_path = tmp_path / "test_audio.wav"
    sf.write(audio_path, np.zeros(16000 * 5, dtype=np.float32), 16000) # 5 seconds of silence
    return str(audio_path)

@pytest.fixture
def mock_pipeline_components():
    with patch('src.core.input_handler.InputHandler') as MockInputHandler, \
         patch('src.core.audio_processor.AudioProcessor') as MockAudioProcessor, \
         patch('src.core.transcription.TranscriptionProcessor') as MockTranscriptionProcessor, \
         patch('src.core.speaker_analyzer.ModernSpeakerDiarization') as MockModernSpeakerDiarization, \
         patch('src.core.synchronizer.Synchronizer') as MockSynchronizer, \
         patch('src.core.translator.Translator') as MockTranslator, \
         patch('src.ai.emotion_detector.EmotionDetector') as MockEmotionDetector:
        
        # Configure mocks
        mock_input_handler_instance = MockInputHandler.return_value
        mock_input_handler_instance.process_local_file.return_value = Path("mock_input.wav")
        mock_input_handler_instance.download_youtube_video.return_value = Path("mock_downloaded.mp4")

        mock_audio_processor_instance = MockAudioProcessor.return_value
        mock_audio_processor_instance.process.return_value = "mock_processed_audio.wav"

        mock_transcription_processor_instance = MockTranscriptionProcessor.return_value
        mock_transcription_processor_instance.transcribe_audio.return_value = {
            'segments': [
                {'start': 0.0, 'end': 1.0, 'text': 'Hello speaker one.', 'words': []},
                {'start': 1.5, 'end': 2.5, 'text': 'And speaker two.', 'words': []},
            ],
            'language': 'en'
        }
        # Mock the _load_model method as it's called internally
        mock_transcription_processor_instance._load_model.return_value = None

        mock_speaker_profile_1 = SpeakerProfile(id="speaker_0", segment_times=[(0.0, 1.2)])
        mock_speaker_profile_2 = SpeakerProfile(id="speaker_1", segment_times=[(1.3, 2.7)])
        mock_modern_speaker_diarization_instance = MockModernSpeakerDiarization.return_value
        mock_modern_speaker_diarization_instance.process_audio.return_value = [mock_speaker_profile_1, mock_speaker_profile_2]

        mock_synchronizer_instance = MockSynchronizer.return_value
        mock_synchronizer_instance.synchronize.return_value = [
            {'start': 0.0, 'end': 1.0, 'text': 'Hello speaker one.', 'speaker': 'speaker_0'},
            {'start': 1.5, 'end': 2.5, 'text': 'And speaker two.', 'speaker': 'speaker_1'},
        ]

        mock_translator_instance = MockTranslator.return_value
        mock_translator_instance.translate_segments.return_value = [
            {'start': 0.0, 'end': 1.0, 'text': 'Hola hablante uno.', 'speaker': 'speaker_0'},
            {'start': 1.5, 'end': 2.5, 'text': 'Y hablante dos.', 'speaker': 'speaker_1'},
        ]
        mock_emotion_detector_instance = MockEmotionDetector.return_value
        mock_emotion_detector_instance.detect_emotion.return_value = {"primary_emotion": "neutral", "confidence": 0.8}

        yield {
            'input_handler': mock_input_handler_instance,
            'audio_processor': mock_audio_processor_instance,
            'transcription_processor': mock_transcription_processor_instance,
            'speaker_diarization': mock_modern_speaker_diarization_instance,
            'synchronizer': mock_synchronizer_instance,
            'translator': mock_translator_instance,
            'emotion_detector': mock_emotion_detector_instance
        }


def test_full_pipeline_local_file_srt_ass(tmp_path, dummy_audio_input, mock_pipeline_components):
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"

    args = argparse.Namespace(
        input=dummy_audio_input,
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        whisper_model="base",
        language="en",
        translate=None,
        colorize=False,
        speaker_threshold=0.95,
        device="cpu",
        keep_temp=False,
        prefix=None,
        format=["srt", "ass"],
        enable_emotion_detection=False
    )

    pipeline_manager = SubtitlePipelineManager(args)
    result = pipeline_manager.process(args.input)

    assert result["success"] is True
    assert (output_dir / "test_audio.srt").exists()
    assert (output_dir / "test_audio.ass").exists()

    with open(output_dir / "test_audio.srt", "r") as f:
        srt_content = f.read()
        assert "Hello speaker one." in srt_content
        assert "And speaker two." in srt_content
    
    with open(output_dir / "test_audio.ass", "r") as f:
        ass_content = f.read()
        assert "Hello speaker one." in ass_content
        assert "And speaker two." in ass_content
        assert "Style: Default" in ass_content # Default style as no colorization

    # Verify component calls
    mock_pipeline_components['input_handler'].process_local_file.assert_called_once_with(dummy_audio_input)
    mock_pipeline_components['audio_processor'].process.assert_called_once()
    mock_pipeline_components['transcription_processor'].transcribe_audio.assert_called_once()
    mock_pipeline_components['speaker_diarization'].process_audio.assert_not_called()
    mock_pipeline_components['synchronizer'].synchronize.assert_not_called()
    mock_pipeline_components['translator'].translate_segments.assert_not_called()
    mock_pipeline_components['emotion_detector'].detect_emotion.assert_not_called()


def test_full_pipeline_with_colorization_and_translation(tmp_path, dummy_audio_input, mock_pipeline_components):
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"

    args = argparse.Namespace(
        input=dummy_audio_input,
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        whisper_model="large",
        language="en",
        translate="es",
        colorize=True,
        speaker_threshold=0.95,
        device="cuda",
        keep_temp=True, # Keep temp for inspection if needed
        prefix="translated",
        format=["srt", "ass", "json"],
        enable_emotion_detection=True
    )

    pipeline_manager = SubtitlePipelineManager(args)
    result = pipeline_manager.process(args.input)

    assert result["success"] is True
    assert (output_dir / "translated_test_audio.srt").exists()
    assert (output_dir / "translated_test_audio.ass").exists()
    assert (output_dir / "translated_test_audio.json").exists()

    with open(output_dir / "translated_test_audio.srt", "r") as f:
        srt_content = f.read()
        assert "[speaker_0] Hola hablante uno." in srt_content
        assert "[speaker_1] Y hablante dos." in srt_content

    with open(output_dir / "translated_test_audio.ass", "r") as f:
        ass_content = f.read()
        assert "Style: Default" not in ass_content # Should use speaker styles
        assert "Style: Speaker_speaker_0" in ass_content
        assert "Style: Speaker_speaker_1" in ass_content
        assert "Hola hablante uno." in ass_content
        assert "Y hablante dos." in ass_content

    with open(output_dir / "translated_test_audio.json", "r") as f:
        json_content = json.load(f)
        assert len(json_content) == 2
        assert json_content[0]['text'] == "Hola hablante uno."
        assert json_content[0]['speaker'] == "speaker_0"

    # Verify component calls
    mock_pipeline_components['input_handler'].process_local_file.assert_called_once_with(dummy_audio_input)
    mock_pipeline_components['audio_processor'].process.assert_called_once()
    mock_pipeline_components['speaker_diarization'].process_audio.assert_called_once()
    mock_pipeline_components['transcription_processor'].transcribe_audio.assert_called_once()
    mock_pipeline_components['synchronizer'].synchronize.assert_called_once()
    mock_pipeline_components['translator'].translate_segments.assert_called_once()
    mock_pipeline_components['emotion_detector'].detect_emotion.assert_called_once()


def test_full_pipeline_youtube_input(tmp_path, mock_pipeline_components):
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    youtube_url = "https://www.youtube.com/watch?v=test_video"

    args = argparse.Namespace(
        input=youtube_url,
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        whisper_model="small",
        language="en",
        translate=None,
        colorize=False,
        speaker_threshold=0.95,
        device="cpu",
        keep_temp=False,
        prefix=None,
        format=["srt"],
        enable_emotion_detection=False
    )

    pipeline_manager = SubtitlePipelineManager(args)
    result = pipeline_manager.process(args.input)

    assert result["success"] is True
    assert (output_dir / "mock_downloaded.srt").exists()
    mock_pipeline_components['input_handler'].download_youtube_video.assert_called_once_with(youtube_url)
