import pytest
import subprocess
from pathlib import Path
import os
import shutil
import json
from unittest.mock import patch, MagicMock
import sys
import numpy as np
import soundfile as sf
import torch # For mocking torch operations

# Adjusting sys.path for direct execution during development
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.insert(0, str(project_root))

# Define the CLI script path
CLI_SCRIPT = project_root / "src" / "interfaces" / "cli" / "main_cli.py"
if not CLI_SCRIPT.exists():
    raise FileNotFoundError(f"CLI script not found at {CLI_SCRIPT}")

@pytest.fixture(scope="module")
def youtube_url_fixture():
    # Use a very short, publicly available YouTube video for E2E testing
    # This URL is for a short, public domain audio clip
    # Disclaimer: Ensure the chosen URL is publicly accessible and appropriate for testing.
    return "https://www.youtube.com/watch?v=sO0iJz72958" # Example: short sound clip, replace if needed

@pytest.fixture
def test_output_dir(tmp_path):
    output_dir = tmp_path / "e2e_output"
    output_dir.mkdir()
    yield output_dir
    # Cleanup is typically handled by tmp_path, but ensure nothing is left behind
    if output_dir.exists():
        shutil.rmtree(output_dir)

@pytest.fixture
def mock_external_models():
    """
    Mocks external model loading and processing to speed up tests
    and avoid actual downloads/heavy computation.
    """
    # Mock Whisper model
    mock_whisper_model = MagicMock()
    mock_whisper_model.transcribe.return_value = {
        'segments': [
            {'start': 0.0, 'end': 1.0, 'text': 'Mocked speech one.', 'words': []},
            {'start': 1.5, 'end': 2.5, 'text': 'Mocked speech two.', 'words': []},
        ],
        'language': 'en'
    }

    # Mock Pyannote audio pipeline (diarization)
    mock_pyannote_pipeline = MagicMock()
    mock_pyannote_pipeline.return_value = MagicMock() # Mock the pipeline instance
    mock_pyannote_pipeline.return_value.__call__.return_value.itertracks.return_value = [
        (MagicMock(start=0.0, end=1.2), None, 'SPEAKER_00'),
        (MagicMock(start=1.3, end=2.7), None, 'SPEAKER_01'),
    ]

    # Mock MarianMT model (translation)
    mock_marian_tokenizer = MagicMock()
    mock_marian_tokenizer.from_pretrained.return_value = MagicMock()
    mock_marian_tokenizer.from_pretrained.return_value.return_value = {"input_ids": torch.tensor([[1,2,3]])} # For __call__
    mock_marian_tokenizer.from_pretrained.return_value.batch_decode.return_value = [
        "Translated mocked speech one.",
        "Translated mocked speech two."
    ]
    mock_marian_model = MagicMock()
    mock_marian_model.from_pretrained.return_value = MagicMock()
    mock_marian_model.from_pretrained.return_value.generate.return_value = torch.tensor([[4,5,6]])

    # Mock SpeechBrain emotion detector
    mock_emotion_classifier = MagicMock()
    mock_emotion_classifier.classify_batch.return_value = (
        torch.tensor([[0.1, 0.9]]), # Example probabilities
        torch.tensor([0.9]),
        torch.tensor([1]),
        ['happy']
    )
    mock_emotion_classifier.hparams.label_encoder.ind2lab = ['neutral', 'happy']


    with patch('whisper.load_model', return_value=mock_whisper_model), \
         patch('pyannote.audio.Pipeline.from_pretrained', return_value=mock_pyannote_pipeline), \
         patch('transformers.MarianTokenizer.from_pretrained', return_value=mock_marian_tokenizer.from_pretrained.return_value), \
         patch('transformers.MarianMTModel.from_pretrained', return_value=mock_marian_model.from_pretrained.return_value), \
         patch('speechbrain.inference.interfaces.foreign_class', return_value=mock_emotion_classifier), \
         patch('torchaudio.load', return_value=(torch.randn(1, 16000), 16000)): # Mock audio loading
        yield


def run_cli_command(command_args: List[str], output_dir: Path):
    """Helper to run the CLI command and return its result."""
    full_command = [sys.executable, str(CLI_SCRIPT)] + command_args
    print(f"\nRunning command: {' '.join(full_command)}")
    result = subprocess.run(
        full_command,
        capture_output=True,
        text=True,
        check=False, # Do not raise CalledProcessError for non-zero exit codes
        env={**os.environ, "HF_TOKEN": "mock_token"} # Pass dummy HF_TOKEN for pyannote auth
    )
    print(f"Stdout:\n{result.stdout}")
    print(f"Stderr:\n{result.stderr}")
    return result

def assert_subtitle_file_exists_and_not_empty(output_path: Path):
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert content.strip() != ""


class TestMainCliE2E:

    def test_basic_local_file_processing(self, test_output_dir, dummy_audio_input, mock_external_models):
        output_base_name = Path(dummy_audio_input).stem
        command_args = [
            dummy_audio_input,
            "--output-dir", str(test_output_dir),
            "--format", "srt", "ass"
        ]
        result = run_cli_command(command_args, test_output_dir)
        assert result.returncode == 0

        assert_subtitle_file_exists_and_not_empty(test_output_dir / f"{output_base_name}.srt")
        assert_subtitle_file_exists_and_not_empty(test_output_dir / f"{output_base_name}.ass")
        assert "Completed:" in result.stdout

    def test_youtube_url_processing_with_colorization_and_json(self, test_output_dir, youtube_url_fixture, mock_external_models):
        output_base_name = Path(youtube_url_fixture).stem # This will be the video ID in actual scenario
        # When mocked, input_handler.download_youtube_video returns Path("mock_downloaded.mp4")
        # So the stem will be "mock_downloaded"
        
        command_args = [
            youtube_url_fixture,
            "--output-dir", str(test_output_dir),
            "--format", "json",
            "--colorize",
            "--keep-temp" # Keep temp files to ensure the pipeline runs fully
        ]
        result = run_cli_command(command_args, test_output_dir)
        assert result.returncode == 0

        # Output file name comes from the mocked input_handler result
        assert_subtitle_file_exists_and_not_empty(test_output_dir / "mock_downloaded.json")
        assert "Completed:" in result.stdout

        # Verify content for json output
        with open(test_output_dir / "mock_downloaded.json", 'r') as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]['text'] == 'Mocked speech one.'
            assert data[0]['speaker'] == 'speaker_0'
            assert data[1]['text'] == 'Mocked speech two.'
            assert data[1]['speaker'] == 'speaker_1'

    def test_processing_with_translation_and_emotion_detection(self, test_output_dir, dummy_audio_input, mock_external_models):
        output_base_name = Path(dummy_audio_input).stem
        command_args = [
            dummy_audio_input,
            "--output-dir", str(test_output_dir),
            "--format", "srt",
            "--language", "en",
            "--translate", "fr",
            "--enable-emotion-detection"
        ]
        result = run_cli_command(command_args, test_output_dir)
        assert result.returncode == 0

        assert_subtitle_file_exists_and_not_empty(test_output_dir / f"{output_base_name}.srt")
        with open(test_output_dir / f"{output_base_name}.srt", "r") as f:
            srt_content = f.read()
            assert "Translated mocked speech one." in srt_content
            assert "Translated mocked speech two." in srt_content
        
        assert "Overall emotion detected: {'primary_emotion': 'happy', 'confidence': 0.9}" in result.stdout # Check for logging of emotion
        assert "Completed:" in result.stdout

    def test_batch_directory_processing(self, test_output_dir, dummy_batch_input_dir, mock_external_models):
        command_args = [
            "--batch", str(dummy_batch_input_dir),
            "--output-dir", str(test_output_dir),
            "--format", "srt",
            "--keep-temp"
        ]
        result = run_cli_command(command_args, test_output_dir)
        assert result.returncode == 0

        assert_subtitle_file_exists_and_not_empty(test_output_dir / "video1.srt")
        assert_subtitle_file_exists_and_not_empty(test_output_dir / "audio2.srt")
        assert_subtitle_file_exists_and_not_empty(test_output_dir / "sub_dir" / "video3.srt")
        assert "Batch processing completed successfully" in result.stdout

    @patch('yt_dlp.YoutubeDL')
    def test_playlist_processing(self, mock_ydl, test_output_dir, mock_external_models):
        # Mock yt_dlp to return a playlist of two dummy videos
        mock_ydl_instance = MagicMock()
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {
            'entries': [
                {'url': 'https://www.youtube.com/watch?v=video_a', 'title': 'Mock Video A'},
                {'url': 'https://www.youtube.com/watch?v=video_b', 'title': 'Mock Video B'},
            ]
        }
        # Mock InputHandler to return specific downloaded paths for each video
        with patch('src.core.input_handler.InputHandler.download_youtube_video') as mock_download_video:
            mock_download_video.side_effect = [
                Path("mock_video_a.mp4"),
                Path("mock_video_b.mp4")
            ]

            playlist_url = "https://www.youtube.com/playlist?list=test_playlist"
            command_args = [
                "--playlist", playlist_url,
                "--output-dir", str(test_output_dir),
                "--format", "srt",
                "--keep-temp"
            ]
            result = run_cli_command(command_args, test_output_dir)
            assert result.returncode == 0

            # The output directories will be created under test_output_dir with the video titles
            assert_subtitle_file_exists_and_not_empty(test_output_dir / "Mock Video A" / "mock_video_a.srt")
            assert_subtitle_file_exists_and_not_empty(test_output_dir / "Mock Video B" / "mock_video_b.srt")
            assert "Batch processing completed successfully" in result.stdout

    def test_info_command(self, test_output_dir):
        command_args = ["--info"]
        result = run_cli_command(command_args, test_output_dir)
        assert result.returncode == 0
        assert "System Information" in result.stdout
        assert "Python Version" in result.stdout
        assert "PyTorch Version" in result.stdout
        assert "Dependency Status" in result.stdout
        assert "System Resources" in result.stdout
