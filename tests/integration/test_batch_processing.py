import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import shutil
import argparse
import json

# TEMPORARY: Adjusting sys.path for direct execution during development
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.insert(0, str(project_root))

from src.processing.batch_processor import BatchProcessor
from src.processing.pipeline_manager import SubtitlePipelineManager


@pytest.fixture
def dummy_batch_input_dir(tmp_path):
    input_dir = tmp_path / "batch_inputs"
    input_dir.mkdir()
    (input_dir / "video1.mp4").touch()
    (input_dir / "audio2.mp3").touch()
    (input_dir / "sub_dir").mkdir()
    (input_dir / "sub_dir" / "video3.mov").touch()
    return input_dir

@pytest.fixture
def mock_subtitle_pipeline_manager():
    with patch('src.processing.batch_processor.SubtitlePipelineManager') as MockPipelineManager:
        mock_instance = MockPipelineManager.return_value
        # Default success result
        mock_instance.process.return_value = {
            "success": True, 
            "generated_files": ["output/file.srt"], 
            "output_dir": "output_dir"
        }
        yield MockPipelineManager


def test_add_directory_to_batch(mock_subtitle_pipeline_manager, dummy_batch_input_dir, tmp_path):
    output_dir = tmp_path / "batch_output"
    config = {
        'output_dir': str(output_dir),
        'temp_dir': str(tmp_path / "temp"),
        'whisper_model': 'base',
        'device': 'cpu',
        'language': None,
        'colorize': False,
        'speaker_threshold': 0.95,
        'keep_temp': True,
        'prefix': None,
        'format': ['srt', 'ass'],
        'enable_emotion_detection': False
    }
    batch_processor = BatchProcessor(config)
    
    jobs_added = batch_processor.add_directory(str(dummy_batch_input_dir), str(output_dir))
    
    assert jobs_added == 3
    assert len(batch_processor.jobs) == 3
    assert any("video1.mp4" in job.input_path for job in batch_processor.jobs)
    assert any("audio2.mp3" in job.input_path for job in batch_processor.jobs)
    assert any("video3.mov" in job.input_path for job in batch_processor.jobs)

@patch('src.processing.batch_processor.yt_dlp.YoutubeDL')
def test_add_playlist_to_batch(mock_ydl, mock_subtitle_pipeline_manager, tmp_path):
    output_dir = tmp_path / "playlist_output"
    config = {
        'output_dir': str(output_dir),
        'temp_dir': str(tmp_path / "temp"),
        'whisper_model': 'base',
        'device': 'cpu',
        'language': None,
        'colorize': False,
        'speaker_threshold': 0.95,
        'keep_temp': True,
        'prefix': None,
        'format': ['srt', 'ass'],
        'enable_emotion_detection': False
    }
    batch_processor = BatchProcessor(config)

    mock_ydl_instance = MagicMock()
    mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.extract_info.return_value = {
        'entries': [
            {'url': 'https://youtube.com/watch?v=videoA', 'title': 'Video A'},
            {'url': 'https://youtube.com/watch?v=videoB', 'title': 'Video B'},
        ]
    }

    jobs_added = batch_processor.add_playlist("https://youtube.com/playlist?list=test_list", str(output_dir))

    assert jobs_added == 2
    assert len(batch_processor.jobs) == 2
    assert any("videoA" in job.input_path for job in batch_processor.jobs)
    assert any("videoB" in job.input_path for job in batch_processor.jobs)
    mock_ydl_instance.extract_info.assert_called_once()

def test_process_all_success(mock_subtitle_pipeline_manager, dummy_batch_input_dir, tmp_path):
    output_dir = tmp_path / "batch_output"
    config = {
        'output_dir': str(output_dir),
        'temp_dir': str(tmp_path / "temp"),
        'whisper_model': 'base',
        'device': 'cpu',
        'language': None,
        'colorize': False,
        'speaker_threshold': 0.95,
        'keep_temp': True,
        'prefix': None,
        'format': ['srt', 'ass'],
        'enable_emotion_detection': False
    }
    batch_processor = BatchProcessor(config, max_workers=1)
    batch_processor.add_directory(str(dummy_batch_input_dir), str(output_dir))

    report = batch_processor.process_all()

    assert report['completed'] == 3
    assert report['failed'] == 0
    assert report['success_rate'] == 100.0
    assert mock_subtitle_pipeline_manager.call_count == 3 # Pipeline manager called for each job

def test_process_all_partial_failure(mock_subtitle_pipeline_manager, dummy_batch_input_dir, tmp_path):
    # Make one job fail
    mock_subtitle_pipeline_manager.return_value.process.side_effect = [
        {"success": True, "generated_files": ["output/file1.srt"]},
        {"success": False, "error": "Simulated error"},
        {"success": True, "generated_files": ["output/file3.srt"]}
    ]

    output_dir = tmp_path / "batch_output_fail"
    config = {
        'output_dir': str(output_dir),
        'temp_dir': str(tmp_path / "temp"),
        'whisper_model': 'base',
        'device': 'cpu',
        'language': None,
        'colorize': False,
        'speaker_threshold': 0.95,
        'keep_temp': True,
        'prefix': None,
        'format': ['srt', 'ass'],
        'enable_emotion_detection': False
    }
    batch_processor = BatchProcessor(config, max_workers=1)
    batch_processor.add_directory(str(dummy_batch_input_dir), str(output_dir))

    report = batch_processor.process_all()

    assert report['completed'] == 2
    assert report['failed'] == 1
    assert report['success_rate'] == pytest.approx(66.666, 0.001)
    assert mock_subtitle_pipeline_manager.call_count == 3

    failed_job = next(job for job in batch_processor.jobs if job.status == 'failed')
    assert failed_job.error_message == "Simulated error"
