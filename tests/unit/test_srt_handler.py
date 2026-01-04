import pytest
from pathlib import Path
from datetime import timedelta
from src.formats.srt_handler import SrtHandler

@pytest.fixture
def srt_handler():
    return SrtHandler()

@pytest.fixture
def sample_transcription_data():
    return [
        {'start': 0.0, 'end': 1.5, 'text': 'Hello world.'},
        {'start': 2.123, 'end': 3.456, 'text': 'This is a test.'},
        {'start': 10.0, 'end': 12.0, 'text': 'Line with\nnewline.'},
        {'start': 60.0, 'end': 61.0, 'text': 'One minute mark.'},
        {'start': 3600.0, 'end': 3601.0, 'text': 'One hour mark.'},
    ]

def test_format_timestamp(srt_handler):
    assert srt_handler._format_timestamp(0.0) == "00:00:00,000"
    assert srt_handler._format_timestamp(1.234) == "00:00:01,234"
    assert srt_handler._format_timestamp(60.0) == "00:01:00,000"
    assert srt_handler._format_timestamp(3600.0) == "01:00:00,000"
    assert srt_handler._format_timestamp(3661.12345) == "01:01:01,123"

def test_generate_srt(srt_handler, sample_transcription_data, tmp_path):
    output_path = tmp_path / "test.srt"
    srt_handler.generate_srt(sample_transcription_data, output_path)

    expected_content = """1
00:00:00,000 --> 00:00:01,500
Hello world.

2
00:00:02,123 --> 00:00:03,456
This is a test.

3
00:00:10,000 --> 00:00:12,000
Line with\nnewline.

4
00:01:00,000 --> 00:01:01,000
One minute mark.

5
01:00:00,000 --> 01:00:01,000
One hour mark.

"""
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert content == expected_content

def test_generate_srt_empty_data(srt_handler, tmp_path):
    output_path = tmp_path / "empty.srt"
    srt_handler.generate_srt([], output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == ""

def test_generate_srt_output_dir_creation(srt_handler, sample_transcription_data, tmp_path):
    nested_dir = tmp_path / "nested" / "output"
    output_path = nested_dir / "test.srt"
    srt_handler.generate_srt(sample_transcription_data, output_path)
    assert output_path.exists()
    assert output_path.parent.exists()
