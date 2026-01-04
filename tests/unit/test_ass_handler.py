import pytest
from pathlib import Path
from datetime import timedelta
from src.formats.ass_handler import AssHandler

@pytest.fixture
def ass_handler():
    return AssHandler()

@pytest.fixture
def sample_transcription_data():
    return [
        {'start': 0.0, 'end': 1.5, 'text': 'Hello world.', 'speaker': 'speaker_0'},
        {'start': 2.123, 'end': 3.456, 'text': 'This is a test.', 'speaker': 'speaker_1'},
        {'start': 10.0, 'end': 12.0, 'text': 'Line with\Nnewline.', 'speaker': 'speaker_0'},
        {'start': 60.0, 'end': 61.0, 'text': 'One minute mark.', 'speaker': 'speaker_1'},
        {'start': 3600.0, 'end': 3601.0, 'text': 'One hour mark.'}, # No speaker
    ]

def test_format_timestamp(ass_handler):
    assert ass_handler._format_timestamp(0.0) == "0:00:00.00"
    assert ass_handler._format_timestamp(1.234) == "0:00:01.23"
    assert ass_handler._format_timestamp(60.0) == "0:01:00.00"
    assert ass_handler._format_timestamp(3600.0) == "1:00:00.00"
    assert ass_handler._format_timestamp(3661.12345) == "1:01:01.12"

def test_generate_ass_no_colors(ass_handler, sample_transcription_data, tmp_path):
    output_path = tmp_path / "test.ass"
    ass_handler.generate_ass(sample_transcription_data, output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "Style: Default" in content
    assert "[Events]" in content
    assert "0:00:00.00,0:00:01.50,Default,,10,10,10,,Hello world." in content
    assert "0:00:02.12,0:00:03.45,Default,,10,10,10,,This is a test." in content
    assert "0:01:00.00,0:01:01.00,Default,,10,10,10,,One minute mark." in content
    assert "0:00:10.00,0:00:12.00,Default,,10,10,10,,Line with\Nnewline." in content
    assert "0:00:00.00,0:00:01.50,Default,,10,10,10,,[speaker_0] Hello world." not in content # Ensure no speaker added to text if no colors

def test_generate_ass_with_colors(ass_handler, sample_transcription_data, tmp_path):
    output_path = tmp_path / "test_colors.ass"
    speaker_colors = {
        'speaker_0': '&H0000FF&', # Blue
        'speaker_1': '&H00FF00&'  # Green
    }
    ass_handler.generate_ass(sample_transcription_data, output_path, speaker_colors=speaker_colors)

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Style: Speaker_speaker_0" in content
    assert "PrimaryColour:&H0000FF&" in content # Check for speaker_0 color
    assert "Style: Speaker_speaker_1" in content
    assert "PrimaryColour:&H00FF00&" in content # Check for speaker_1 color

    assert "0:00:00.00,0:00:01.50,Speaker_speaker_0,,10,10,10,,Hello world." in content
    assert "0:00:02.12,0:00:03.45,Speaker_speaker_1,,10,10,10,,This is a test." in content
    assert "0:00:10.00,0:00:12.00,Speaker_speaker_0,,10,10,10,,Line with\Nnewline." in content
    # Ensure segments without assigned speakers (or no color for speaker) use default style
    assert "0:01:00.00,0:01:01.00,Speaker_speaker_1,,10,10,10,,One minute mark." in content
    assert "1:00:00.00,1:00:01.00,Default,,10,10,10,,One hour mark." in content

def test_generate_ass_empty_data(ass_handler, tmp_path):
    output_path = tmp_path / "empty.ass"
    ass_handler.generate_ass([], output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "[Events]" in content
    assert "Dialogue:" not in content # No dialogue entries
