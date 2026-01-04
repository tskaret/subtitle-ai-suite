import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import shutil

# TEMPORARY: Adjusting sys.path for direct execution during development
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2] # Adjust if needed
sys.path.insert(0, str(project_root))

from src.core.input_handler import InputHandler

@pytest.fixture
def input_handler():
    # Use a temporary directory for outputs during tests
    temp_output_dir = Path("test_temp_downloads")
    temp_output_dir.mkdir(exist_ok=True)
    handler = InputHandler(output_dir=str(temp_output_dir))
    yield handler
    # Clean up after test
    shutil.rmtree(temp_output_dir)

@pytest.fixture
def dummy_local_file(tmp_path):
    file_path = tmp_path / "test_video.mp4"
    file_path.touch()
    return file_path

def test_process_local_file_exists(input_handler, dummy_local_file):
    processed_path = input_handler.process_local_file(str(dummy_local_file))
    assert processed_path == dummy_local_file.resolve()

def test_process_local_file_not_exists(input_handler):
    with pytest.raises(FileNotFoundError):
        input_handler.process_local_file("non_existent_file.mp4")

def test_process_local_file_unsupported_type(input_handler, tmp_path):
    unsupported_file = tmp_path / "document.txt"
    unsupported_file.touch()
    with pytest.raises(ValueError, match="Unsupported file type"):
        input_handler.process_local_file(str(unsupported_file))

@pytest.mark.parametrize("url, expected", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", True),
    ("http://youtu.be/dQw4w9WgXcQ", True),
    ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", True),
    ("https://www.youtube.com/playlist?list=PLrnXJ_w7V2mGj2b4L9u2iG1X0S5L0B0J0", True),
    ("not_a_youtube_url.com", False),
    ("https://vimeo.com/12345", False),
])
def test_is_youtube_url(input_handler, url, expected):
    assert input_handler._is_youtube_url(url) == expected

@patch('pytube.YouTube')
def test_download_youtube_video_success(mock_youtube, input_handler):
    mock_stream = MagicMock()
    mock_stream.download.return_value = str(input_handler.output_dir / "downloaded_video.mp4")
    mock_youtube.return_value.streams.filter.return_value.order_by.return_value.desc.return_value.first.return_value = mock_stream
    mock_youtube.return_value.title = "Test Video"

    url = "https://www.youtube.com/watch?v=test"
    downloaded_path = input_handler.download_youtube_video(url)

    assert downloaded_path == input_handler.output_dir / "downloaded_video.mp4"
    mock_youtube.return_value.streams.filter.assert_called_once_with(progressive=True, file_extension='mp4')
    mock_stream.download.assert_called_once_with(output_path=input_handler.output_dir)

@patch('pytube.YouTube')
def test_download_youtube_video_no_stream(mock_youtube, input_handler):
    mock_youtube.return_value.streams.filter.return_value.order_by.return_value.desc.return_value.first.return_value = None
    url = "https://www.youtube.com/watch?v=no_stream"
    with pytest.raises(Exception, match="No suitable stream found for download"):
        input_handler.download_youtube_video(url)

@patch('pytube.Playlist')
@patch('pytube.YouTube')
def test_download_youtube_playlist_success(mock_youtube, mock_playlist, input_handler):
    mock_video1_stream = MagicMock()
    mock_video1_stream.download.return_value = str(input_handler.output_dir / "video1.mp4")
    mock_video1_youtube = MagicMock()
    mock_video1_youtube.streams.filter.return_value.order_by.return_value.desc.return_value.first.return_value = mock_video1_stream
    mock_video1_youtube.title = "Video 1"

    mock_video2_stream = MagicMock()
    mock_video2_stream.download.return_value = str(input_handler.output_dir / "video2.mp4")
    mock_video2_youtube = MagicMock()
    mock_video2_youtube.streams.filter.return_value.order_by.return_value.desc.return_value.first.return_value = mock_video2_stream
    mock_video2_youtube.title = "Video 2"

    # Configure pytube.YouTube to return different mocks for different URLs
    def youtube_side_effect(url):
        if "video1" in url:
            return mock_video1_youtube
        elif "video2" in url:
            return mock_video2_youtube
        return MagicMock() # Fallback for other URLs

    mock_youtube.side_effect = youtube_side_effect

    mock_playlist.return_value.title = "Test Playlist"
    mock_playlist.return_value.video_urls = ["https://www.youtube.com/watch?v=video1", "https://www.youtube.com/watch?v=video2"]

    url = "https://www.youtube.com/playlist?list=test_playlist"
    downloaded_paths = input_handler.download_youtube_playlist(url)

    assert len(downloaded_paths) == 2
    assert input_handler.output_dir / "video1.mp4" in downloaded_paths
    assert input_handler.output_dir / "video2.mp4" in downloaded_paths
    mock_playlist.assert_called_once_with(url)

@patch('pytube.Playlist')
def test_download_youtube_playlist_video_error(mock_playlist, input_handler):
    mock_playlist.return_value.title = "Test Playlist"
    mock_playlist.return_value.video_urls = ["https://www.youtube.com/watch?v=video1_fail", "https://www.youtube.com/watch?v=video2_success"]

    with patch('src.core.input_handler.InputHandler.download_youtube_video') as mock_download_video:
        mock_download_video.side_effect = [
            Exception("Simulated download error"),
            str(input_handler.output_dir / "video2.mp4") # Second video succeeds
        ]
        url = "https://www.youtube.com/playlist?list=test_playlist_partial_fail"
        downloaded_paths = input_handler.download_youtube_playlist(url)

        assert len(downloaded_paths) == 1
        assert input_handler.output_dir / "video2.mp4" in downloaded_paths
        assert "Simulated download error" in capsys.readouterr().out # Check if error was printed (requires capsys fixture)
