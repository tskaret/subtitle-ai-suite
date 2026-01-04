import os
import re
from pathlib import Path

from pytube import YouTube, Playlist
from yt_dlp import YoutubeDL

class InputHandler:
    def __init__(self, output_dir="temp/downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _is_youtube_url(self, url):
        """Checks if the given URL is a valid YouTube video or playlist URL."""
        youtube_regex = (
            r"(https?://)?(www\.)?"
            "(youtube|youtu|youtube-nocookie)\.(com|be)/"
            "(watch\?v=|embed/|v/|.+\?v=|playlist\?list=|e/|f/|.+\?list=)"
            "([^#&?%\s]+)"
        )
        return re.match(youtube_regex, url) is not None

    def download_youtube_video(self, url):
        """
        Downloads a single YouTube video.
        Returns the path to the downloaded video.
        """
        if not self._is_youtube_url(url):
            raise ValueError("Invalid YouTube URL provided.")
        
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not stream:
                raise Exception("No suitable stream found for download.")
            
            print(f"Downloading: {yt.title}...")
            file_path = stream.download(output_path=self.output_dir)
            print(f"Downloaded: {file_path}")
            return Path(file_path)
        except Exception as e:
            raise Exception(f"Failed to download YouTube video {url}: {e}")

    def download_youtube_playlist(self, url):
        """
        Downloads all videos from a YouTube playlist.
        Returns a list of paths to the downloaded videos.
        """
        if not self._is_youtube_url(url):
            raise ValueError("Invalid YouTube Playlist URL provided.")

        try:
            playlist = Playlist(url)
            print(f"Downloading playlist: {playlist.title}")
            downloaded_paths = []
            for video_url in playlist.video_urls:
                try:
                    video_path = self.download_youtube_video(video_url)
                    downloaded_paths.append(video_path)
                except Exception as e:
                    print(f"Skipping video {video_url} due to error: {e}")
            return downloaded_paths
        except Exception as e:
            raise Exception(f"Failed to download YouTube playlist {url}: {e}")

    def process_local_file(self, file_path):
        """
        Validates a local media file path.
        Returns the absolute path to the file.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Local file not found: {file_path}")
        if not any(path.suffix.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.mov', '.mp3', '.wav', '.flac']):
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return path.resolve()
