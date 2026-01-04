"""
Advanced Audio Processor module
Handles high-quality audio extraction, noise reduction, and normalization.
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from moviepy.editor import VideoFileClip
from pydub import AudioSegment  # Though pydub is imported, it's not directly used for core processing here. Consider removing if not used.
from typing import Optional, Tuple, Dict, Any

class AudioProcessor:
    """
    Advanced audio processing pipeline.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize audio processor.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.target_sr = self.config.get('sample_rate', 48000)
        self.enable_noise_reduction = self.config.get('enable_noise_reduction', True)
        self.enable_normalization = self.config.get('enable_normalization', True)

    def _is_video_file(self, file_path: str) -> bool:
        """Checks if the file has a common video extension."""
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv']
        return any(file_path.lower().endswith(ext) for ext in video_extensions)

    def extract_audio_from_video(self, video_path: str, output_audio_path: str) -> str:
        """
        Extracts high-quality audio from a video file using moviepy.
        
        Args:
            video_path (str): Path to the input video file.
            output_audio_path (str): Path to save the extracted audio file.
            
        Returns:
            str: Path to the extracted audio file.
        """
        self.logger.info(f"Extracting audio from video: {video_path} to {output_audio_path}")
        try:
            with VideoFileClip(video_path) as video_clip:
                audio_clip = video_clip.audio
                # Ensure 48kHz and mono
                audio_clip.write_audiofile(
                    output_audio_path, 
                    fps=self.target_sr, 
                    nbytes=2,  # 16-bit audio
                    nchannels=1 # mono
                )
            self.logger.info(f"Audio extracted successfully to: {output_audio_path}")
            return output_audio_path
        except Exception as e:
            self.logger.error(f"Error extracting audio from video {video_path}: {e}")
            raise

    def process(self, input_path: str, output_path: str = None) -> str:
        """
        Process audio/video file: extract audio if video, then enhance (noise reduction, normalize), and save.
        
        Args:
            input_path (str): Path to input audio/video file.
            output_path (str, optional): Path to save processed audio. 
                                         If None, creates a temp file.
        
        Returns:
            str: Path to processed audio file.
        """
        # Determine intermediate audio path if input is a video
        intermediate_audio_path = input_path
        if self._is_video_file(input_path):
            intermediate_audio_path = os.path.join(self.output_dir, f"{Path(input_path).stem}_extracted.wav")
            intermediate_audio_path = self.extract_audio_from_video(input_path, intermediate_audio_path)

        if output_path is None:
            base, _ = os.path.splitext(intermediate_audio_path)
            output_path = f"{base}_processed.wav"

        self.logger.info(f"Processing audio: {intermediate_audio_path}")

        try:
            # 1. Load Audio
            # We use librosa to load. It handles resampling.
            # Note: librosa loads as float32.
            y, sr = librosa.load(intermediate_audio_path, sr=self.target_sr, mono=True)
            self.logger.info(f"Loaded audio with sample rate {sr}")

            # 2. Noise Reduction
            if self.enable_noise_reduction:
                self.logger.info("Applying noise reduction...")
                # nr.reduce_noise performs stationary noise reduction by default
                y = nr.reduce_noise(y=y, sr=sr, stationary=True)

            # 3. Normalization
            if self.enable_normalization:
                self.logger.info("Applying normalization...")
                max_val = np.max(np.abs(y))
                if max_val > 0:
                    y = y / max_val
            
            # 4. Save
            self.logger.info(f"Saving processed audio to: {output_path}")
            sf.write(output_path, y, sr, subtype='PCM_16')
            
            # Clean up intermediate audio file if it was extracted from video
            if intermediate_audio_path != input_path and os.path.exists(intermediate_audio_path):
                os.remove(intermediate_audio_path)
                self.logger.info(f"Cleaned up intermediate audio file: {intermediate_audio_path}")

            return output_path

        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            # Clean up intermediate audio file if an error occurs
            if intermediate_audio_path != input_path and os.path.exists(intermediate_audio_path):
                os.remove(intermediate_audio_path)
            raise

    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about the audio file.
        
        Args:
            file_path (str): Path to audio file.
            
        Returns:
            Dict containing audio metadata.
        """
        try:
            info = sf.info(file_path)
            return {
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'duration': info.duration,
                'format': info.format,
                'subtype': info.subtype
            }
        except Exception as e:
            self.logger.error(f"Error getting audio info: {e}")
            return {}