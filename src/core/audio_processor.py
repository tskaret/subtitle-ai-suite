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
from pydub import AudioSegment
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

    def process(self, input_path: str, output_path: str = None) -> str:
        """
        Process audio file: load, enhance (noise reduction, normalize), and save.
        
        Args:
            input_path (str): Path to input audio/video file.
            output_path (str, optional): Path to save processed audio. 
                                         If None, creates a temp file.
        
        Returns:
            str: Path to processed audio file.
        """
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = f"{base}_processed.wav"

        self.logger.info(f"Processing audio: {input_path}")

        try:
            # 1. Load Audio
            # We use librosa to load. It handles resampling.
            # Note: librosa loads as float32.
            y, sr = librosa.load(input_path, sr=self.target_sr, mono=True)
            self.logger.info(f"Loaded audio with sample rate {sr}")

            # 2. Noise Reduction
            if self.enable_noise_reduction:
                self.logger.info("Applying noise reduction...")
                # Assume the first 0.5 seconds is noise if possible, or use stationary noise reduction
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
            
            return output_path

        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
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
