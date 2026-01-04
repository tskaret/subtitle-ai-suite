import os
import logging
import whisper
from typing import Dict, Any, Optional

class TranscriptionProcessor:
    """
    Handles speech-to-text transcription using the OpenAI Whisper model.
    """

    def __init__(self, model_name: str = "large-v2", device: Optional[str] = None):
        """
        Initializes the TranscriptionProcessor with a Whisper model.

        Args:
            model_name (str): The name of the Whisper model to use (e.g., "base", "small", "medium", "large-v2").
            device (Optional[str]): The device to load the model on (e.g., "cpu", "cuda"). If None, Whisper will
                                    attempt to use GPU if available.
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = device
        self.model = None

        self._load_model()

    def _load_model(self):
        """Loads the Whisper model."""
        try:
            self.logger.info(f"Loading Whisper model '{self.model_name}' on device: {self.device if self.device else 'auto'}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.logger.info("Whisper model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model '{self.model_name}': {e}")
            raise

    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribes an audio file and returns the transcription with word-level timestamps.

        Args:
            audio_path (str): The path to the audio file to transcribe.
            language (Optional[str]): The language of the audio (e.g., "en", "es"). If None,
                                      Whisper will attempt to detect the language.

        Returns:
            Dict[str, Any]: A dictionary containing the transcription segments,
                            detected language, and other Whisper outputs.
                            Each segment will include word-level timestamps.
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded. Call _load_model() first.")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.logger.info(f"Transcribing audio file: {audio_path}")
        options = {
            "word_timestamps": True,
            "language": language,
            "condition_on_previous_text": False, # Helps with hallucination
            "fp16": False # Use float32 for CPU for better compatibility
        }

        try:
            result = self.model.transcribe(audio_path, **options)
            self.logger.info(f"Transcription complete. Detected language: {result.get('language', 'N/A')}")
            return result
        except Exception as e:
            self.logger.error(f"Error during transcription of {audio_path}: {e}")
            raise

