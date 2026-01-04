"""
Emotion Detector module using SpeechBrain
"""

import os
import logging
import torch
import torchaudio
from typing import Dict, Any, List, Optional, Tuple

class EmotionDetector:
    """
    Detects emotions from audio segments using SpeechBrain.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Emotion Detector.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = None
        self._load_model()

    def _load_model(self):
        """
        Load the emotion recognition model.
        """
        try:
            from speechbrain.inference.interfaces import foreign_class
            
            # Using a well-known emotion recognition model from SpeechBrain
            # This model detects: neutral, happy, sad, angry
            # Removed pymodule_file="custom_interface.py" as it's not provided and may not be necessary.
            self.classifier = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
                classname="EncoderWav2vec2Classifier", # Using the default classname for this model
                run_opts={"device": self.device}
            )
            self.logger.info("Emotion recognition model loaded successfully.")
            
        except Exception as e:
            self.logger.error(f"Failed to load emotion recognition model: {e}")
            self.logger.warning("Emotion detection will be disabled due to model loading failure.")
            self.classifier = None

    def detect_emotion(self, audio_path: str) -> Dict[str, float]:
        """
        Detect emotion from an audio file.
        
        Args:
            audio_path (str): Path to audio file.
            
        Returns:
            Dict[str, float]: Dictionary mapping emotions to confidence scores.
        """
        if not self.classifier:
            return {}

        try:
            # Load audio
            signal, fs = torchaudio.load(audio_path)
            
            # Resample if needed (model expects 16kHz usually)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                signal = resampler(signal)
            
            # Ensure proper shape (batch, time)
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)
            elif signal.dim() > 2:
                signal = signal.mean(dim=0, keepdim=True)
            
            # Classify
            # SpeechBrain's classify_batch expects signal with batch dim.
            # out_prob: tensor of probabilities for each class
            # score: tensor of scores (usually max prob)
            # index: tensor of index of max score
            # text_lab: list of class labels
            out_prob, score, index, text_lab = self.classifier.classify_batch(signal)
            
            emotions = {}
            if text_lab and len(text_lab) > 0:
                # Assuming text_lab[0] is the predicted emotion label
                # And score[0] is its confidence
                emotions['primary_emotion'] = text_lab[0]
                emotions['confidence'] = float(score[0])
                
                # Optionally, if you want all emotion probabilities
                for i, label in enumerate(self.classifier.hparams.label_encoder.ind2lab):
                    emotions[label] = float(out_prob[0, i])
            
            return emotions

        except Exception as e:
            self.logger.error(f"Error detecting emotion from {audio_path}: {e}")
            return {}