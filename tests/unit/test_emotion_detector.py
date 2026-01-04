import pytest
from unittest.mock import MagicMock, patch
import torch
import torchaudio # Import torchaudio for mocking
import numpy as np # Import numpy for mocking

# TEMPORARY: Adjusting sys.path for direct execution during development
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.insert(0, str(project_root))

from src.ai.emotion_detector import EmotionDetector

class TestEmotionDetector:
    
    @pytest.fixture
    def mock_classifier(self):
        classifier = MagicMock()
        # Mock classify_batch return values: out_prob, score, index, text_lab
        classifier.classify_batch.return_value = (
            torch.tensor([[0.1, 0.9, 0.0, 0.0]]), # Example probabilities
            torch.tensor([0.9]), # Score for the primary emotion
            torch.tensor([1]), # Index of the primary emotion (e.g., 'happy')
            ['happy'] # Primary emotion label
        )
        # Mock hparams.label_encoder.ind2lab for full emotion probability dict
        classifier.hparams.label_encoder.ind2lab = ['neutral', 'happy', 'sad', 'angry']
        return classifier

    @patch('src.ai.emotion_detector.torch.cuda.is_available', return_value=False) # Force CPU
    @patch('src.ai.emotion_detector.foreign_class')
    def test_initialization(self, mock_foreign_class, mock_cuda_available, mock_classifier):
        mock_foreign_class.return_value = mock_classifier
        
        detector = EmotionDetector()
        assert detector.classifier is not None
        mock_foreign_class.assert_called_once_with(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            classname="EncoderWav2vec2Classifier",
            run_opts={"device": "cpu"} # Expecting cpu as cuda is mocked to be unavailable
        )
        assert detector.device == "cpu"

    @patch('src.ai.emotion_detector.torch.cuda.is_available', return_value=True) # Force CUDA
    @patch('src.ai.emotion_detector.foreign_class')
    def test_initialization_cuda(self, mock_foreign_class, mock_cuda_available, mock_classifier):
        mock_foreign_class.return_value = mock_classifier
        
        detector = EmotionDetector()
        assert detector.classifier is not None
        mock_foreign_class.assert_called_once_with(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            classname="EncoderWav2vec2Classifier",
            run_opts={"device": "cuda"} # Expecting cuda
        )
        assert detector.device == "cuda"


    @patch('src.ai.emotion_detector.foreign_class')
    @patch('torchaudio.load')
    def test_detect_emotion(self, mock_load, mock_foreign_class, mock_classifier):
        mock_foreign_class.return_value = mock_classifier
        
        # Mock audio load to return a signal and sample rate
        mock_load.return_value = (torch.randn(1, 16000), 16000) # 1 channel, 1 second at 16kHz
        
        detector = EmotionDetector()
        emotions = detector.detect_emotion("dummy.wav")
        
        assert emotions['primary_emotion'] == 'happy'
        assert emotions['confidence'] == 0.9
        assert emotions['neutral'] == 0.1
        assert emotions['happy'] == 0.9
        assert emotions['sad'] == 0.0
        assert emotions['angry'] == 0.0

    @patch('src.ai.emotion_detector.foreign_class')
    @patch('torchaudio.load', return_value=(torch.randn(1, 48000), 48000)) # Test resampling
    @patch('torchaudio.transforms.Resample')
    def test_detect_emotion_resampling(self, mock_resample_class, mock_load, mock_foreign_class, mock_classifier):
        mock_foreign_class.return_value = mock_classifier
        mock_resampler_instance = MagicMock()
        mock_resample_class.return_value = mock_resampler_instance
        mock_resampler_instance.return_value = torch.randn(1, 16000) # Resampled signal

        detector = EmotionDetector()
        detector.detect_emotion("dummy_high_sr.wav")

        mock_load.assert_called_once_with("dummy_high_sr.wav")
        mock_resample_class.assert_called_once_with(48000, 16000)
        mock_resampler_instance.assert_called_once()

    @patch('src.ai.emotion_detector.foreign_class')
    def test_initialization_failure(self, mock_foreign_class):
        mock_foreign_class.side_effect = Exception("Model not found")
        
        detector = EmotionDetector()
        assert detector.classifier is None
        
        # Should handle detection gracefully
        emotions = detector.detect_emotion("dummy.wav")
        assert emotions == {}