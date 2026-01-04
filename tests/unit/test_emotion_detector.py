import pytest
from unittest.mock import MagicMock, patch
import torch
from src.ai.emotion_detector import EmotionDetector

class TestEmotionDetector:
    
    @pytest.fixture
    def mock_classifier(self):
        classifier = MagicMock()
        # Mock classify_batch return values
        # out_prob, score, index, text_lab
        classifier.classify_batch.return_value = (
            torch.tensor([[0.9]]), 
            torch.tensor([0.9]), 
            torch.tensor([0]), 
            ['happy']
        )
        return classifier

    @patch('src.ai.emotion_detector.foreign_class')
    def test_initialization(self, mock_foreign_class, mock_classifier):
        mock_foreign_class.return_value = mock_classifier
        
        detector = EmotionDetector()
        assert detector.classifier is not None
        mock_foreign_class.assert_called_once()

    @patch('src.ai.emotion_detector.foreign_class')
    @patch('torchaudio.load')
    def test_detect_emotion(self, mock_load, mock_foreign_class, mock_classifier):
        mock_foreign_class.return_value = mock_classifier
        
        # Mock audio load
        mock_load.return_value = (torch.randn(1, 16000), 16000)
        
        detector = EmotionDetector()
        emotions = detector.detect_emotion("dummy.wav")
        
        assert emotions['primary'] == 'happy'
        assert emotions['confidence'] == 0.9

    @patch('src.ai.emotion_detector.foreign_class')
    def test_initialization_failure(self, mock_foreign_class):
        mock_foreign_class.side_effect = Exception("Model not found")
        
        detector = EmotionDetector()
        assert detector.classifier is None
        
        # Should handle detection gracefully
        emotions = detector.detect_emotion("dummy.wav")
        assert emotions == {}
