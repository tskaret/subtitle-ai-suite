import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import sys
import torch

# TEMPORARY: Adjusting sys.path for direct execution during development
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.insert(0, str(project_root))

from src.core.translator import Translator

@pytest.fixture
def mock_marian_model():
    with patch('transformers.MarianMTModel.from_pretrained') as mock_model_from_pretrained:
        mock_model_instance = MagicMock()
        mock_model_from_pretrained.return_value = mock_model_instance
        yield mock_model_instance

@pytest.fixture
def mock_marian_tokenizer():
    with patch('transformers.MarianTokenizer.from_pretrained') as mock_tokenizer_from_pretrained:
        mock_tokenizer_instance = MagicMock()
        # Mock the tokenizer's __call__ method for return_tensors
        mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        # Mock the batch_decode method
        mock_tokenizer_instance.batch_decode.return_value = ["Translated text."]
        yield mock_tokenizer_instance

@pytest.fixture
def translator_instance(mock_marian_model, mock_marian_tokenizer):
    with patch('torch.cuda.is_available', return_value=False): # Force CPU for consistent testing
        translator = Translator(device="cpu")
        yield translator

def test_init_sets_device(translator_instance):
    assert translator_instance.device == "cpu"

@patch('torch.cuda.is_available', return_value=True)
def test_init_cuda_device(mock_cuda_available):
    translator = Translator()
    assert translator.device == "cuda"

def test_load_model_success(translator_instance, mock_marian_model, mock_marian_tokenizer):
    translator_instance._load_model("en", "fr")
    mock_marian_tokenizer.assert_called_once_with("Helsinki-NLP/opus-mt-en-fr")
    mock_marian_model.assert_called_once_with("Helsinki-NLP/opus-mt-en-fr")
    assert translator_instance.current_model_name == "Helsinki-NLP/opus-mt-en-fr"

def test_load_model_failure(translator_instance):
    with patch('transformers.MarianMTModel.from_pretrained', side_effect=Exception("Model not found")):
        with patch('transformers.MarianTokenizer.from_pretrained'):
            with pytest.raises(Exception, match="Failed to load translation model"):
                translator_instance._load_model("en", "de")
            assert translator_instance.model is None
            assert translator_instance.tokenizer is None

def test_load_model_caches_model(translator_instance, mock_marian_model, mock_marian_tokenizer):
    translator_instance._load_model("en", "fr")
    mock_marian_model.assert_called_once() # Called once for initial load
    mock_marian_model.reset_mock()
    mock_marian_tokenizer.reset_mock()

    translator_instance._load_model("en", "fr") # Call again with same language pair
    mock_marian_model.assert_not_called() # Should not be called again
    mock_marian_tokenizer.assert_not_called()

def test_translate_text_success(translator_instance, mock_marian_model, mock_marian_tokenizer):
    translator_instance._load_model("en", "fr")
    text = "Hello world."
    expected_translation = "Translated text."
    mock_marian_tokenizer.batch_decode.return_value = [expected_translation]

    translated_text = translator_instance.translate_text(text, "en", "fr")
    
    mock_marian_tokenizer.assert_called_with(text, return_tensors="pt", padding=True, truncation=True)
    mock_marian_model.return_value.generate.assert_called_once()
    mock_marian_tokenizer.batch_decode.assert_called_once()
    assert translated_text == expected_translation

def test_translate_text_empty(translator_instance):
    translated_text = translator_instance.translate_text("", "en", "fr")
    assert translated_text == ""

def test_translate_segments_success(translator_instance, mock_marian_model, mock_marian_tokenizer):
    translator_instance._load_model("en", "fr")
    segments = [
        {'start': 0.0, 'end': 1.0, 'text': 'Segment 1.'},
        {'start': 1.5, 'end': 2.5, 'text': 'Segment 2.'}
    ]
    mock_marian_tokenizer.batch_decode.return_value = ["Translated Segment 1.", "Translated Segment 2."]

    translated_segments = translator_instance.translate_segments(segments, "en", "fr")

    assert len(translated_segments) == 2
    assert translated_segments[0]['text'] == "Translated Segment 1."
    assert translated_segments[1]['text'] == "Translated Segment 2."
    mock_marian_model.return_value.generate.assert_called() # Called for each batch
    mock_marian_tokenizer.batch_decode.assert_called()

def test_translate_segments_empty(translator_instance):
    translated_segments = translator_instance.translate_segments([], "en", "fr")
    assert translated_segments == []
