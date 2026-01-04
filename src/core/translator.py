import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from transformers import MarianMTModel, MarianTokenizer
import torch

# Load environment variables
load_dotenv()

class Translator:
    """
    Handles translation of text and subtitle segments using Hugging Face models.
    Supports dynamic loading of models based on source and target languages.
    """

    def __init__(self, device: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Translator.
        
        Args:
            device (Optional[str]): The device to load the model on (e.g., "cpu", "cuda").
            config (Optional[Dict[str, Any]]): Configuration dictionary for translator settings.
        """
        self.logger = logging.getLogger(__name__)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        
        self.tokenizer = None
        self.model = None
        self.current_model_name = None

    def _load_model(self, src_lang: str, tgt_lang: str):
        """
        Loads the MarianMT translation model and tokenizer for the specified language pair.
        Models are typically named 'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'.
        """
        model_tag = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        if self.current_model_name == model_tag and self.model is not None:
            self.logger.debug(f"Model '{model_tag}' already loaded.")
            return

        self.logger.info(f"Loading translation model '{model_tag}' on device: {self.device}")
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_tag)
            self.model = MarianMTModel.from_pretrained(model_tag).to(self.device)
            self.current_model_name = model_tag
            self.logger.info(f"Translation model '{model_tag}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load translation model '{model_tag}': {e}")
            self.tokenizer = None
            self.model = None
            self.current_model_name = None
            raise

    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translates a single piece of text from source to target language.
        """
        if not text:
            return ""

        try:
            self._load_model(src_lang, tgt_lang)
            if not self.model or not self.tokenizer:
                raise RuntimeError(f"Translation model not loaded for {src_lang}-{tgt_lang}.")

            encoded_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated_tokens = self.model.generate(**encoded_text)
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            self.logger.error(f"Error translating text '{text}' from {src_lang} to {tgt_lang}: {e}")
            raise

    def translate_segments(self, segments: List[Dict[str, Any]], src_lang: str, tgt_lang: str) -> List[Dict[str, Any]]:
        """
        Translates a list of transcription segments.
        Each segment dictionary should have a 'text' key.
        """
        if not segments:
            return []

        self.logger.info(f"Translating {len(segments)} segments from {src_lang} to {tgt_lang}.")
        translated_segments = []
        
        try:
            self._load_model(src_lang, tgt_lang)
            if not self.model or not self.tokenizer:
                raise RuntimeError(f"Translation model not loaded for {src_lang}-{tgt_lang}.")

            texts_to_translate = [segment.get('text', '') for segment in segments]
            
            # Batch translation for efficiency
            translated_texts = []
            batch_size = 8 # Can be configured
            for i in range(0, len(texts_to_translate), batch_size):
                batch = texts_to_translate[i:i+batch_size]
                if not batch: continue
                encoded_batch = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                translated_tokens = self.model.generate(**encoded_batch)
                translated_batch = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                translated_texts.extend(translated_batch)

            for i, segment in enumerate(segments):
                translated_segment = segment.copy()
                translated_segment['text'] = translated_texts[i]
                translated_segments.append(translated_segment)
            
            self.logger.info("Segments translated successfully.")
            return translated_segments
        except Exception as e:
            self.logger.error(f"Error translating segments from {src_lang} to {tgt_lang}: {e}")
            raise

