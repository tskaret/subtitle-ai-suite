import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import argparse

# TEMPORARY: Adjusting sys.path for direct execution during development
# This will be removed once the project is installable as a package
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
sys.path.insert(0, str(project_root))

from src.core.input_handler import InputHandler
from src.core.audio_processor import AudioProcessor
from src.core.transcription import TranscriptionProcessor
from src.core.speaker_analyzer import ModernSpeakerDiarization, SpeakerProfile
from src.core.synchronizer import Synchronizer
from src.core.translator import Translator # Added Translator import
from src.formats.srt_handler import SrtHandler
from src.formats.ass_handler import AssHandler
from src.subtitle_suite.utils.logger import setup_logging
from src.ai.emotion_detector import EmotionDetector

class SubtitlePipelineManager:
    """
    Orchestrates the entire subtitle generation pipeline for a single media file.
    Encapsulates input handling, audio processing, speaker diarization,
    transcription, synchronization, and subtitle output.
    """

    DEFAULT_COLORS = {
        'red': '&H6B6BFF&',
        'cyan': '&HC4CD4E&',
        'blue': '&HD1B745&',
        'green': '&HB4CE96&',
        'yellow': '&HA7EAFF&',
        'plum': '&HDDA0DD&',
        'orange': '&H129CF3&'
    }

    def __init__(self, args: argparse.Namespace):
        """
        Initializes the pipeline manager with command-line arguments.
        """
        self.args = args
        self.logger = setup_logging()
        
        self.input_handler = InputHandler(output_dir=Path(self.args.temp_dir) / "downloads")
        self.audio_processor = AudioProcessor()
        self.transcription_processor = TranscriptionProcessor(
            model_name=self.args.whisper_model, 
            device=self.args.device if self.args.device != 'auto' else None
        )
        self.speaker_diarization = ModernSpeakerDiarization()
        self.synchronizer = Synchronizer()
        self.translator = Translator(device=self.args.device) # Initialize Translator
        self.srt_handler = SrtHandler()
        self.ass_handler = AssHandler()
        self.emotion_detector = EmotionDetector()

        self.input_path_processed: Optional[Path] = None
        self.processed_audio_path: Optional[Path] = None

    def _analyze_speaker_distribution(self, segments: List[Dict[str, Any]], speaker_profiles: List[SpeakerProfile], threshold: float) -> Dict[str, str]:
        """
        Analyzes speaker distribution and assigns colors based on a threshold.
        Returns a mapping of speaker ID to ASS color code.
        """
        if not speaker_profiles or not segments:
            self.logger.warning("No speaker profiles or segments for colorization.")
            return {}

        speaker_word_counts = {profile.id: 0 for profile in speaker_profiles}
        speaker_speech_times = {profile.id: 0.0 for profile in speaker_profiles}

        total_word_count = 0
        total_speech_time = 0.0

        for segment in segments:
            speaker_id = segment.get('speaker')
            if speaker_id and speaker_id in speaker_word_counts:
                word_count = len(segment.get('text', '').split())
                speech_time = segment.get('end', 0) - segment.get('start', 0)
                
                speaker_word_counts[speaker_id] += word_count
                speaker_speech_times[speaker_id] += speech_time
                total_word_count += word_count
                total_speech_time += speech_time

        if total_word_count == 0:
            self.logger.warning("No words transcribed for speaker distribution analysis.")
            return {}
        
        # Calculate percentage of speech time for each speaker
        speaker_percentages = {
            speaker_id: (time / total_speech_time) if total_speech_time > 0 else 0
            for speaker_id, time in speaker_speech_times.items()
        }

        # Identify primary speakers
        primary_speakers = []
        sorted_speakers = sorted(speaker_percentages.items(), key=lambda item: item[1], reverse=True)

        cumulative_percentage = 0.0
        for speaker_id, percentage in sorted_speakers:
            cumulative_percentage += percentage
            primary_speakers.append(speaker_id)
            if cumulative_percentage >= threshold:
                break
        
        # Assign colors from the predefined palette
        speaker_colors: Dict[str, str] = {}
        color_names = list(self.DEFAULT_COLORS.keys())
        
        for i, speaker_id in enumerate(primary_speakers):
            if i < len(color_names):
                speaker_colors[speaker_id] = self.DEFAULT_COLORS[color_names[i]]
            else:
                self.logger.warning(f"Ran out of unique colors for speaker {speaker_id}. Assigning default white.")
                speaker_colors[speaker_id] = "&H00FFFFFF&" # Fallback to white

        self.logger.info(f"Speaker color assignment: {speaker_colors}")
        return speaker_colors

    def process(self, input_source: str) -> Dict[str, Any]:
        """
        Runs the full subtitle generation pipeline for a given input source.
        
        Args:
            input_source (str): Path to local file or YouTube URL.
            
        Returns:
            Dict[str, Any]: A dictionary containing information about the processed job,
                            including paths to generated subtitle files.
        """
        try:
            # 1. Handle Input (local file or download YouTube)
            if "youtube.com" in input_source or "youtu.be" in input_source:
                self.logger.info(f"Downloading YouTube video: {input_source}")
                self.input_path_processed = self.input_handler.download_youtube_video(input_source)
            else:
                self.logger.info(f"Processing local file: {input_source}")
                self.input_path_processed = self.input_handler.process_local_file(input_source)
            
            if not self.input_path_processed:
                raise ValueError("No valid input provided for processing.")

            # Ensure temp directory for audio processing exists
            temp_audio_dir = Path(self.args.temp_dir) / "audio_processing"
            temp_audio_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. Process Audio
            self.logger.info(f"Extracting and processing audio from: {self.input_path_processed}")
            self.processed_audio_path = self.audio_processor.process(
                str(self.input_path_processed), 
                output_path=str(temp_audio_dir / f"{self.input_path_processed.stem}_processed.wav")
            )

            # 3. Speaker Diarization
            speaker_profiles: List[SpeakerProfile] = []
            if self.args.colorize or (hasattr(self.args, 'translate') and self.args.translate): # Run diarization if colorization or translation is requested
                self.logger.info(f"Performing speaker diarization on: {self.processed_audio_path}")
                speaker_profiles = self.speaker_diarization.process_audio(str(self.processed_audio_path))
                self.logger.info(f"Diarization found {len(speaker_profiles)} speakers.")


            # 4. Transcribe Audio
            self.logger.info(f"Transcribing audio: {self.processed_audio_path}")
            # Ensure model is loaded with correct parameters
            self.transcription_processor._load_model()
                
            transcription_result = self.transcription_processor.transcribe_audio(
                str(self.processed_audio_path), language=self.args.language
            )
            raw_transcription_segments = transcription_result.get('segments', [])

            # 5. Synchronize Transcription with Diarization (if enabled)
            synchronized_segments = raw_transcription_segments
            if speaker_profiles:
                self.logger.info("Synchronizing transcription segments with speaker diarization.")
                synchronized_segments = self.synchronizer.synchronize(raw_transcription_segments, speaker_profiles)
            
            # 6. Emotion Detection (Optional)
            if hasattr(self.args, 'enable_emotion_detection') and self.args.enable_emotion_detection:
                self.logger.info("Performing emotion detection.")
                overall_emotion = self.emotion_detector.detect_emotion(str(self.processed_audio_path))
                self.logger.info(f"Overall emotion detected: {overall_emotion}")
            
            # 7. Translation (Optional)
            final_segments = synchronized_segments
            if hasattr(self.args, 'translate') and self.args.translate:
                if not self.args.language:
                    self.logger.warning("Translation requested but no source language specified. Using detected language from Whisper.")
                    source_language = transcription_result.get('language', 'en') # Fallback to English
                else:
                    source_language = self.args.language

                self.logger.info(f"Translating segments from {source_language} to {self.args.translate}.")
                final_segments = self.translator.translate_segments(
                    synchronized_segments, 
                    src_lang=source_language, 
                    tgt_lang=self.args.translate
                )

            # 8. Generate Subtitles
            base_output_name = self.input_path_processed.stem
            if self.args.prefix:
                base_output_name = f"{self.args.prefix}_{base_output_name}"

            final_output_dir = Path(self.args.output_dir)
            final_output_dir.mkdir(parents=True, exist_ok=True)

            speaker_colors_map = {}
            if self.args.colorize and speaker_profiles:
                self.logger.info("Analyzing speaker distribution for colorization.")
                speaker_colors_map = self._analyze_speaker_distribution(final_segments, speaker_profiles, self.args.speaker_threshold)
            elif self.args.colorize and not speaker_profiles:
                self.logger.warning("Colorization requested but no speaker profiles available.")

            generated_files = []
            for fmt in self.args.format:
                output_subtitle_path = final_output_dir / f"{base_output_name}.{fmt}"
                if fmt == 'srt':
                    self.logger.info(f"Generating SRT: {output_subtitle_path}")
                    srt_data = []
                    for segment in final_segments:
                        srt_data.append({
                            'start': segment['start'],
                            'end': segment['end'],
                            'text': f"[{segment['speaker']}] {segment['text']}" if 'speaker' in segment else segment['text']
                        })
                    self.srt_handler.generate_srt(srt_data, output_subtitle_path)
                    generated_files.append(str(output_subtitle_path))
                elif fmt == 'ass':
                    self.logger.info(f"Generating ASS: {output_subtitle_path}")
                    self.ass_handler.generate_ass(final_segments, output_subtitle_path, speaker_colors=speaker_colors_map if self.args.colorize else None)
                    generated_files.append(str(output_subtitle_path))
                elif fmt == 'json':
                    self.logger.info(f"Generating JSON transcription result: {output_subtitle_path}")
                    with open(output_subtitle_path, 'w', encoding='utf-8') as f:
                        json.dump(final_segments, f, ensure_ascii=False, indent=4)
                    generated_files.append(str(output_subtitle_path))
                else:
                    self.logger.warning(f"Unsupported format for basic generation: {fmt}")
            
            return {
                "success": True,
                "input_source": input_source,
                "output_dir": str(final_output_dir),
                "generated_files": generated_files,
                "final_segments": final_segments
            }
            
        except Exception as e:
            self.logger.error(f"Error during pipeline processing for {input_source}: {e}", exc_info=True)
            return {"success": False, "input_source": input_source, "error": str(e)}
        finally:
            self._cleanup()

    def _cleanup(self):
        """Cleans up temporary files generated during processing."""
        if not self.args.keep_temp:
            if self.processed_audio_path and os.path.exists(self.processed_audio_path):
                os.remove(self.processed_audio_path)
                self.logger.info(f"Cleaned up processed audio file: {self.processed_audio_path}")

            temp_audio_dir = Path(self.args.temp_dir) / "audio_processing"
            if temp_audio_dir.exists():
                shutil.rmtree(temp_audio_dir)
                self.logger.info(f"Cleaned up temporary audio directory: {temp_audio_dir}")
            
            # Additional cleanup for downloaded videos if they are temporary and not original input
            # This needs a more sophisticated way to track downloaded files vs. user-provided ones
            self.logger.debug(f"Basic cleanup done. Consider --keep-temp or more advanced cleanup.")