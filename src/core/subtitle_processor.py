import os
import json
import torch
import logging
import whisper
import torchaudio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from .speaker_diarization import AdvancedSpeakerDiarization, SpeakerProfile

@dataclass
class SubtitleSegment:
    """
    Comprehensive subtitle segment with speaker information
    """
    text: str
    start: float
    end: float
    speaker_id: Optional[str] = None
    speaker_confidence: float = 0.0
    speaker_color: Optional[str] = None
    language: Optional[str] = None

class EnhancedSubtitleProcessor:
    """
    Advanced subtitle processing pipeline with integrated speaker diarization
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhanced subtitle processing pipeline
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.temp_dir = config.get('temp_dir', './temp')
        self.output_dir = config.get('output_dir', './output')
        
        # Create necessary directories
        for dir_path in [self.temp_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize key components
        self.whisper_model = self._load_whisper_model()
        self.speaker_diarization = AdvancedSpeakerDiarization(config)
        
        # Color palette for speakers
        self.speaker_colors = {
            'red': '&H6B6BFF&',
            'cyan': '&HC4CD4E&',
            'blue': '&HD1B745&',
            'green': '&HB4CE96&',
            'yellow': '&HA7EAFF&',
            'plum': '&HDDA0DD&',
            'orange': '&H129CF3&'
        }

    def _load_whisper_model(self, model_name: str = 'large-v2'):
        """
        Load Whisper model with optimized settings
        
        Args:
            model_name (str): Name of Whisper model to load
        
        Returns:
            Whisper model instance
        """
        try:
            model = whisper.load_model(model_name, device=self.device)
            self.logger.info(f"Loaded Whisper {model_name} model successfully")
            return model
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {e}")
            raise

    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Comprehensive audio processing with speaker diarization
        
        Args:
            audio_path (str): Path to input audio file
        
        Returns:
            Dict containing processing results
        """
        try:
            # Perform speaker diarization
            speaker_profiles = self.speaker_diarization.process_audio(audio_path)
            
            # Transcribe audio with word-level timestamps
            transcription = self._transcribe_with_timestamps(audio_path)
            
            # Synchronize transcription with speaker profiles
            synchronized_subtitles = self._synchronize_speakers(
                transcription['segments'], 
                speaker_profiles
            )
            
            # Prepare result
            result = {
                'input_file': audio_path,
                'speakers': [asdict(profile) for profile in speaker_profiles],
                'subtitles': [asdict(segment) for segment in synchronized_subtitles],
                'language': transcription.get('language', 'unknown')
            }
            
            # Export results
            self._export_results(result)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            raise

    def _transcribe_with_timestamps(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio with word-level timestamps
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            Dict containing transcription results
        """
        try:
            # Transcribe with word-level timestamps
            # Use None for auto-detection in Whisper
            language = self.config.get('language')
            if language == 'auto':
                language = None
            
            result = self.whisper_model.transcribe(
                audio_path, 
                word_timestamps=True,
                language=language
            )
            
            return {
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'segments': [
                    {
                        'text': segment['text'],
                        'start': segment['start'],
                        'end': segment['end'],
                        'words': segment.get('words', [])
                    } for segment in result['segments']
                ]
            }
        
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            raise

    def _synchronize_speakers(
        self, 
        transcription_segments: List[Dict], 
        speaker_profiles: List[SpeakerProfile]
    ) -> List[SubtitleSegment]:
        """
        Synchronize transcription segments with speaker profiles
        
        Args:
            transcription_segments (List[Dict]): Transcription segments
            speaker_profiles (List[SpeakerProfile]): Detected speaker profiles
        
        Returns:
            List of synchronized subtitle segments
        """
        # Assign colors to primary speakers (top 95% of dialogue)
        primary_speakers = sorted(
            speaker_profiles, 
            key=lambda x: x.total_speech_time, 
            reverse=True
        )[:len(self.speaker_colors)]
        
        # Assign colors to primary speakers
        speaker_color_map = {}
        for i, speaker in enumerate(primary_speakers):
            color_name = list(self.speaker_colors.keys())[i]
            speaker_color_map[speaker.id] = self.speaker_colors[color_name]
        
        # Create synchronized subtitle segments
        synchronized_segments = []
        
        for segment in transcription_segments:
            # Basic implementation of speaker assignment
            # In a more advanced version, this would use more sophisticated 
            # methods like embedding similarity or temporal overlap
            speaker_id = primary_speakers[0].id if primary_speakers else None
            
            subtitle_segment = SubtitleSegment(
                text=segment['text'],
                start=segment['start'],
                end=segment['end'],
                speaker_id=speaker_id,
                speaker_color=speaker_color_map.get(speaker_id),
                speaker_confidence=1.0  # Placeholder
            )
            
            synchronized_segments.append(subtitle_segment)
        
        return synchronized_segments

    def _export_results(self, result: Dict[str, Any]):
        """
        Export processing results to various files
        
        Args:
            result (Dict[str, Any]): Processing results
        """
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Export speaker profiles
            speakers_path = os.path.join(self.output_dir, 'speakers.json')
            with open(speakers_path, 'w') as f:
                json.dump(result['speakers'], f, indent=2)
            
            # Export subtitles in SRT format
            srt_path = os.path.join(self.output_dir, 'subtitles.srt')
            self._generate_srt(result['subtitles'], srt_path)
            
            # Export ASS format with speaker colors
            ass_path = os.path.join(self.output_dir, 'subtitles.ass')
            self._generate_ass(result['subtitles'], ass_path)
            
            self.logger.info("Results exported successfully")
        
        except Exception as e:
            self.logger.error(f"Result export error: {e}")
            raise

    def _generate_srt(self, subtitle_segments: List[SubtitleSegment], output_path: str):
        """
        Generate SRT subtitle file
        
        Args:
            subtitle_segments (List[SubtitleSegment]): Subtitle segments
            output_path (str): Path to save SRT file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(subtitle_segments, 1):
                # Handle both dict and object formats
                if isinstance(segment, dict):
                    start_time = self._format_timestamp(segment['start'])
                    end_time = self._format_timestamp(segment['end'])
                    text = segment['text']
                    speaker_id = segment.get('speaker_id', '')
                else:
                    start_time = self._format_timestamp(segment.start)
                    end_time = self._format_timestamp(segment.end)
                    text = segment.text
                    speaker_id = segment.speaker_id
                
                # Write SRT entry
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                
                # Include speaker ID if available
                speaker_prefix = f"[{speaker_id}] " if speaker_id else ""
                f.write(f"{speaker_prefix}{text}\n\n")

    def _generate_ass(self, subtitle_segments: List[SubtitleSegment], output_path: str):
        """
        Generate ASS subtitle file with speaker colors
        
        Args:
            subtitle_segments (List[SubtitleSegment]): Subtitle segments
            output_path (str): Path to save ASS file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # ASS header
            f.write("[Script Info]\n")
            f.write("Title: Subtitle AI Suite Output\n")
            f.write("ScriptType: v4.00+\n")
            f.write("WrapStyle: 0\n")
            f.write("ScaledBorderAndShadow: yes\n")
            
            # Styles
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            
            # Default style
            f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n")
            
            # Speaker styles - handle both dict and object formats
            unique_speakers = set()
            for seg in subtitle_segments:
                speaker_id = seg.get('speaker_id', '') if isinstance(seg, dict) else seg.speaker_id
                if speaker_id:
                    unique_speakers.add(speaker_id)
            
            for i, speaker in enumerate(unique_speakers):
                color = self.speaker_colors.get(list(self.speaker_colors.keys())[i % len(self.speaker_colors)])
                f.write(f"Style: {speaker},Arial,20,{color},&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n")
            
            # Events
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            
            # Subtitle entries
            for segment in subtitle_segments:
                # Handle both dict and object formats
                if isinstance(segment, dict):
                    start_time = self._format_ass_timestamp(segment['start'])
                    end_time = self._format_ass_timestamp(segment['end'])
                    text = segment['text']
                    speaker_id = segment.get('speaker_id', '')
                else:
                    start_time = self._format_ass_timestamp(segment.start)
                    end_time = self._format_ass_timestamp(segment.end)
                    text = segment.text
                    speaker_id = segment.speaker_id
                
                # Use speaker color style if available
                style = speaker_id if speaker_id else "Default"
                
                f.write(f"Dialogue: 0,{start_time},{end_time},{style},,0,0,0,,{text}\n")

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp for SRT file
        
        Args:
            seconds (float): Time in seconds
        
        Returns:
            str: Formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

    def _format_ass_timestamp(self, seconds: float) -> str:
        """
        Format timestamp for ASS file
        
        Args:
            seconds (float): Time in seconds
        
        Returns:
            str: Formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:d}:{minutes:02d}:{secs:05.2f}"