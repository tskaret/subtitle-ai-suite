#!/usr/bin/env python3
"""
Complete CLI interface for Subtitle AI Suite
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import shutil # For rmtree

# Load environment variables from .env file
load_dotenv()

# TEMPORARY: Adjusting sys.path for direct execution during development
# This will be removed once the project is installable as a package
current_file_path = Path(__file__).resolve()
# Assuming project root is 3 levels up from src/interfaces/cli/main_cli.py
project_root = current_file_path.parents[3]
sys.path.insert(0, str(project_root))

from src.core.input_handler import InputHandler
from src.core.audio_processor import AudioProcessor
from src.core.transcription import TranscriptionProcessor
from src.core.speaker_analyzer import ModernSpeakerDiarization, SpeakerProfile # New import
from src.core.synchronizer import Synchronizer # New import
from src.formats.srt_handler import SrtHandler
from src.formats.ass_handler import AssHandler
from src.subtitle_suite.utils.logger import setup_logging

# For now, temporarily re-add DeviceManager import for system info
from src.utils.device_manager import DeviceManager # Used in show_system_info
import psutil # Used in show_system_info

class SubtitleCLI:
    """Complete CLI interface for subtitle processing"""
    
    # Predefined color palette for speakers (from gameplan.md)
    DEFAULT_COLORS = {
        'red': '&H6B6BFF&',
        'cyan': '&HC4CD4E&',
        'blue': '&HD1B745&',
        'green': '&HB4CE96&',
        'yellow': '&HA7EAFF&',
        'plum': '&HDDA0DD&',
        'orange': '&H129CF3&'
    }

    def __init__(self):
        self.logger = setup_logging()
        self.input_handler = InputHandler()
        self.audio_processor = AudioProcessor()
        self.transcription_processor = TranscriptionProcessor()
        self.speaker_diarization = ModernSpeakerDiarization() # Initialize Diarization
        self.synchronizer = Synchronizer() # Initialize Synchronizer
        self.srt_handler = SrtHandler()
        self.ass_handler = AssHandler()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser"""
        parser = argparse.ArgumentParser(
            description="Subtitle AI Suite - Professional subtitle generation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f'''
Examples:
  %(prog)s video.mp4                           # Basic processing
  %(prog)s video.mp4 --colorize                # With speaker colors
  %(prog)s "https://youtube.com/watch?v=..."   # YouTube video
  %(prog)s --batch ./videos/                   # Batch process folder
  %(prog)s video.mp4 --format ass srt vtt      # Multiple formats
  %(prog)s video.mp4 --language en --translate es  # Translation
            '''
        )
        
        # Input arguments
        input_group = parser.add_argument_group('Input Options')
        input_group.add_argument(
            'input',
            nargs='?',
            help='Input video file, audio file, or YouTube URL'
        )
        input_group.add_argument(
            '--batch', '-b',
            metavar='DIR',
            help='Process all media files in directory'
        )
        input_group.add_argument(
            '--playlist', '-p',
            metavar='URL',
            help='Process YouTube playlist'
        )
        
        # Output arguments
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument(
            '--output-dir', '-o',
            default='./output',
            help='Output directory (default: ./output)'
        )
        output_group.add_argument(
            '--format', '-f',
            nargs='+',
            choices=['srt', 'ass', 'vtt', 'json'],
            default=['srt', 'ass'],
            help='Output formats (default: srt ass)'
        )
        output_group.add_argument(
            '--prefix',
            help='Prefix for output files'
        )
        
        # Processing arguments
        processing_group = parser.add_argument_group('Processing Options')
        processing_group.add_argument(
            '--whisper-model', '-m',
            choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
            default='large-v2',
            help='Whisper model size (default: large-v2)'
        )
        processing_group.add_argument(
            '--language', '-l',
            help='Input language (ISO 639-1 code, e.g., en, es, fr)'
        )
        processing_group.add_argument(
            '--translate', '-t',
            help='Translate to language (ISO 639-1 code)'
        )
        processing_group.add_argument(
            '--device',
            choices=['auto', 'cpu', 'cuda', 'mps'],
            default='auto',
            help='Processing device (default: auto)'
        )
        
        # Speaker options
        speaker_group = parser.add_argument_group('Speaker Options')
        speaker_group.add_argument(
            '--colorize', '-c',
            action='store_true',
            help='Enable speaker colorization'
        )
        speaker_group.add_argument(
            '--speaker-threshold',
            type=float,
            default=0.95,
            help='Speaker identification threshold (default: 0.95)'
        )
        speaker_group.add_argument(
            '--min-speakers',
            type=int,
            default=1,
            help='Minimum number of speakers (default: 1)'
        )
        speaker_group.add_argument(
            '--max-speakers',
            type=int,
            default=10,
            help='Maximum number of speakers (default: 10)'
        )
        
        # Quality options
        quality_group = parser.add_argument_group('Quality Options')
        quality_group.add_argument(
            '--audio-quality',
            choices=['low', 'medium', 'high'],
            default='high',
            help='Audio extraction quality (default: high)'
        )
        quality_group.add_argument(
            '--denoise',
            action='store_true',
            help='Enable audio denoising'
        )
        quality_group.add_argument(
            '--normalize',
            action='store_true',
            help='Enable audio normalization'
        )
        
        # Advanced options
        advanced_group = parser.add_argument_group('Advanced Options')
        advanced_group.add_argument(
            '--config',
            help='Configuration file path'
        )
        advanced_group.add_argument(
            '--resume',
            help='Resume from checkpoint file'
        )
        advanced_group.add_argument(
            '--temp-dir',
            default='./temp',
            help='Temporary files directory (default: ./temp)'
        )
        advanced_group.add_argument(
            '--keep-temp',
            action='store_true',
            help='Keep temporary files after processing'
        )
        advanced_group.add_argument(
            '--parallel',
            type=int,
            default=1,
            help='Number of parallel processes (default: 1)'
        )
        
        # Utility options
        utility_group = parser.add_argument_group('Utility Options')
        utility_group.add_argument(
            '--verbose', '-v',
            action='count',
            default=0,
            help='Increase verbosity (-v, -vv, -vvv)'
        )
        utility_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Quiet mode (minimal output)'
        )
        utility_group.add_argument(
            '--version',
            action='version',
            version='Subtitle AI Suite 0.1.0'
        )
        utility_group.add_argument(
            '--info',
            action='store_true',
            help='Show system information and exit'
        )
        
        return parser
    
    def validate_args(self, args) -> bool:
        """Validate command line arguments"""
        if args.info:
            return True
            
        if not args.input and not args.batch and not args.playlist:
            print("Error: Must specify input file, batch directory, or playlist URL")
            return False
            
        if args.input and (args.batch or args.playlist):
            print("Error: Cannot specify input file with batch directory or playlist URL")
            return False
        
        if args.batch and args.playlist:
            print("Error: Cannot specify both batch directory and playlist URL")
            return False

        if args.language and len(args.language) != 2:
            print("Error: Language must be 2-letter ISO 639-1 code (e.g., 'en', 'es')")
            return False
            
        if args.translate and len(args.translate) != 2:
            print("Error: Translation language must be 2-letter ISO 639-1 code")
            return False
            
        return True
    
    def show_system_info(self):
        """Show system information"""
        print("🖥️  Subtitle AI Suite - System Information")
        print("=" * 50)
        
        # Device information
        DeviceManager.print_device_info() # Now imported
        
        # Python and package versions
        import torch
        import sys
        print(f"\nPython Version: {sys.version}")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Check for optional dependencies (simplified for now)
        print("\nDependency Status:")
        try:
            import whisper
            print("  ✓ OpenAI Whisper")
        except ImportError:
            print("  ❌ OpenAI Whisper (not installed)")
        try:
            import moviepy
            print("  ✓ MoviePy")
        except ImportError:
            print("  ❌ MoviePy (not installed)")
        try:
            import pyannote.audio
            print("  ✓ Pyannote Audio")
        except ImportError:
            print("  ❌ Pyannote Audio (not installed)")

        # System resources
        print("\nSystem Resources:")
        print(f"  CPU Cores: {psutil.cpu_count()}")
        print(f"  Memory: {psutil.virtual_memory().total / 1e9:.1f} GB")


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


    def process_single(self, args) -> bool:
        """Process single input"""
        input_path_processed = None
        processed_audio_path = None
        try:
            # 1. Handle Input (local file or download YouTube)
            if args.input:
                if "youtube.com" in args.input or "youtu.be" in args.input:
                    self.logger.info(f"Downloading YouTube video: {args.input}")
                    input_path_processed = self.input_handler.download_youtube_video(args.input)
                else:
                    self.logger.info(f"Processing local file: {args.input}")
                    input_path_processed = self.input_handler.process_local_file(args.input)
            
            if not input_path_processed:
                raise ValueError("No valid input provided for processing.")

            # Ensure temp directory exists for AudioProcessor to use
            temp_output_dir = Path(args.temp_dir) / "audio_processing"
            temp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. Process Audio
            self.logger.info(f"Extracting and processing audio from: {input_path_processed}")
            processed_audio_path = self.audio_processor.process(
                str(input_path_processed), 
                output_path=str(temp_output_dir / f"{Path(input_path_processed).stem}_processed.wav")
            )

            # 3. Speaker Diarization
            speaker_profiles = []
            if args.colorize: # Only run diarization if colorization is requested
                self.logger.info(f"Performing speaker diarization on: {processed_audio_path}")
                speaker_profiles = self.speaker_diarization.process_audio(processed_audio_path)
                self.logger.info(f"Diarization found {len(speaker_profiles)} speakers.")


            # 4. Transcribe Audio
            self.logger.info(f"Transcribing audio: {processed_audio_path}")
            self.transcription_processor.model_name = args.whisper_model # Update model if specified
            self.transcription_processor.device = args.device if args.device != 'auto' else None
            
            # Re-load model if parameters changed, or if it's the first run
            if not self.transcription_processor.model or \
               self.transcription_processor.model.model_name != args.whisper_model or \
               self.transcription_processor.device != (args.device if args.device != 'auto' else None):
                self.transcription_processor._load_model() # Force reload with new params
                
            transcription_result = self.transcription_processor.transcribe_audio(
                processed_audio_path, language=args.language
            )
            raw_transcription_segments = transcription_result.get('segments', [])

            # 5. Synchronize Transcription with Diarization (if enabled)
            synchronized_segments = raw_transcription_segments
            if speaker_profiles:
                self.logger.info("Synchronizing transcription segments with speaker diarization.")
                synchronized_segments = self.synchronizer.synchronize(raw_transcription_segments, speaker_profiles)
            
            # 6. Generate Subtitles
            base_output_name = Path(input_path_processed).stem
            if args.prefix:
                base_output_name = f"{args.prefix}_{base_output_name}"

            final_output_dir = Path(args.output_dir)
            final_output_dir.mkdir(parents=True, exist_ok=True)

            speaker_colors_map = {}
            if args.colorize and speaker_profiles:
                self.logger.info("Analyzing speaker distribution for colorization.")
                speaker_colors_map = self._analyze_speaker_distribution(synchronized_segments, speaker_profiles, args.speaker_threshold)
            elif args.colorize and not speaker_profiles:
                self.logger.warning("Colorization requested but no speaker profiles available.")


            for fmt in args.format:
                output_subtitle_path = final_output_dir / f"{base_output_name}.{fmt}"
                if fmt == 'srt':
                    self.logger.info(f"Generating SRT: {output_subtitle_path}")
                    # SrtHandler expects list of dicts with 'start', 'end', 'text'
                    srt_data = []
                    for segment in synchronized_segments: # Use synchronized segments
                        srt_data.append({
                            'start': segment['start'],
                            'end': segment['end'],
                            'text': f"[{segment['speaker']}] {segment['text']}" if 'speaker' in segment else segment['text']
                        })
                    self.srt_handler.generate_srt(srt_data, output_subtitle_path)
                elif fmt == 'ass':
                    self.logger.info(f"Generating ASS: {output_subtitle_path}")
                    # AssHandler can take 'speaker' if available. Use synchronized segments.
                    # Pass speaker_colors_map if colorization is enabled
                    self.ass_handler.generate_ass(synchronized_segments, output_subtitle_path, speaker_colors=speaker_colors_map if args.colorize else None)
                elif fmt == 'json':
                    self.logger.info(f"Generating JSON transcription result: {output_subtitle_path}")
                    # Dump the synchronized segments for JSON output
                    with open(output_subtitle_path, 'w', encoding='utf-8') as f:
                        json.dump(synchronized_segments, f, ensure_ascii=False, indent=4)
                else:
                    self.logger.warning(f"Unsupported format for basic generation: {fmt}")

            print(f"✅ Completed: {args.input}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {args.input}: {e}")
            self.logger.error(f"Processing error: {e}", exc_info=True)
            return False
        finally:
            if processed_audio_path and os.path.exists(processed_audio_path) and not args.keep_temp:
                os.remove(processed_audio_path)
                self.logger.info(f"Cleaned up processed audio file: {processed_audio_path}")
            if not args.keep_temp:
                # Basic cleanup, a more robust cleanup will be in a dedicated util or processor
                temp_audio_dir = Path(args.temp_dir) / "audio_processing"
                if temp_audio_dir.exists():
                    shutil.rmtree(temp_audio_dir)
                    self.logger.info(f"Cleaned up temporary audio directory: {temp_audio_dir}")
                # Also clean up downloaded videos if they are in self.input_handler.output_dir
                # This needs a more sophisticated way to track downloaded files vs. user-provided ones
                self.logger.info(f"Basic cleanup done. Consider --keep-temp or more advanced cleanup.")

    
    def process_batch(self, args) -> bool:
        """Process batch directory (placeholder for now)"""
        print("❌ Batch processing not yet fully implemented with new pipeline.")
        return False
    
    def build_config(self, args) -> Dict[str, Any]:
        """Build configuration from arguments (simplified for now)"""
        # This function might become more complex when ConfigManager is fully integrated
        # For now, relevant config is passed directly or updated in component instances
        return {} # Return empty dict as config is managed by individual processors
    
    def run(self, argv=None):
        """Main CLI entry point"""
        parser = self.create_parser()
        args = parser.parse_args(argv)
        
        # Setup logging based on verbosity
        if args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif args.verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)
        elif args.verbose >= 1:
            logging.getLogger().setLevel(logging.INFO)
        else: # Default logging level if not specified
            logging.getLogger().setLevel(logging.WARNING)
        
        # Validate arguments
        if not self.validate_args(args):
            return 1
        
        # Show system info if requested
        if args.info:
            self.show_system_info()
            return 0
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True) # Ensure temp_dir exists

        # Process based on input type
        try:
            if args.batch:
                success = self.process_batch(args)
            elif args.playlist:
                print("❌ Playlist processing not yet fully implemented.")
                return 1
            else:
                success = self.process_single(args)
            
            return 0 if success else 1
            
        except KeyboardInterrupt:
            print("\n❌ Processing interrupted by user")
            self.logger.info("Processing interrupted by user.")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.logger.error(f"Unexpected error in CLI run: {e}", exc_info=True)
            return 1
        finally:
            if not args.keep_temp:
                # Basic cleanup, a more robust cleanup will be in a dedicated util or processor
                temp_audio_dir = Path(args.temp_dir) / "audio_processing"
                if temp_audio_dir.exists():
                    shutil.rmtree(temp_audio_dir)
                    self.logger.info(f"Cleaned up temporary audio directory: {temp_audio_dir}")
                # Also clean up downloaded videos if they are in self.input_handler.output_dir
                # This needs a more sophisticated way to track downloaded files vs. user-provided ones
                self.logger.info(f"Basic cleanup done. Consider --keep-temp or more advanced cleanup.")


def main():
    """CLI entry point"""
    cli = SubtitleCLI()
    return cli.run()

if __name__ == '__main__':
    sys.exit(main())