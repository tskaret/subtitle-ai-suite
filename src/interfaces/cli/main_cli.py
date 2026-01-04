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
from typing import Dict, Any, Optional
from dotenv import load_dotenv

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
from src.formats.srt_handler import SrtHandler
from src.formats.ass_handler import AssHandler
# Assuming setup_logging is now in src/subtitle_suite/utils/logger.py
from src.subtitle_suite.utils.logger import setup_logging

# For now, temporarily remove ConfigManager and DeviceManager until they are fully integrated
# from src.utils.device_manager import DeviceManager # Used in show_system_info
# from src.utils.config_manager import ConfigManager # Used for loading config

class SubtitleCLI:
    """Complete CLI interface for subtitle processing"""
    
    def __init__(self):
        # self.config_manager = ConfigManager() # Not strictly needed for this basic CLI implementation
        self.logger = setup_logging() # Assuming setup_logging takes care of basic logging setup
        self.input_handler = InputHandler()
        self.audio_processor = AudioProcessor()
        self.transcription_processor = TranscriptionProcessor()
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
        
        # Device information (placeholder - needs DeviceManager integration)
        print("\nDevice Information: (Not implemented yet)")
        
        # Python and package versions
        import torch
        import sys
        print(f"\nPython Version: {sys.version}")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Check for optional dependencies (simplified for now)
        print("\nDependency Status: (Simplified for now)")
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
        
        # System resources (placeholder - needs psutil)
        print("\nSystem Resources: (Not implemented yet)")

    def process_single(self, args) -> bool:
        """Process single input"""
        try:
            # 1. Handle Input (local file or download YouTube)
            input_path_processed = None
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

            # 3. Transcribe Audio
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
            transcription_segments = transcription_result.get('segments', [])

            # 4. Generate Subtitles
            base_output_name = Path(input_path_processed).stem
            if args.prefix:
                base_output_name = f"{args.prefix}_{base_output_name}"

            final_output_dir = Path(args.output_dir)
            final_output_dir.mkdir(parents=True, exist_ok=True)

            for fmt in args.format:
                output_subtitle_path = final_output_dir / f"{base_output_name}.{fmt}"
                if fmt == 'srt':
                    self.logger.info(f"Generating SRT: {output_subtitle_path}")
                    # SrtHandler expects list of dicts with 'start', 'end', 'text'
                    srt_data = []
                    for segment in transcription_segments:
                        srt_data.append({
                            'start': segment['start'],
                            'end': segment['end'],
                            'text': segment['text']
                        })
                    self.srt_handler.generate_srt(srt_data, output_subtitle_path)
                elif fmt == 'ass':
                    self.logger.info(f"Generating ASS: {output_subtitle_path}")
                    ass_data = []
                    for segment in transcription_segments:
                        ass_data.append({
                            'start': segment['start'],
                            'end': segment['end'],
                            'text': segment['text']
                            # Add 'speaker' key here if speaker diarization is integrated
                        })
                    self.ass_handler.generate_ass(ass_data, output_subtitle_path)
                elif fmt == 'json':
                    self.logger.info(f"Generating JSON transcription result: {output_subtitle_path}")
                    with open(output_subtitle_path, 'w', encoding='utf-8') as f:
                        json.dump(transcription_result, f, ensure_ascii=False, indent=4)
                else:
                    self.logger.warning(f"Unsupported format for basic generation: {fmt}")

            print(f"✅ Completed: {args.input}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {args.input}: {e}")
            self.logger.error(f"Processing error: {e}", exc_info=True)
            return False
    
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
                    import shutil
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
