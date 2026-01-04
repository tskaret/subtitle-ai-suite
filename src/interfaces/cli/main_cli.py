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

# Import the SubtitlePipelineManager and BatchProcessor
from src.processing.pipeline_manager import SubtitlePipelineManager
from src.processing.batch_processor import BatchProcessor # New Import

# Assuming setup_logging is now in src/subtitle_suite/utils/logger.py
from src.subtitle_suite.utils.logger import setup_logging

# For now, temporarily re-add DeviceManager import for system info
from src.utils.device_manager import DeviceManager # Used in show_system_info
import psutil # Used in show_system_info

class SubtitleCLI:
    """Complete CLI interface for subtitle processing"""
    
    def __init__(self):
        self.logger = setup_logging()
        
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

        # AI Options (New group for advanced AI features)
        ai_group = parser.add_argument_group('AI Options')
        ai_group.add_argument(
            '--enable-emotion-detection',
            action='store_true',
            help='Enable emotion detection in audio'
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
        
        # Check for optional dependencies
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

    def process_single(self, args) -> bool:
        """Process single input using SubtitlePipelineManager"""
        try:
            self.logger.info(f"Starting pipeline for: {args.input}")
            pipeline_manager = SubtitlePipelineManager(args)
            result = pipeline_manager.process(args.input)
            
            if result["success"]:
                print(f"✅ Completed: {args.input}")
                self.logger.info(f"Pipeline completed successfully for {args.input}")
                return True
            else:
                print(f"❌ Error processing {args.input}: {result.get('error', 'Unknown error')}")
                self.logger.error(f"Pipeline failed for {args.input}: {result.get('error', 'Unknown error')}")
                return False
            
        except Exception as e:
            print(f"❌ Unexpected error during single file processing: {e}")
            self.logger.error(f"Unexpected error in process_single: {e}", exc_info=True)
            return False
    
    def process_batch(self, args) -> bool:
        """Process batch directory or playlist using BatchProcessor"""
        try:
            # Create a dictionary from args for BatchProcessor config
            batch_config = vars(args) 
            batch_processor = BatchProcessor(batch_config, max_workers=args.parallel)

            jobs_added = 0
            if args.batch:
                self.logger.info(f"Adding directory {args.batch} to batch processing.")
                jobs_added = batch_processor.add_directory(args.batch, args.output_dir)
            elif args.playlist:
                self.logger.info(f"Adding playlist {args.playlist} to batch processing.")
                jobs_added = batch_processor.add_playlist(args.playlist, args.output_dir)

            if jobs_added == 0:
                print("❌ No jobs were added for batch processing.")
                return False

            self.logger.info(f"Starting batch processing for {jobs_added} jobs.")
            report = batch_processor.process_all()

            if report['failed'] == 0:
                print(f"✅ Batch processing completed successfully for all {report['completed']} jobs.")
                return True
            elif report['completed'] > 0:
                print(f"⚠️ Batch processing completed with {report['completed']} successes and {report['failed']} failures.")
                return False
            else:
                print(f"❌ Batch processing failed for all {report['failed']} jobs.")
                return False

        except Exception as e:
            print(f"❌ Unexpected error during batch processing: {e}")
            self.logger.error(f"Unexpected error in process_batch: {e}", exc_info=True)
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
            if args.batch or args.playlist: # Handle both batch and playlist with process_batch
                success = self.process_batch(args)
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
            # Cleanup is now handled by SubtitlePipelineManager's _cleanup method for single process
            # For batch, individual pipelines handle their own cleanup.
            # A central batch cleanup might be added later if needed.
            pass


def main():
    """CLI entry point"""
    cli = SubtitleCLI()
    return cli.run()

if __name__ == '__main__':
    sys.exit(main())
