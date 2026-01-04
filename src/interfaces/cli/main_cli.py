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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.subtitle_processor import EnhancedSubtitleProcessor
from utils.device_manager import DeviceManager
from utils.config_manager import ConfigManager
from utils.logger import setup_logging

class SubtitleCLI:
    """Complete CLI interface for subtitle processing"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.logger = setup_logging()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser"""
        parser = argparse.ArgumentParser(
            description="Subtitle AI Suite - Professional subtitle generation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s video.mp4                           # Basic processing
  %(prog)s video.mp4 --colorize                # With speaker colors
  %(prog)s "https://youtube.com/watch?v=..."   # YouTube video
  %(prog)s --batch ./videos/                   # Batch process folder
  %(prog)s video.mp4 --format ass srt vtt      # Multiple formats
  %(prog)s video.mp4 --language en --translate es  # Translation
            """
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
            
        if args.input and args.batch:
            print("Error: Cannot specify both input file and batch directory")
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
        DeviceManager.print_device_info()
        
        # Python and package versions
        import torch
        import sys
        print(f"\nPython Version: {sys.version}")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Check for optional dependencies
        optional_deps = {
            'whisper': 'OpenAI Whisper',
            'speechbrain': 'SpeechBrain',
            'moviepy': 'MoviePy',
            'yt_dlp': 'YT-DLP'
        }
        
        print("\nDependency Status:")
        for module, name in optional_deps.items():
            try:
                __import__(module)
                print(f"  ✓ {name}")
            except ImportError:
                print(f"  ❌ {name} (not installed)")
        
        # System resources
        import psutil
        print(f"\nSystem Resources:")
        print(f"  CPU Cores: {psutil.cpu_count()}")
        print(f"  Memory: {psutil.virtual_memory().total / 1e9:.1f} GB")
        
    def process_single(self, args) -> bool:
        """Process single input"""
        try:
            config = self.build_config(args)
            processor = EnhancedSubtitleProcessor(config)
            
            print(f"📝 Processing: {args.input}")
            result = processor.process_audio(args.input)
            
            print(f"✅ Completed: {args.input}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {args.input}: {e}")
            self.logger.error(f"Processing error: {e}", exc_info=True)
            return False
    
    def process_batch(self, args) -> bool:
        """Process batch directory"""
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"❌ Batch directory not found: {batch_dir}")
            return False
        
        # Find media files
        media_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.flac'}
        media_files = []
        
        for file_path in batch_dir.rglob('*'):
            if file_path.suffix.lower() in media_extensions:
                media_files.append(file_path)
        
        if not media_files:
            print(f"❌ No media files found in: {batch_dir}")
            return False
        
        print(f"📁 Found {len(media_files)} media files")
        
        success_count = 0
        for file_path in media_files:
            args.input = str(file_path)
            if self.process_single(args):
                success_count += 1
        
        print(f"✅ Processed {success_count}/{len(media_files)} files")
        return success_count > 0
    
    def build_config(self, args) -> Dict[str, Any]:
        """Build configuration from arguments"""
        config = {
            'output_dir': args.output_dir,
            'temp_dir': args.temp_dir,
            'whisper_model': args.whisper_model,
            'language': args.language,
            'formats': args.format,
            'speaker_colorization': {
                'enabled': args.colorize,
                'threshold': args.speaker_threshold,
                'min_speakers': args.min_speakers,
                'max_speakers': args.max_speakers
            },
            'audio_processing': {
                'quality': args.audio_quality,
                'denoise': args.denoise,
                'normalize': args.normalize
            },
            'processing': {
                'device': args.device,
                'parallel': args.parallel,
                'keep_temp': args.keep_temp
            }
        }
        
        # Load config file if specified
        if args.config:
            config.update(self.config_manager.load_config(args.config))
        
        return config
    
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
        
        # Validate arguments
        if not self.validate_args(args):
            return 1
        
        # Show system info if requested
        if args.info:
            self.show_system_info()
            return 0
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process based on input type
        try:
            if args.batch:
                success = self.process_batch(args)
            elif args.playlist:
                print("❌ Playlist processing not yet implemented")
                return 1
            else:
                success = self.process_single(args)
            
            return 0 if success else 1
            
        except KeyboardInterrupt:
            print("\n❌ Processing interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            return 1

def main():
    """CLI entry point"""
    cli = SubtitleCLI()
    return cli.run()

if __name__ == '__main__':
    sys.exit(main())