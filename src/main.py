#!/usr/bin/env python3
"""
Main entry point for Subtitle AI Suite
Supports CLI, GUI, and direct Python usage
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.subtitle_processor import EnhancedSubtitleProcessor
from utils.device_manager import DeviceManager
from utils.config_manager import ConfigManager
from utils.logger import setup_logging
from utils.error_handler import ErrorHandler, GracefulShutdown

class SubtitleAISuite:
    """
    High-level API for Subtitle AI Suite
    Allows direct Python usage of the suite's functionality
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the suite with configuration
        
        Args:
            config (Dict, optional): Configuration dictionary
        """
        self.config = config or {}
        # Set defaults if not present
        self.config.setdefault('output_dir', './output')
        self.config.setdefault('temp_dir', './temp')
        self.config.setdefault('whisper_model', 'large-v2')
        self.config.setdefault('device', 'auto')

    def process(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process an input file or URL
        
        Args:
            input_path (str): Path to file or YouTube URL
            **kwargs: Override configuration options
            
        Returns:
            Dict: Processing results
        """
        run_config = self.config.copy()
        run_config.update(kwargs)
        
        processor = EnhancedSubtitleProcessor(run_config)
        return processor.process_audio(input_path)

def create_parser():
    """Create main argument parser"""
    parser = argparse.ArgumentParser(
        description="Subtitle AI Suite - Professional AI-powered subtitle generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Interface selection
    interface_group = parser.add_mutually_exclusive_group()
    interface_group.add_argument(
        '--gui', '-g',
        action='store_true',
        help='Launch GUI interface'
    )
    interface_group.add_argument(
        '--cli', '-c',
        action='store_true',
        help='Use CLI interface (default)'
    )
    
    # Pass remaining arguments to specific interface
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Arguments passed to selected interface'
    )
    
    # Global options
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Subtitle AI Suite 0.1.0'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show system information and exit'
    )
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    parser.add_argument(
        '--config',
        help='Configuration file path'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser

def parse_arguments(args=None):
    """
    Parse arguments helper for testing and direct usage
    
    Args:
        args (list): List of arguments (default: sys.argv[1:])
        
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = create_parser()
    return parser.parse_known_args(args)[0]

def show_system_info():
    """Show comprehensive system information"""
    print("🖥️  Subtitle AI Suite - System Information")
    print("=" * 60)
    
    # Device information
    DeviceManager.print_device_info()
    
    # Python information
    import sys, platform
    print(f"\nPython Information:")
    print(f"  Version: {sys.version}")
    print(f"  Executable: {sys.executable}")
    print(f"  Platform: {platform.platform()}")
    
    # Dependency status
    print(f"\nDependency Status:")
    dependencies = {
        'torch': 'PyTorch',
        'whisper': 'OpenAI Whisper',
        'speechbrain': 'SpeechBrain',
        'moviepy': 'MoviePy',
        'pydub': 'PyDub',
        'yt_dlp': 'YT-DLP',
        'transformers': 'Transformers',
        'librosa': 'Librosa',
        'scipy': 'SciPy',
        'numpy': 'NumPy'
    }
    
    for module, name in dependencies.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name} ({version})")
        except ImportError:
            print(f"  ❌ {name} (not installed)")
    
    # System resources
    try:
        import psutil
        print(f"\nSystem Resources:")
        print(f"  CPU Cores: {psutil.cpu_count()}")
        print(f"  Memory: {psutil.virtual_memory().total / 1e9:.1f} GB")
        print(f"  Disk Space: {psutil.disk_usage('.').total / 1e9:.1f} GB")
    except ImportError:
        print(f"\nSystem Resources: (psutil not available)")

def check_dependencies():
    """Check and validate all dependencies"""
    print("🔍 Checking Dependencies...")
    print("=" * 40)
    
    error_handler = ErrorHandler()
    
    try:
        result = error_handler.validate_dependencies()
        print("✅ All dependencies are satisfied")
        return True
    except Exception as e:
        print(f"❌ Dependency check failed: {e}")
        
        print("\n📦 Installation Instructions:")
        print("1. Install FFmpeg: https://ffmpeg.org/download.html")
        print("2. Install Python dependencies:")
        print("   pip install -r requirements.txt")
        print("3. Download AI models:")
        print("   python scripts/download_models.py")
        
        return False

def launch_gui():
    """Launch GUI interface"""
    try:
        from interfaces.gui import SubtitleGUI
        
        print("🖥️  Launching GUI interface...")
        app = SubtitleGUI()
        app.run()
        
    except ImportError as e:
        print(f"❌ GUI dependencies not available: {e}")
        print("Install GUI dependencies with: pip install tkinter")
        return 1
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        return 1
    
    return 0

def launch_cli(args):
    """Launch CLI interface"""
    try:
        from interfaces.cli import SubtitleCLI
        
        cli = SubtitleCLI()
        return cli.run(args)
        
    except ImportError as e:
        print(f"❌ CLI dependencies not available: {e}")
        return 1
    except Exception as e:
        print(f"❌ CLI launch failed: {e}")
        return 1

def main():
    """Main entry point"""
    parser = create_parser()
    
    # Parse only known args to avoid conflicts with interface-specific args
    args, remaining = parser.parse_known_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Setup graceful shutdown
    shutdown = GracefulShutdown()
    shutdown.setup_signal_handlers()
    
    try:
        # Handle global options
        if args.info:
            show_system_info()
            return 0
        
        if args.check_deps:
            success = check_dependencies()
            return 0 if success else 1
        
        # Load configuration if specified
        if args.config:
            config_manager = ConfigManager(args.config)
            if not config_manager.validate_config():
                print("❌ Configuration validation failed")
                return 1
        
        # Determine interface to launch
        if args.gui:
            return launch_gui()
        elif args.cli or remaining:
            # CLI is default if there are remaining args or explicitly requested
            return launch_cli(remaining)
        else:
            # No arguments provided, show help
            parser.print_help()
            
            # Ask user which interface to use
            print("\n" + "=" * 50)
            print("Choose an interface:")
            print("1. GUI (Graphical User Interface)")
            print("2. CLI (Command Line Interface)")
            print("3. Show system info")
            print("4. Check dependencies")
            
            try:
                choice = input("\nEnter choice (1-4, or press Enter for GUI): ").strip()
                
                if choice in ['1', '']:
                    return launch_gui()
                elif choice == '2':
                    return launch_cli([])
                elif choice == '3':
                    show_system_info()
                    return 0
                elif choice == '4':
                    success = check_dependencies()
                    return 0 if success else 1
                else:
                    print("Invalid choice")
                    return 1
                    
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                return 0
    
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())