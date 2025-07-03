"""
Comprehensive error handling for Subtitle AI Suite
"""

import logging
import traceback
import sys
from typing import Optional, Dict, Any, Callable
from functools import wraps
from pathlib import Path

class SubtitleProcessingError(Exception):
    """Base exception for subtitle processing errors"""
    pass

class InputError(SubtitleProcessingError):
    """Errors related to input validation and processing"""
    pass

class AudioProcessingError(SubtitleProcessingError):
    """Errors during audio extraction and processing"""
    pass

class TranscriptionError(SubtitleProcessingError):
    """Errors during speech-to-text transcription"""
    pass

class SpeakerDiarizationError(SubtitleProcessingError):
    """Errors during speaker diarization"""
    pass

class OutputGenerationError(SubtitleProcessingError):
    """Errors during subtitle output generation"""
    pass

class ModelLoadError(SubtitleProcessingError):
    """Errors during AI model loading"""
    pass

class ConfigurationError(SubtitleProcessingError):
    """Errors in configuration or setup"""
    pass

class DependencyError(SubtitleProcessingError):
    """Errors related to missing dependencies"""
    pass

class ErrorHandler:
    """Comprehensive error handling system"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler
        
        Args:
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
    
    def handle_error(self, 
                     error: Exception, 
                     context: Optional[str] = None,
                     reraise: bool = True,
                     log_level: str = 'ERROR') -> Optional[Exception]:
        """
        Handle and log errors
        
        Args:
            error (Exception): The exception to handle
            context (str, optional): Additional context information
            reraise (bool): Whether to reraise the exception
            log_level (str): Logging level for the error
        
        Returns:
            Exception: The original exception if not reraised
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Count errors
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create log message
        log_message = f"{error_type}: {error_message}"
        if context:
            log_message = f"{context} - {log_message}"
        
        # Log the error
        log_method = getattr(self.logger, log_level.lower())
        log_method(log_message, exc_info=True)
        
        # Additional handling based on error type
        self._handle_specific_error(error, context)
        
        if reraise:
            raise error
        
        return error
    
    def _handle_specific_error(self, error: Exception, context: Optional[str]):
        """Handle specific error types with custom logic"""
        
        if isinstance(error, InputError):
            self.logger.warning("Input validation failed. Please check file paths and formats.")
        
        elif isinstance(error, AudioProcessingError):
            self.logger.warning("Audio processing failed. Check FFmpeg installation and audio file integrity.")
        
        elif isinstance(error, ModelLoadError):
            self.logger.warning("Model loading failed. Check internet connection and model availability.")
        
        elif isinstance(error, DependencyError):
            self.logger.error("Missing dependencies. Run: pip install -r requirements.txt")
        
        elif isinstance(error, ConfigurationError):
            self.logger.error("Configuration error. Check settings and environment variables.")
    
    def validate_input(self, input_path: str) -> bool:
        """
        Validate input file or URL
        
        Args:
            input_path (str): Path to input file or URL
        
        Returns:
            bool: True if valid
        
        Raises:
            InputError: If input is invalid
        """
        try:
            # Check if it's a URL
            if input_path.startswith(('http://', 'https://', 'www.')):
                # Basic URL validation
                if 'youtube.com' in input_path or 'youtu.be' in input_path:
                    return True
                else:
                    raise InputError(f"Unsupported URL: {input_path}")
            
            # Check local file
            file_path = Path(input_path)
            if not file_path.exists():
                raise InputError(f"File not found: {input_path}")
            
            # Check file extension
            supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.flac', '.m4a'}
            if file_path.suffix.lower() not in supported_extensions:
                raise InputError(f"Unsupported file format: {file_path.suffix}")
            
            # Check file size (limit to 2GB)
            max_size = 2 * 1024 * 1024 * 1024  # 2GB
            if file_path.stat().st_size > max_size:
                raise InputError(f"File too large: {file_path.stat().st_size / 1024**3:.1f}GB (max 2GB)")
            
            return True
            
        except Exception as e:
            if isinstance(e, InputError):
                raise
            else:
                raise InputError(f"Input validation failed: {e}")
    
    def validate_dependencies(self) -> bool:
        """
        Validate required dependencies
        
        Returns:
            bool: True if all dependencies are available
        
        Raises:
            DependencyError: If critical dependencies are missing
        """
        required_modules = {
            'torch': 'PyTorch',
            'whisper': 'OpenAI Whisper',
            'speechbrain': 'SpeechBrain',
            'moviepy': 'MoviePy',
            'pydub': 'PyDub'
        }
        
        missing_modules = []
        
        for module, name in required_modules.items():
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(name)
        
        if missing_modules:
            raise DependencyError(f"Missing required dependencies: {', '.join(missing_modules)}")
        
        # Check FFmpeg
        import subprocess
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL, 
                          check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise DependencyError("FFmpeg not found. Please install FFmpeg.")
        
        return True
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts"""
        return self.error_counts.copy()
    
    def reset_error_counts(self):
        """Reset error counters"""
        self.error_counts.clear()

def error_handler(error_type: type = SubtitleProcessingError,
                  context: Optional[str] = None,
                  reraise: bool = True,
                  log_level: str = 'ERROR'):
    """
    Decorator for automatic error handling
    
    Args:
        error_type (type): Type of exception to catch
        context (str, optional): Context information
        reraise (bool): Whether to reraise the exception
        log_level (str): Logging level
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            try:
                return func(*args, **kwargs)
            except error_type as e:
                func_context = context or f"{func.__module__}.{func.__name__}"
                return handler.handle_error(e, func_context, reraise, log_level)
            except Exception as e:
                # Convert unexpected errors to processing errors
                processing_error = SubtitleProcessingError(f"Unexpected error in {func.__name__}: {e}")
                func_context = context or f"{func.__module__}.{func.__name__}"
                return handler.handle_error(processing_error, func_context, reraise, log_level)
        return wrapper
    return decorator

def safe_execute(func: Callable, 
                 *args, 
                 default_return=None,
                 log_errors: bool = True,
                 **kwargs) -> Any:
    """
    Safely execute a function with error handling
    
    Args:
        func (Callable): Function to execute
        *args: Function arguments
        default_return: Default return value on error
        log_errors (bool): Whether to log errors
        **kwargs: Function keyword arguments
    
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
        return default_return

class GracefulShutdown:
    """Handle graceful shutdown on interruption"""
    
    def __init__(self):
        self.shutdown = False
        self.cleanup_functions = []
    
    def register_cleanup(self, cleanup_func: Callable):
        """Register cleanup function"""
        self.cleanup_functions.append(cleanup_func)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.shutdown = True
        
        # Execute cleanup functions
        for cleanup_func in self.cleanup_functions:
            try:
                cleanup_func()
            except Exception as e:
                print(f"Error in cleanup: {e}")
        
        sys.exit(0)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        import signal
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

def create_error_report(error: Exception, 
                       context: Optional[str] = None,
                       system_info: bool = True) -> Dict[str, Any]:
    """
    Create detailed error report
    
    Args:
        error (Exception): The exception
        context (str, optional): Additional context
        system_info (bool): Include system information
    
    Returns:
        Dict containing error report
    """
    import platform
    import sys
    
    report = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'context': context,
        'timestamp': str(logging.Formatter().formatTime(logging.LogRecord(
            '', 0, '', 0, '', (), None
        )))
    }
    
    if system_info:
        report['system_info'] = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'python_executable': sys.executable
        }
    
    return report