"""
Comprehensive logging system for Subtitle AI Suite
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{self.RESET}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted

class SubtitleLogger:
    """Advanced logging system for subtitle processing"""
    
    def __init__(self, 
                 name: str = 'subtitle_ai_suite',
                 log_dir: str = './logs',
                 level: str = 'INFO',
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 enable_json_logging: bool = False,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Initialize comprehensive logging system
        
        Args:
            name (str): Logger name
            log_dir (str): Directory for log files
            level (str): Logging level
            enable_file_logging (bool): Enable file logging
            enable_console_logging (bool): Enable console logging
            enable_json_logging (bool): Enable structured JSON logging
            max_file_size (int): Maximum log file size in bytes
            backup_count (int): Number of backup files to keep
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = getattr(logging, level.upper())
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.enable_json_logging = enable_json_logging
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            
            # Use colored formatter for console
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file_logging:
            log_file = self.log_dir / f'{self.name}.log'
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.level)
            
            # Use standard formatter for file
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # JSON handler for structured logging
        if self.enable_json_logging:
            json_log_file = self.log_dir / f'{self.name}.json'
            json_handler = logging.handlers.RotatingFileHandler(
                json_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            json_handler.setLevel(self.level)
            json_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(json_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        return self.logger
    
    def log_processing_start(self, input_file: str, config: Dict[str, Any]):
        """Log processing start"""
        self.logger.info(
            f"Starting processing: {input_file}",
            extra={'extra_fields': {
                'event_type': 'processing_start',
                'input_file': input_file,
                'config': config
            }}
        )
    
    def log_processing_complete(self, input_file: str, duration: float, output_files: list):
        """Log processing completion"""
        self.logger.info(
            f"Processing complete: {input_file} ({duration:.2f}s)",
            extra={'extra_fields': {
                'event_type': 'processing_complete',
                'input_file': input_file,
                'duration': duration,
                'output_files': output_files
            }}
        )
    
    def log_processing_error(self, input_file: str, error: Exception):
        """Log processing error"""
        self.logger.error(
            f"Processing error: {input_file} - {error}",
            extra={'extra_fields': {
                'event_type': 'processing_error',
                'input_file': input_file,
                'error_type': type(error).__name__,
                'error_message': str(error)
            }},
            exc_info=True
        )
    
    def log_model_load(self, model_name: str, device: str, load_time: float):
        """Log model loading"""
        self.logger.info(
            f"Model loaded: {model_name} on {device} ({load_time:.2f}s)",
            extra={'extra_fields': {
                'event_type': 'model_load',
                'model_name': model_name,
                'device': device,
                'load_time': load_time
            }}
        )
    
    def log_speaker_analysis(self, input_file: str, speaker_count: int, primary_speakers: list):
        """Log speaker analysis results"""
        self.logger.info(
            f"Speaker analysis: {input_file} - {speaker_count} speakers detected",
            extra={'extra_fields': {
                'event_type': 'speaker_analysis',
                'input_file': input_file,
                'speaker_count': speaker_count,
                'primary_speakers': primary_speakers
            }}
        )
    
    def log_checkpoint_save(self, input_file: str, stage: str, checkpoint_path: str):
        """Log checkpoint save"""
        self.logger.debug(
            f"Checkpoint saved: {input_file} at stage {stage}",
            extra={'extra_fields': {
                'event_type': 'checkpoint_save',
                'input_file': input_file,
                'stage': stage,
                'checkpoint_path': checkpoint_path
            }}
        )
    
    def log_system_info(self, system_info: Dict[str, Any]):
        """Log system information"""
        self.logger.info(
            "System information logged",
            extra={'extra_fields': {
                'event_type': 'system_info',
                'system_info': system_info
            }}
        )

def setup_logging(name: str = 'subtitle_ai_suite',
                  log_dir: str = './logs',
                  level: str = 'INFO',
                  enable_file_logging: bool = True,
                  enable_console_logging: bool = True,
                  enable_json_logging: bool = False) -> logging.Logger:
    """
    Setup logging system with defaults
    
    Args:
        name (str): Logger name
        log_dir (str): Log directory
        level (str): Logging level
        enable_file_logging (bool): Enable file logging
        enable_console_logging (bool): Enable console logging
        enable_json_logging (bool): Enable JSON logging
    
    Returns:
        logging.Logger: Configured logger
    """
    subtitle_logger = SubtitleLogger(
        name=name,
        log_dir=log_dir,
        level=level,
        enable_file_logging=enable_file_logging,
        enable_console_logging=enable_console_logging,
        enable_json_logging=enable_json_logging
    )
    
    return subtitle_logger.get_logger()

def get_performance_logger() -> logging.Logger:
    """Get performance-specific logger"""
    return setup_logging(
        name='subtitle_ai_suite.performance',
        log_dir='./logs/performance',
        level='DEBUG',
        enable_json_logging=True
    )

def get_error_logger() -> logging.Logger:
    """Get error-specific logger"""
    return setup_logging(
        name='subtitle_ai_suite.errors',
        log_dir='./logs/errors',
        level='ERROR',
        enable_json_logging=True
    )