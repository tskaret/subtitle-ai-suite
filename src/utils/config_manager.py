"""
Configuration management for Subtitle AI Suite
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

@dataclass
class SubtitleConfig:
    """Configuration dataclass for subtitle processing"""
    
    # Model settings
    whisper_model: str = 'large-v2'
    device: str = 'auto'
    
    # Audio processing
    audio_quality: str = 'high'
    sample_rate: int = 48000
    denoise: bool = True
    normalize: bool = True
    
    # Speaker analysis
    enable_diarization: bool = True
    speaker_threshold: float = 0.95
    min_speakers: int = 1
    max_speakers: int = 10
    enable_colorization: bool = True
    
    # Output settings
    output_formats: list = None
    max_chars_per_line: int = 40
    max_lines_per_subtitle: int = 2
    min_display_time: float = 1.5
    
    # Processing settings
    parallel_workers: int = 1
    keep_temp_files: bool = False
    enable_checkpoints: bool = True
    
    # Directories
    temp_dir: str = './temp'
    output_dir: str = './output'
    models_dir: str = './data/models'
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['srt', 'ass']

class ConfigManager:
    """Configuration manager with multiple source support"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = SubtitleConfig()
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from multiple sources"""
        # 1. Load from environment variables
        self._load_from_env()
        
        # 2. Load from config file
        if self.config_path:
            self._load_from_file(self.config_path)
        else:
            # Try default config files
            default_configs = [
                './config.yaml',
                './config.yml',
                './config.json',
                './.subtitle-ai-suite.yaml',
                os.path.expanduser('~/.subtitle-ai-suite.yaml')
            ]
            
            for config_file in default_configs:
                if os.path.exists(config_file):
                    self._load_from_file(config_file)
                    break
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'SUBTITLE_WHISPER_MODEL': 'whisper_model',
            'SUBTITLE_DEVICE': 'device',
            'SUBTITLE_AUDIO_QUALITY': 'audio_quality',
            'SUBTITLE_SAMPLE_RATE': ('sample_rate', int),
            'SUBTITLE_DENOISE': ('denoise', bool),
            'SUBTITLE_NORMALIZE': ('normalize', bool),
            'SUBTITLE_ENABLE_DIARIZATION': ('enable_diarization', bool),
            'SUBTITLE_SPEAKER_THRESHOLD': ('speaker_threshold', float),
            'SUBTITLE_MIN_SPEAKERS': ('min_speakers', int),
            'SUBTITLE_MAX_SPEAKERS': ('max_speakers', int),
            'SUBTITLE_ENABLE_COLORIZATION': ('enable_colorization', bool),
            'SUBTITLE_PARALLEL_WORKERS': ('parallel_workers', int),
            'SUBTITLE_KEEP_TEMP': ('keep_temp_files', bool),
            'SUBTITLE_TEMP_DIR': 'temp_dir',
            'SUBTITLE_OUTPUT_DIR': 'output_dir',
            'SUBTITLE_MODELS_DIR': 'models_dir'
        }
        
        for env_var, config_attr in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if isinstance(config_attr, tuple):
                    attr_name, attr_type = config_attr
                    try:
                        if attr_type == bool:
                            value = env_value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            value = attr_type(env_value)
                        setattr(self.config, attr_name, value)
                    except ValueError:
                        self.logger.warning(f"Invalid value for {env_var}: {env_value}")
                else:
                    setattr(self.config, config_attr, env_value)
    
    def _load_from_file(self, config_path: str):
        """Load configuration from file"""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                self.logger.warning(f"Config file not found: {config_path}")
                return
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    self.logger.warning(f"Unsupported config file format: {config_path}")
                    return
            
            # Update config with file values
            self._update_config(file_config)
            self.logger.info(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading config file {config_path}: {e}")
    
    def _update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
    
    def get_config(self) -> SubtitleConfig:
        """Get current configuration"""
        return self.config
    
    def get_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return asdict(self.config)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self._update_config(updates)
    
    def save_config(self, output_path: str):
        """Save current configuration to file"""
        try:
            output_path = Path(output_path)
            config_dict = self.get_dict()
            
            with open(output_path, 'w') as f:
                if output_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False)
                elif output_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {output_path.suffix}")
            
            self.logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            raise
    
    def validate_config(self) -> bool:
        """Validate current configuration"""
        errors = []
        
        # Validate whisper model
        valid_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
        if self.config.whisper_model not in valid_models:
            errors.append(f"Invalid whisper model: {self.config.whisper_model}")
        
        # Validate device
        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if self.config.device not in valid_devices:
            errors.append(f"Invalid device: {self.config.device}")
        
        # Validate audio quality
        valid_qualities = ['low', 'medium', 'high']
        if self.config.audio_quality not in valid_qualities:
            errors.append(f"Invalid audio quality: {self.config.audio_quality}")
        
        # Validate numeric ranges
        if not 8000 <= self.config.sample_rate <= 48000:
            errors.append(f"Invalid sample rate: {self.config.sample_rate}")
        
        if not 0.0 <= self.config.speaker_threshold <= 1.0:
            errors.append(f"Invalid speaker threshold: {self.config.speaker_threshold}")
        
        if self.config.min_speakers < 1 or self.config.min_speakers > self.config.max_speakers:
            errors.append(f"Invalid speaker range: {self.config.min_speakers}-{self.config.max_speakers}")
        
        # Validate directories
        for dir_attr in ['temp_dir', 'output_dir', 'models_dir']:
            dir_path = getattr(self.config, dir_attr)
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create {dir_attr} '{dir_path}': {e}")
        
        if errors:
            for error in errors:
                self.logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def create_default_config(self, output_path: str):
        """Create a default configuration file"""
        default_config = {
            'whisper_model': 'large-v2',
            'device': 'auto',
            'audio_quality': 'high',
            'sample_rate': 48000,
            'denoise': True,
            'normalize': True,
            'enable_diarization': True,
            'speaker_threshold': 0.95,
            'min_speakers': 1,
            'max_speakers': 10,
            'enable_colorization': True,
            'output_formats': ['srt', 'ass'],
            'max_chars_per_line': 40,
            'max_lines_per_subtitle': 2,
            'min_display_time': 1.5,
            'parallel_workers': 1,
            'keep_temp_files': False,
            'enable_checkpoints': True,
            'temp_dir': './temp',
            'output_dir': './output',
            'models_dir': './data/models'
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(default_config, f, default_flow_style=False)
            else:
                json.dump(default_config, f, indent=2)
        
        print(f"Created default configuration: {output_path}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from specific file"""
        self._load_from_file(config_path)
        return self.get_dict()