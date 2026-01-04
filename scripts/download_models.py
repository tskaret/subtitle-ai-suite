#!/usr/bin/env python3
"""
Model Download Script for Subtitle AI Suite
Downloads and sets up required AI models
"""

import os
import sys
import urllib.request
import subprocess
from pathlib import Path

def ensure_directory(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def download_whisper_models():
    """Download Whisper models"""
    print("📥 Downloading Whisper models...")
    
    # Whisper models will be downloaded automatically on first use
    # We just need to ensure the directory exists
    models_dir = Path("data/models/whisper")
    ensure_directory(models_dir)
    
    # Test download small model to verify setup
    try:
        import whisper
        print("  ✓ Testing Whisper installation...")
        model = whisper.load_model("tiny")
        print("  ✓ Whisper models ready")
        return True
    except Exception as e:
        print(f"  ❌ Error downloading Whisper models: {e}")
        return False

def download_speechbrain_models():
    """Download SpeechBrain models"""
    print("📥 Downloading SpeechBrain models...")
    
    models_dir = Path("data/models/speechbrain")
    ensure_directory(models_dir)
    
    try:
        # Test SpeechBrain model download
        import speechbrain as sb
        print("  ✓ Testing SpeechBrain installation...")
        
        # This will download the model if not present
        # We'll catch any errors but not fail completely
        try:
            recognizer = sb.pretrained.SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="data/models/speechbrain/spkrec-ecapa-voxceleb"
            )
            print("  ✓ SpeechBrain models ready")
        except Exception as model_error:
            print(f"  ⚠️  SpeechBrain model download will happen on first use: {model_error}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error setting up SpeechBrain: {e}")
        return False

def check_pyannote_setup():
    """Check Pyannote setup"""
    print("📥 Checking Pyannote setup...")
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("  ⚠️  HF_TOKEN environment variable not found.")
        print("     Pyannote models require a Hugging Face token.")
        print("     Please set HF_TOKEN in your .env file.")
        print("     Models: pyannote/speaker-diarization-3.1")
        return True # Don't fail, just warn
        
    print("  ✓ HF_TOKEN found")
    return True

def verify_system_dependencies():
    """Verify system dependencies are installed"""
    print("🔍 Verifying system dependencies...")
    
    dependencies = {
        'ffmpeg': ['ffmpeg', '-version'],
        'git': ['git', '--version']
    }
    
    missing_deps = []
    
    for name, command in dependencies.items():
        try:
            result = subprocess.run(
                command, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                check=True
            )
            print(f"  ✓ {name} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ❌ {name} is not installed")
            missing_deps.append(name)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them before proceeding:")
        print("  - ffmpeg: https://ffmpeg.org/download.html")
        print("  - git: https://git-scm.com/downloads")
        return False
    
    return True

def create_config_files():
    """Create necessary configuration files"""
    print("📝 Creating configuration files...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print("  ✓ Created .env file from template")
        else:
            # Create basic .env file
            with open(env_file, 'w') as f:
                f.write("""# API Keys (optional)
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Processing Settings
DEFAULT_WHISPER_MODEL=large-v2
TEMP_DIR=./temp
OUTPUT_DIR=./output
MAX_CONCURRENT_JOBS=2

# Feature Flags
ENABLE_GPU=true
ENABLE_SPEAKER_DIARIZATION=true
""")
            print("  ✓ Created basic .env file")
    else:
        print("  ✓ .env file already exists")

def main():
    """Main function"""
    print("🚀 Subtitle AI Suite - Model Download Script")
    print("=" * 50)
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    success = True
    
    # Verify system dependencies
    if not verify_system_dependencies():
        success = False
    
    # Create configuration files
    create_config_files()
    
    # Download models
    if not download_whisper_models():
        success = False
    
    if not download_speechbrain_models():
        success = False

    if not check_pyannote_setup():
        success = False
    
    if success:
        print("\n✨ Model setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and update .env file with your API keys")
        print("2. Run: python src/main.py --help")
        print("3. Test with: python src/main.py path/to/your/audio.wav")
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("Please resolve the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()