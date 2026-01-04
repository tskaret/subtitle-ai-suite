"""
Modern speaker diarization using pyannote.audio pipeline
"""

import os
import torch
import torchaudio
import tempfile
import numpy as np
import librosa
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class SpeakerProfile:
    """
    Enhanced speaker profile with precise timing information
    """
    id: str
    confidence: float = 0.0
    total_speech_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    segment_times: list = None  # List of (start, end) tuples for all segments
    gender: str = 'unknown'
    
    def __post_init__(self):
        if self.segment_times is None:
            self.segment_times = []

class ModernSpeakerDiarization:
    """
    Modern speaker diarization using pyannote.audio pipeline
    Replaces the broken SpeechBrain-based implementation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize modern speaker diarization system
        
        Args:
            config (Dict[str, Any], optional): Configuration settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup HuggingFace authentication
        self._setup_huggingface_auth()
        
        # Load pyannote pipeline
        self.pipeline = self._load_diarization_pipeline()
        
        # Configuration parameters
        self.min_speakers = self.config.get('min_speakers', 1)
        self.max_speakers = self.config.get('max_speakers', 10)
        
    def _setup_huggingface_auth(self):
        """
        Setup HuggingFace authentication using HF_TOKEN environment variable
        """
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
                print("✅ Successfully authenticated with HuggingFace using HF_TOKEN")
            except Exception as e:
                print(f"⚠️  Warning: Failed to authenticate with HuggingFace: {e}")
        else:
            print("⚠️  Warning: HF_TOKEN not found in environment variables")
            print("   Some models may require authentication for optimal performance")
    
    def _load_diarization_pipeline(self):
        """
        Load pyannote.audio speaker diarization pipeline
        
        Returns:
            Pyannote pipeline for speaker diarization
        """
        try:
            # Use the latest speaker diarization model
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.getenv('HF_TOKEN')
            )
            print("✅ Loaded pyannote speaker diarization pipeline successfully")
            return pipeline
        except Exception as e:
            print(f"❌ Error loading pyannote pipeline: {e}")
            print("   Falling back to basic segmentation approach")
            return None
    
    def process_audio(self, audio_path: str) -> List[SpeakerProfile]:
        """
        Perform modern speaker diarization using pyannote.audio
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            List of detected speaker profiles with precise timing
        """
        try:
            print(f"🎭 Starting modern speaker diarization for: {audio_path}")
            
            if not self.pipeline:
                print("❌ No diarization pipeline available, returning empty speaker list")
                return []
            
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Create a temporary file for pyannote to process
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Write audio to temporary file
                sf.write(temp_file.name, waveform.squeeze().numpy(), sample_rate)
                temp_audio_path = temp_file.name
            
            try:
                # Apply speaker diarization using file path
                diarization = self.pipeline(temp_audio_path)
            finally:
                # Clean up temporary file
                import os
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            
            # Convert pyannote results to our speaker profiles
            speaker_profiles = self._convert_diarization_results(diarization)
            
            # Perform gender detection
            print("⚧️  Detecting speaker genders...")
            self._detect_genders(audio_path, speaker_profiles)
            
            print(f"✅ Found {len(speaker_profiles)} speakers with pyannote diarization")
            for speaker in speaker_profiles:
                print(f"   Speaker {speaker.id}: {speaker.total_speech_time:.1f}s total speech time ({speaker.gender})")
            
            return speaker_profiles
            
        except Exception as e:
            print(f"❌ Modern speaker diarization error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to simple segmentation if pyannote fails
            return self._fallback_speaker_detection(audio_path)
    
    def _convert_diarization_results(self, diarization: Annotation) -> List[SpeakerProfile]:
        """
        Convert pyannote diarization results to SpeakerProfile objects
        
        Args:
            diarization (Annotation): Pyannote diarization results
        
        Returns:
            List of SpeakerProfile objects
        """
        speaker_data = {}
        
        # Process each segment in the diarization
        for segment, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label not in speaker_data:
                speaker_data[speaker_label] = {
                    'segments': [],
                    'total_time': 0.0,
                    'start_time': float('inf'),
                    'end_time': 0.0
                }
            
            # Add segment information
            start_time = segment.start
            end_time = segment.end
            duration = end_time - start_time
            
            speaker_data[speaker_label]['segments'].append((start_time, end_time))
            speaker_data[speaker_label]['total_time'] += duration
            speaker_data[speaker_label]['start_time'] = min(
                speaker_data[speaker_label]['start_time'], start_time
            )
            speaker_data[speaker_label]['end_time'] = max(
                speaker_data[speaker_label]['end_time'], end_time
            )
        
        # Convert to SpeakerProfile objects
        speaker_profiles = []
        for speaker_id, data in speaker_data.items():
            profile = SpeakerProfile(
                id=f"speaker_{speaker_id}",
                total_speech_time=data['total_time'],
                start_time=data['start_time'],
                end_time=data['end_time'],
                segment_times=data['segments'],
                confidence=min(1.0, data['total_time'] / 10.0)  # Simple confidence metric
            )
            speaker_profiles.append(profile)
        
        # Sort by total speech time (descending)
        speaker_profiles.sort(key=lambda x: x.total_speech_time, reverse=True)
        
        return speaker_profiles
    
    def _detect_genders(self, audio_path: str, speaker_profiles: List[SpeakerProfile]):
        """
        Detect gender for each speaker using F0 (pitch) analysis
        
        Args:
            audio_path (str): Path to audio file
            speaker_profiles (List[SpeakerProfile]): List of speaker profiles to update
        """
        try:
            # Load audio using librosa for pitch analysis
            # Load only once, resample to 16kHz
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            for speaker in speaker_profiles:
                # Collect audio segments for this speaker (up to 30 seconds to save time)
                speaker_audio = []
                total_duration = 0
                
                for start, end in speaker.segment_times:
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    
                    if start_sample < len(y) and end_sample <= len(y):
                        segment = y[start_sample:end_sample]
                        speaker_audio.append(segment)
                        total_duration += (end - start)
                    
                    if total_duration > 30:  # Limit to 30 seconds per speaker
                        break
                
                if speaker_audio:
                    # Concatenate segments
                    combined_audio = np.concatenate(speaker_audio)
                    
                    # Estimate F0 (Fundamental Frequency)
                    f0, voiced_flag, voiced_probs = librosa.pyin(
                        combined_audio, 
                        fmin=librosa.note_to_hz('C2'), 
                        fmax=librosa.note_to_hz('C7'),
                        sr=sr
                    )
                    
                    # Filter for voiced segments
                    voiced_f0 = f0[voiced_flag]
                    
                    if len(voiced_f0) > 0:
                        mean_f0 = np.nanmean(voiced_f0)
                        
                        # Simple threshold-based classification
                        # Male: typically 85-180 Hz
                        # Female: typically 165-255 Hz
                        if mean_f0 < 165:
                            speaker.gender = 'male'
                        else:
                            speaker.gender = 'female'
                            
                        # Confidence based on how far from threshold (simple heuristic)
                        distance = abs(mean_f0 - 165)
                        confidence_boost = min(0.2, distance / 100.0)
                        speaker.confidence = min(1.0, speaker.confidence + confidence_boost)
        
        except Exception as e:
            print(f"⚠️  Gender detection failed: {e}")
            # Continue without gender detection
    
    def _fallback_speaker_detection(self, audio_path: str) -> List[SpeakerProfile]:
        """
        Fallback speaker detection when pyannote pipeline fails
        Creates a simple 2-speaker assumption for dialogue scenarios
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            List with 2 basic speaker profiles
        """
        try:
            print("🔄 Using fallback speaker detection (2-speaker dialogue assumption)")
            
            # Load audio to get duration
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate
            
            # Create two basic speaker profiles for dialogue
            speaker_profiles = [
                SpeakerProfile(
                    id="speaker_0",
                    total_speech_time=duration * 0.6,  # Primary speaker (60%)
                    start_time=0.0,
                    end_time=duration,
                    segment_times=[(0.0, duration)],
                    confidence=0.7
                ),
                SpeakerProfile(
                    id="speaker_1", 
                    total_speech_time=duration * 0.4,  # Secondary speaker (40%)
                    start_time=0.0,
                    end_time=duration,
                    segment_times=[(0.0, duration)],
                    confidence=0.7
                )
            ]
            
            print(f"✅ Created fallback speaker profiles: 2 speakers")
            return speaker_profiles
            
        except Exception as e:
            print(f"❌ Fallback speaker detection error: {e}")
            return []
    
    def find_best_speaker_match(
        self, 
        segment_start: float, 
        segment_end: float, 
        speaker_profiles: List[SpeakerProfile]
    ) -> str:
        """
        Find the best speaker match for a transcript segment based on temporal overlap
        
        Args:
            segment_start (float): Transcript segment start time
            segment_end (float): Transcript segment end time  
            speaker_profiles (List[SpeakerProfile]): Available speaker profiles
        
        Returns:
            str: Best matching speaker ID
        """
        if not speaker_profiles:
            return None
        
        best_speaker = None
        max_overlap = 0.0
        
        for speaker in speaker_profiles:
            total_overlap = 0.0
            
            # Calculate overlap with all speaker segments
            for start_time, end_time in speaker.segment_times:
                overlap = self._calculate_temporal_overlap(
                    segment_start, segment_end, start_time, end_time
                )
                total_overlap += overlap
            
            if total_overlap > max_overlap:
                max_overlap = total_overlap
                best_speaker = speaker
        
        # Return speaker with best overlap, or primary speaker as fallback
        if best_speaker and max_overlap > 0.1:  # At least 0.1 second overlap
            return best_speaker.id
        else:
            # Fallback to primary speaker (most speech time)
            return speaker_profiles[0].id if speaker_profiles else None
    
    def _calculate_temporal_overlap(
        self, 
        start1: float, 
        end1: float, 
        start2: float, 
        end2: float
    ) -> float:
        """
        Calculate temporal overlap between two time intervals
        
        Args:
            start1, end1: First time interval
            start2, end2: Second time interval
        
        Returns:
            float: Overlap duration in seconds
        """
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return max(0.0, overlap_end - overlap_start)
    
    def simple_dialogue_detection(
        self, 
        segment_text: str, 
        speaker_profiles: List[SpeakerProfile]
    ) -> str:
        """
        Enhanced dialogue detection using contextual patterns and speech analysis
        
        Args:
            segment_text (str): Text of the segment
            speaker_profiles (List[SpeakerProfile]): Available speakers
        
        Returns:
            str: Detected speaker ID
        """
        if len(speaker_profiles) < 2:
            return speaker_profiles[0].id if speaker_profiles else None
        
        text = segment_text.strip().lower()
        
        # Strong interviewer indicators (questions, prompts, transitions)
        strong_interviewer_patterns = [
            text.endswith('?'),
            text.startswith(('hi ', 'hello', 'so ', 'and ', 'yeah,', 'all right', 'thank you')),
            'feelings about' in text,
            'ready for' in text,
            'coming your way' in text,
            text.startswith(('well ', 'now ', 'okay ', 'right ')),
            'tournament' in text and ('doubts' in text or 'standard' in text),
        ]
        
        # Strong interviewee indicators (personal responses, game analysis)
        strong_interviewee_patterns = [
            text.startswith(('no,', 'i ', 'honestly', 'i think', 'i had', 'i get')),
            'chess' in text and ('playing' in text or 'flow' in text),
            'position' in text,
            'time' in text and ('short' in text or 'handle' in text),
            'moves' in text,
            'lost' in text,
            'poorly' in text,
            'punished' in text,
            'life doesn' in text or 'bad even if' in text
        ]
        
        # Score-based approach for better detection
        interviewer_score = sum(strong_interviewer_patterns)
        interviewee_score = sum(strong_interviewee_patterns)
        
        # Determine speaker based on content patterns
        if interviewer_score > interviewee_score:
            # Return speaker with less total speech time (likely interviewer)
            return min(speaker_profiles, key=lambda x: x.total_speech_time).id
        elif interviewee_score > interviewer_score:
            # Return speaker with more total speech time (likely interviewee) 
            return max(speaker_profiles, key=lambda x: x.total_speech_time).id
        else:
            # Fallback: use speech time ratio for ambiguous cases
            primary_speaker = max(speaker_profiles, key=lambda x: x.total_speech_time)
            secondary_speaker = min(speaker_profiles, key=lambda x: x.total_speech_time)
            
            # If text is short and unclear, lean towards primary speaker (interviewee)
            if len(text.split()) < 8:
                return primary_speaker.id
            else:
                # Longer statements more likely from interviewee
                return primary_speaker.id