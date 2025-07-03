import os
import torch
import numpy as np
import torchaudio
import speechbrain as sb
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class SpeakerProfile:
    """
    Comprehensive speaker profile with advanced characteristics
    """
    id: str
    confidence: float = 0.0
    total_speech_time: float = 0.0
    gender: str = 'unknown'
    age_group: str = 'unknown'
    speaking_style: str = 'unknown'
    emotion: str = 'neutral'
    language: str = 'unknown'
    accent: str = 'unknown'

class AdvancedSpeakerDiarization:
    """
    Advanced speaker diarization with multi-modal analysis
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize diarization system with optional configuration
        
        Args:
            config (Dict[str, Any], optional): Configuration settings
        """
        # Default configuration
        self.config = config or {}
        
        # Speaker recognition model
        self.speaker_recognizer = self._load_speaker_recognizer()
        
        # Emotion detection model
        self.emotion_detector = self._load_emotion_detector()
        
        # Gender detection model
        self.gender_detector = self._load_gender_detector()

    def _load_speaker_recognizer(self):
        """
        Load pre-trained speaker recognition model
        
        Returns:
            SpeechBrain speaker recognition model
        """
        try:
            recognizer = sb.pretrained.SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            return recognizer
        except Exception as e:
            print(f"Speaker recognition model loading error: {e}")
            return None

    def _load_emotion_detector(self):
        """
        Load emotion detection model
        
        Returns:
            Emotion detection model or None
        """
        try:
            # Note: This is a placeholder. Replace with actual SpeechBrain emotion model
            emotion_model = sb.pretrained.EncoderClassifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="pretrained_models/emotion-recognition"
            )
            return emotion_model
        except Exception as e:
            print(f"Emotion detection model loading error: {e}")
            return None

    def _load_gender_detector(self):
        """
        Load gender detection model
        
        Returns:
            Gender detection model or None
        """
        try:
            # Note: This is a placeholder. SpeechBrain might need a specific gender detection model
            gender_model = sb.pretrained.EncoderClassifier.from_hparams(
                source="speechbrain/gender-recognition-wav2vec2-16kHz",
                savedir="pretrained_models/gender-recognition"
            )
            return gender_model
        except Exception as e:
            print(f"Gender detection model loading error: {e}")
            return None

    def process_audio(self, audio_path: str) -> List[SpeakerProfile]:
        """
        Perform comprehensive speaker diarization
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            List of detected speaker profiles
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ensure mono and resample if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Perform speaker embeddings
            speaker_embeddings = self._extract_speaker_embeddings(waveform)
            
            # Cluster speakers
            clustered_speakers = self._cluster_speakers(speaker_embeddings)
            
            # Enrich speaker profiles
            enriched_speakers = self._enrich_speaker_profiles(waveform, clustered_speakers)
            
            return enriched_speakers
        
        except Exception as e:
            print(f"Speaker diarization error: {e}")
            return []

    def _extract_speaker_embeddings(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract speaker embeddings from audio
        
        Args:
            waveform (torch.Tensor): Input audio waveform
        
        Returns:
            List of speaker embeddings
        """
        if not self.speaker_recognizer:
            return []
        
        # Split audio into segments (e.g., 5-second windows)
        segment_length = int(5 * 16000)  # 5 seconds at 16kHz
        embeddings = []
        
        for start in range(0, waveform.shape[1], segment_length):
            segment = waveform[:, start:start+segment_length]
            if segment.shape[1] < segment_length:
                break
            
            embedding = self.speaker_recognizer.encode_batch(segment)
            embeddings.append(embedding.squeeze())
        
        return embeddings

    def _cluster_speakers(self, embeddings: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        Cluster speaker embeddings
        
        Args:
            embeddings (List[torch.Tensor]): Speaker embeddings
        
        Returns:
            Dict of clustered speakers
        """
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        if not embeddings:
            return {}
        
        # Prepare embeddings for clustering
        embeddings_array = torch.stack(embeddings).numpy()
        scaled_embeddings = StandardScaler().fit_transform(embeddings_array)
        
        # Perform clustering
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(scaled_embeddings)
        
        # Group embeddings by cluster
        clustered_speakers = {}
        for cluster_id in set(clustering.labels_):
            if cluster_id != -1:  # Ignore noise points
                cluster_mask = clustering.labels_ == cluster_id
                clustered_speakers[f'speaker_{cluster_id}'] = embeddings_array[cluster_mask]
        
        return clustered_speakers

    def _enrich_speaker_profiles(
        self, 
        waveform: torch.Tensor, 
        clustered_speakers: Dict[str, List[torch.Tensor]]
    ) -> List[SpeakerProfile]:
        """
        Enrich speaker profiles with additional characteristics
        
        Args:
            waveform (torch.Tensor): Original audio waveform
            clustered_speakers (Dict): Clustered speaker embeddings
        
        Returns:
            List of enriched speaker profiles
        """
        speaker_profiles = []
        
        for speaker_id, speaker_embeddings in clustered_speakers.items():
            # Calculate total speech time (estimated)
            total_speech_time = len(speaker_embeddings) * 5.0  # 5-second segments
            
            # Create base speaker profile
            profile = SpeakerProfile(
                id=speaker_id,
                total_speech_time=total_speech_time,
                confidence=len(speaker_embeddings) / (len(clustered_speakers) or 1)
            )
            
            # Detect gender (if model available)
            if self.gender_detector:
                try:
                    gender_result = self.gender_detector.classify_batch(torch.from_numpy(speaker_embeddings))
                    profile.gender = gender_result[0][0]
                except Exception as e:
                    print(f"Gender detection error: {e}")
            
            # Detect emotion (if model available)
            if self.emotion_detector:
                try:
                    emotion_result = self.emotion_detector.classify_batch(torch.from_numpy(speaker_embeddings))
                    profile.emotion = emotion_result[0][0]
                except Exception as e:
                    print(f"Emotion detection error: {e}")
            
            speaker_profiles.append(profile)
        
        return speaker_profiles

    def export_speaker_profiles(self, profiles: List[SpeakerProfile], output_path: str):
        """
        Export speaker profiles to JSON
        
        Args:
            profiles (List[SpeakerProfile]): Speaker profiles to export
            output_path (str): Path to export JSON file
        """
        import json
        
        try:
            with open(output_path, 'w') as f:
                json.dump([asdict(profile) for profile in profiles], f, indent=2)
            print(f"Speaker profiles exported to {output_path}")
        except Exception as e:
            print(f"Speaker profile export error: {e}")