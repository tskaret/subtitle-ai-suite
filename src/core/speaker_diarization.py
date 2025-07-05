import os
import torch
import numpy as np
import torchaudio
import speechbrain as sb
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from huggingface_hub import login

@dataclass
class SpeakerProfile:
    """
    Comprehensive speaker profile with advanced characteristics
    """
    id: str
    confidence: float = 0.0
    total_speech_time: float = 0.0
    start_time: float = 0.0  # Add timing information
    end_time: float = 0.0    # Add timing information
    segment_times: list = None  # List of (start, end) tuples for all segments
    gender: str = 'unknown'
    age_group: str = 'unknown'
    speaking_style: str = 'unknown'
    emotion: str = 'neutral'
    language: str = 'unknown'
    accent: str = 'unknown'
    
    def __post_init__(self):
        if self.segment_times is None:
            self.segment_times = []

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
        
        # Configure HuggingFace authentication
        self._setup_huggingface_auth()
        
        # Speaker recognition model
        self.speaker_recognizer = self._load_speaker_recognizer()
        
        # Emotion detection model
        self.emotion_detector = self._load_emotion_detector()
        
        # Gender detection model
        self.gender_detector = self._load_gender_detector()

    def _setup_huggingface_auth(self):
        """
        Setup HuggingFace authentication using HF_TOKEN environment variable
        """
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            try:
                login(token=hf_token, add_to_git_credential=False)
                print("Successfully authenticated with HuggingFace using HF_TOKEN")
            except Exception as e:
                print(f"Warning: Failed to authenticate with HuggingFace: {e}")
        else:
            print("Warning: HF_TOKEN not found in environment variables")

    def _load_speaker_recognizer(self):
        """
        Load pre-trained speaker recognition model
        
        Returns:
            SpeechBrain speaker recognition model
        """
        try:
            # Use HuggingFace cache directly to avoid symlink issues in WSL
            import tempfile
            temp_dir = tempfile.mkdtemp()
            recognizer = sb.pretrained.SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=temp_dir,
                use_auth_token=os.getenv('HF_TOKEN')
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
            # Use a working emotion recognition model
            import tempfile
            temp_dir = tempfile.mkdtemp()
            emotion_model = sb.pretrained.EncoderClassifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir=temp_dir,
                use_auth_token=os.getenv('HF_TOKEN')
            )
            return emotion_model
        except Exception as e:
            print(f"Emotion detection model loading error: {e}")
            print("Skipping emotion detection - continuing without this feature")
            return None

    def _load_gender_detector(self):
        """
        Load gender detection model
        
        Returns:
            Gender detection model or None
        """
        try:
            # This model appears to be unavailable, skip for now
            print("Gender detection model not available - skipping")
            return None
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
            print(f"Starting speaker diarization for: {audio_path}")
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
            print(f"Extracted {len(speaker_embeddings)} speaker embeddings")
            
            # Cluster speakers
            clustered_speakers = self._cluster_speakers(speaker_embeddings)
            print(f"Found {len(clustered_speakers)} clustered speakers: {list(clustered_speakers.keys())}")
            
            # Enrich speaker profiles
            enriched_speakers = self._enrich_speaker_profiles(waveform, clustered_speakers)
            
            return enriched_speakers
        
        except Exception as e:
            print(f"Speaker diarization error: {e}")
            import traceback
            traceback.print_exc()
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

    def _cluster_speakers(self, embeddings: List[torch.Tensor]) -> Dict[str, Dict]:
        """
        Cluster speaker embeddings with better parameters for dialogue
        
        Args:
            embeddings (List[torch.Tensor]): Speaker embeddings
        
        Returns:
            Dict of clustered speakers with timing information
        """
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not embeddings:
            return {}
        
        # Prepare embeddings for clustering
        embeddings_array = torch.stack(embeddings).numpy()
        scaled_embeddings = StandardScaler().fit_transform(embeddings_array)
        
        # Try multiple clustering approaches
        clustered_speakers = {}
        
        # Method 1: DBSCAN with better parameters
        dbscan = DBSCAN(eps=0.8, min_samples=2).fit(scaled_embeddings)
        n_clusters_dbscan = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        
        # Method 2: K-means for dialogue (assume 2 speakers)
        if n_clusters_dbscan > 10 or n_clusters_dbscan < 2:
            print(f"DBSCAN found {n_clusters_dbscan} clusters, using K-means with k=2 for dialogue")
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaled_embeddings)
            clustering_labels = kmeans.labels_
        else:
            print(f"DBSCAN found {n_clusters_dbscan} clusters - using DBSCAN results")
            clustering_labels = dbscan.labels_
        
        # Group embeddings by cluster with timing information
        for cluster_id in set(clustering_labels):
            if cluster_id != -1:  # Ignore noise points
                cluster_mask = clustering_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                # Calculate timing information for this cluster
                start_time = cluster_indices[0] * 5.0  # First segment start time
                end_time = (cluster_indices[-1] + 1) * 5.0  # Last segment end time
                
                # Create segment times list
                segment_times = [(idx * 5.0, (idx + 1) * 5.0) for idx in cluster_indices]
                
                clustered_speakers[f'speaker_{cluster_id}'] = {
                    'embeddings': embeddings_array[cluster_mask],
                    'start_time': start_time,
                    'end_time': end_time,
                    'segment_times': segment_times,
                    'indices': cluster_indices
                }
        
        return clustered_speakers

    def _enrich_speaker_profiles(
        self, 
        waveform: torch.Tensor, 
        clustered_speakers: Dict[str, Dict]
    ) -> List[SpeakerProfile]:
        """
        Enrich speaker profiles with additional characteristics
        
        Args:
            waveform (torch.Tensor): Original audio waveform
            clustered_speakers (Dict): Clustered speaker data with timing info
        
        Returns:
            List of enriched speaker profiles
        """
        speaker_profiles = []
        
        for speaker_id, speaker_data in clustered_speakers.items():
            speaker_embeddings = speaker_data['embeddings']
            
            # Calculate total speech time (estimated)
            total_speech_time = len(speaker_embeddings) * 5.0  # 5-second segments
            
            # Create base speaker profile with timing information
            profile = SpeakerProfile(
                id=speaker_id,
                total_speech_time=total_speech_time,
                start_time=speaker_data['start_time'],
                end_time=speaker_data['end_time'],
                segment_times=speaker_data['segment_times'],
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