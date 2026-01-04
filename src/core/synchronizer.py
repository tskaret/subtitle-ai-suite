import logging
from typing import List, Dict, Any

# Assuming SpeakerProfile and transcription segment structure are as defined previously
# For this file, we assume SpeakerProfile is an object with an 'id' and 'segment_times' (list of (start, end) tuples)
# And transcription_segments are dicts with 'start', 'end', 'text', and optionally 'words'

class Synchronizer:
    """
    Synchronizes transcription segments with speaker diarization results.
    Assigns speaker labels to transcription segments based on temporal overlap.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _calculate_temporal_overlap(self, start1: float, end1: float, start2: float, end2: float) -> float:
        """
        Calculate temporal overlap between two time intervals.
        """
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return max(0.0, overlap_end - overlap_start)

    def _assign_speaker_to_segment(self, segment: Dict[str, Any], speaker_profiles: List[Any]) -> str:
        """
        Determines the most likely speaker for a given transcription segment
        based on maximum temporal overlap with speaker diarization segments.
        """
        segment_start = segment.get('start', 0.0)
        segment_end = segment.get('end', 0.0)

        best_speaker_id: Optional[str] = None
        max_overlap_time: float = 0.0

        for speaker_profile in speaker_profiles:
            current_speaker_overlap = 0.0
            for spk_seg_start, spk_seg_end in speaker_profile.segment_times:
                overlap = self._calculate_temporal_overlap(
                    segment_start, segment_end, spk_seg_start, spk_seg_end
                )
                current_speaker_overlap += overlap
            
            if current_speaker_overlap > max_overlap_time:
                max_overlap_time = current_speaker_overlap
                best_speaker_id = speaker_profile.id
        
        if best_speaker_id:
            self.logger.debug(f"Assigned speaker {best_speaker_id} to segment from {segment_start:.2f}-{segment_end:.2f} (overlap: {max_overlap_time:.2f}s)")
        else:
            self.logger.warning(f"No speaker found for segment from {segment_start:.2f}-{segment_end:.2f}. Assigning default.")
            # Fallback: assign to a default speaker if no overlap or no speakers provided
            if speaker_profiles:
                best_speaker_id = speaker_profiles[0].id # Assign to the first speaker
            else:
                best_speaker_id = "unknown_speaker" # Or a generic label

        return best_speaker_id

    def synchronize(self, 
                    transcription_segments: List[Dict[str, Any]], 
                    speaker_profiles: List[Any]) -> List[Dict[str, Any]]:
        """
        Synchronizes transcription segments with speaker diarization results.
        Adds a 'speaker' key to each transcription segment.

        Args:
            transcription_segments (List[Dict[str, Any]]): List of transcription segments,
                                                          each with 'start', 'end', 'text' (and 'words' potentially).
            speaker_profiles (List[Any]): List of SpeakerProfile objects from speaker_analyzer.py.

        Returns:
            List[Dict[str, Any]]: Updated transcription segments, each with an assigned 'speaker' key.
        """
        if not speaker_profiles:
            self.logger.warning("No speaker profiles provided for synchronization. Segments will not have speaker labels.")
            for segment in transcription_segments:
                segment['speaker'] = "unknown_speaker"
            return transcription_segments

        self.logger.info(f"Synchronizing {len(transcription_segments)} transcription segments with {len(speaker_profiles)} speakers.")
        
        synchronized_segments = []
        for segment in transcription_segments:
            speaker_id = self._assign_speaker_to_segment(segment, speaker_profiles)
            segment['speaker'] = speaker_id
            synchronized_segments.append(segment)
            
        self.logger.info("Synchronization complete.")
        return synchronized_segments

