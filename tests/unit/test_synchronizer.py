import pytest
from unittest.mock import MagicMock
from pathlib import Path
import sys

# TEMPORARY: Adjusting sys.path for direct execution during development
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.insert(0, str(project_root))

from src.core.synchronizer import Synchronizer
from src.core.speaker_analyzer import SpeakerProfile # Needed for mocking speaker profiles

@pytest.fixture
def synchronizer():
    return Synchronizer()

def test_calculate_temporal_overlap(synchronizer):
    # No overlap
    assert synchronizer._calculate_temporal_overlap(0, 1, 2, 3) == 0.0
    assert synchronizer._calculate_temporal_overlap(2, 3, 0, 1) == 0.0

    # Partial overlap (start1 in, end1 out)
    assert synchronizer._calculate_temporal_overlap(0, 2, 1, 3) == 1.0

    # Partial overlap (start2 in, end2 out)
    assert synchronizer._calculate_temporal_overlap(1, 3, 0, 2) == 1.0

    # Full overlap (interval1 contains interval2)
    assert synchronizer._calculate_temporal_overlap(0, 5, 1, 4) == 3.0

    # Full overlap (interval2 contains interval1)
    assert synchronizer._calculate_temporal_overlap(1, 4, 0, 5) == 3.0

    # Identical intervals
    assert synchronizer._calculate_temporal_overlap(0, 5, 0, 5) == 5.0

    # Touch at ends (no overlap)
    assert synchronizer._calculate_temporal_overlap(0, 1, 1, 2) == 0.0
    assert synchronizer._calculate_temporal_overlap(1, 2, 0, 1) == 0.0

def test_assign_speaker_to_segment_max_overlap(synchronizer):
    speaker_profiles = [
        SpeakerProfile(id="speaker_A", segment_times=[(0.0, 1.0), (3.0, 4.0)]),
        SpeakerProfile(id="speaker_B", segment_times=[(1.5, 2.5), (4.5, 5.5)]),
    ]
    segment = {'start': 0.8, 'end': 1.8, 'text': 'text'} # Overlaps most with speaker A (0.2s) and B (0.3s)
    
    # Corrected expectation: should overlap more with B. (1.8-1.5 = 0.3s for B, 1.0-0.8 = 0.2s for A)
    # The current code calculates cumulative overlap
    
    # For segment 0.8-1.8:
    # Speaker A: (0.0, 1.0) -> overlap(0.8, 1.0) = 0.2
    # Speaker B: (1.5, 2.5) -> overlap(1.5, 1.8) = 0.3
    # So Speaker B should be chosen

    assigned_speaker = synchronizer._assign_speaker_to_segment(segment, speaker_profiles)
    assert assigned_speaker == "speaker_B"

def test_assign_speaker_to_segment_no_overlap_fallback(synchronizer):
    speaker_profiles = [
        SpeakerProfile(id="speaker_A", segment_times=[(0.0, 1.0)]),
    ]
    segment = {'start': 5.0, 'end': 6.0, 'text': 'text'} # No overlap
    
    assigned_speaker = synchronizer._assign_speaker_to_segment(segment, speaker_profiles)
    assert assigned_speaker == "speaker_A" # Fallback to first speaker

def test_assign_speaker_to_segment_no_speakers(synchronizer):
    segment = {'start': 0.0, 'end': 1.0, 'text': 'text'}
    assigned_speaker = synchronizer._assign_speaker_to_segment(segment, [])
    assert assigned_speaker == "unknown_speaker"

def test_synchronize_with_speakers(synchronizer):
    transcription_segments = [
        {'start': 0.1, 'end': 1.1, 'text': 'Hi there.'},
        {'start': 1.6, 'end': 2.6, 'text': 'How are you?'},
        {'start': 3.1, 'end': 4.1, 'text': 'I am fine.'},
    ]
    speaker_profiles = [
        SpeakerProfile(id="speaker_1", segment_times=[(0.0, 1.5), (3.0, 4.5)]), # Speaker 1
        SpeakerProfile(id="speaker_2", segment_times=[(1.5, 3.0)]),             # Speaker 2
    ]

    synchronized_segments = synchronizer.synchronize(transcription_segments, speaker_profiles)

    assert len(synchronized_segments) == 3
    assert synchronized_segments[0]['speaker'] == "speaker_1"
    assert synchronized_segments[1]['speaker'] == "speaker_2"
    assert synchronized_segments[2]['speaker'] == "speaker_1"

def test_synchronize_no_speakers(synchronizer):
    transcription_segments = [
        {'start': 0.1, 'end': 1.1, 'text': 'Hi there.'},
        {'start': 1.6, 'end': 2.6, 'text': 'How are you?'},
    ]

    synchronized_segments = synchronizer.synchronize(transcription_segments, [])

    assert len(synchronized_segments) == 2
    assert synchronized_segments[0]['speaker'] == "unknown_speaker"
    assert synchronized_segments[1]['speaker'] == "unknown_speaker"
