# Plan

## 1. Analysis & Verification
- [x] Read `src/core/speaker_diarization.py` and `src/core/modern_speaker_diarization.py` to understand the difference.
- [x] Run existing tests to verify current state.
- [x] Check `requirements.txt` vs code imports.

## 2. Implementation: Advanced Audio Processing
- [ ] Create `src/core/audio_processor.py` for advanced audio handling.
    - [ ] High-quality extraction (48kHz).
    - [ ] Noise reduction using `noisereduce`.
    - [ ] Normalization.
    - [ ] Voice Activity Detection (VAD) integration (maybe later, or integrated here).
- [ ] Integrate `AudioProcessor` into `EnhancedSubtitleProcessor` in `src/core/subtitle_processor.py`.
- [ ] Add tests for `AudioProcessor`.

## 3. Fix Known Issues
- [ ] Add `torchvision` to `requirements.txt` if needed (It is installed but maybe not in requirements).
- [ ] Investigate and fix SpeechBrain permissions in WSL (Low priority as we use pyannote).

## 4. Final Polish
- [ ] Verify all tests pass.
- [ ] Update documentation.
