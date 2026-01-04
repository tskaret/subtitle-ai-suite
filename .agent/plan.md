# Plan for subtitle-ai-suite Implementation

This plan outlines the steps to implement the full functionality of the `subtitle-ai-suite` project, prioritizing core features first, followed by advanced capabilities, and concluding with comprehensive testing and documentation.

## Overall Strategy:
1.  **Prioritize Core Functionality:** Start with the most fundamental components that enable the basic operation of the suite (e.g., input handling, audio extraction, basic transcription, and subtitle output).
2.  **Iterative Development:** Implement features in logical blocks, ensuring each block is functional and tested before moving to the next.
3.  **Adhere to `gameplan.md` Structure:** Follow the recommended project structure and dependency list as much as possible.
4.  **Testing:** For each implemented feature, add basic unit/integration tests to verify functionality.
5.  **Commit Frequently:** Commit and push after every significant file edit.
6.  **Use `.agent/`:** Keep track of progress and detailed sub-tasks in this `plan.md` file and other scratchpad files within the `.agent/` directory.

---

## Phase 1: Foundation (Core Pipeline)

*   **Sub-task 1.1: Environment Setup & Dependencies**
    *   Review `requirements.txt` and `setup.py` from `gameplan.md`.
    *   Ensure all necessary dependencies are listed and installable.
    *   Check for `ffmpeg` installation and provide guidance if missing.
    *   *Status: Completed*

*   **Sub-task 1.2: Project Structure Alignment**
    *   Compare the current project structure with the "Recommended Project Structure" in `gameplan.md`.
    *   Create any missing directories and placeholder `__init__.py` files as per the recommended structure.
    *   *Status: Completed*

*   **Sub-task 1.3: Basic Input Handling (`src/core/input_handler.py`)**
    *   Implement basic functionality for downloading YouTube videos using `pytube` and `yt-dlp`.
    *   Implement functionality for reading local media files.
    *   *Status: Completed*

*   **Sub-task 1.4: Audio Extraction (`src/core/audio_processor.py`)**
    *   Implement high-quality audio extraction from video using `moviepy` and `ffmpeg-python`.
    *   Include noise reduction using `noisereduce`.
    *   *Status: Completed*

*   **Sub-task 1.5: Basic Transcription (`src/core/transcription.py`)**
    *   Integrate `whisper` for speech-to-text transcription.
    *   Implement word-level timestamp extraction.
    *   *Status: Completed*

*   **Sub-task 1.6: Basic Subtitle Output (`src/formats/srt_handler.py`, `src/formats/ass_handler.py`)**
    *   Generate basic SRT and ASS files from transcription data.
    *   *Status: Completed*

*   **Sub-task 1.7: Core CLI Integration (`src/interfaces/cli/main_cli.py`)**
    *   Expose basic functionality through a command-line interface.
    *   *Status: Completed*

---

## Phase 2: Speaker Diarization & Colorization

*   **Sub-task 2.1: Speaker Diarization (`src/core/speaker_analyzer.py`)**
    *   Integrate `speechbrain` for speaker diarization.
    *   *Status: Completed*

*   **Sub-task 2.2: Synchronization (`src/core/synchronizer.py`)**
    *   Implement multi-source timestamp validation and synchronization with VAD.
    *   *Status: Completed*

*   **Sub-task 2.3: Speaker Colorization Logic**
    *   Implement logic to analyze speaker distribution and assign colors based on the 95% threshold.
    *   Update `src/formats/ass_handler.py` to apply speaker colorization.
    *   *Status: Completed*

*   **Sub-task 2.4: CLI Enhancements**
    *   Add CLI arguments for enabling colorization and setting thresholds.
    *   *Status: Completed*

---

## Phase 3: Advanced Features & Refinement

*   **Sub-task 3.1: Batch Processing (`src/processing/batch_processor.py`)**
    *   Implement directory and playlist processing.
    *   Add resume functionality.
    *   *Status: Completed*

*   **Sub-task 3.2: Content Enhancement (emotion detection, chapter generation, etc.)**
    *   Integrate `src/ai/emotion_detector.py` and other AI features as described in `gameplan.md`.
    *   *Status: Completed*

*   **Sub-task 3.3: Multilingual Support & Translation (`src/core/translator.py`)**
    *   Implement translation capabilities using local (Ollama) and online (OpenAI GPT, Hugging Face) options.
    *   *Status: Pending*

*   **Sub-task 3.4: GUI Implementation (`src/interfaces/gui/main_window.py`)**
    *   Develop the basic GUI using `tkinter` or `customtkinter`.
    *   *Status: Pending*

---

## Phase 4: Testing & Documentation

*   **Sub-task 4.1: Unit Tests**
    *   Write unit tests for core modules (e.g., `audio_processor`, `transcription`, `speaker_analyzer`).
    *   *Status: Pending*

*   **Sub-task 4.2: Integration Tests**
    *   Write integration tests for the full pipeline, batch processing, and CLI.
    *   *Status: Pending*

*   **Sub-task 4.3: End-to-End Tests**
    *   Verify main application flows from input to final output.
    *   *Status: Pending*

*   **Sub-task 4.4: Update Documentation**
    *   Update `README.md` and create/update relevant documentation files to reflect implemented features, usage, and installation instructions.
    *   *Status: Pending*