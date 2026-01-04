from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Any

class SrtHandler:
    """
    Handles the generation of SubRip (.srt) subtitle files.
    """

    def __init__(self):
        pass

    def _format_timestamp(self, seconds: float) -> str:
        """
        Formats a time in seconds to SRT timestamp format (HH:MM:SS,ms).
        """
        delta = timedelta(seconds=seconds)
        hours, remainder = divmod(delta.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(delta.microseconds / 1000)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    def generate_srt(self, transcription_data: List[Dict[str, Any]], output_path: Path):
        """
        Generates an SRT file from transcription data.

        Args:
            transcription_data (List[Dict[str, Any]]): A list of dictionaries,
                                                      where each dictionary represents a subtitle segment
                                                      and should contain 'start', 'end', and 'text' keys.
            output_path (Path): The path where the SRT file will be saved.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(transcription_data):
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)
                text = segment.get("text", "")

                f.write(f"{i + 1}\n")
                f.write(f"{self._format_timestamp(start_time)} --> {self._format_timestamp(end_time)}\n")
                f.write(f"{text.strip()}\n\n")
