import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys
from pathlib import Path
import argparse # Needed to create Namespace for SubtitlePipelineManager
import logging

# TEMPORARY: Adjusting sys.path for direct execution during development
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
sys.path.insert(0, str(project_root))

from src.processing.pipeline_manager import SubtitlePipelineManager
from src.subtitle_suite.utils.logger import setup_logging


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Subtitle AI Suite")
        self.geometry("800x600")
        self.logger = setup_logging()

        self._create_widgets()

    def _create_widgets(self):
        # Input Frame
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(input_frame, text="Input (URL or Local File):").pack(padx=10, pady=5, anchor="w")
        self.input_entry = ctk.CTkEntry(input_frame, placeholder_text="Enter YouTube URL or local file path", width=500)
        self.input_entry.pack(padx=10, pady=5, fill="x", expand=True)

        browse_button = ctk.CTkButton(input_frame, text="Browse", command=self._browse_file)
        browse_button.pack(padx=10, pady=5, anchor="e")

        # Output Directory
        output_frame = ctk.CTkFrame(self)
        output_frame.pack(padx=10, pady=10, fill="x")
        ctk.CTkLabel(output_frame, text="Output Directory:").pack(padx=10, pady=5, anchor="w")
        self.output_dir_entry = ctk.CTkEntry(output_frame, placeholder_text="Select output directory", width=500)
        self.output_dir_entry.pack(padx=10, pady=5, fill="x", expand=True)
        self.output_dir_entry.insert(0, "./output") # Default value

        output_browse_button = ctk.CTkButton(output_frame, text="Browse", command=self._browse_output_dir)
        output_browse_button.pack(padx=10, pady=5, anchor="e")

        # Options Frame
        options_frame = ctk.CTkFrame(self)
        options_frame.pack(padx=10, pady=10, fill="x")

        # Whisper Model
        ctk.CTkLabel(options_frame, text="Whisper Model:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.whisper_model_option = ctk.CTkOptionMenu(options_frame, values=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
        self.whisper_model_option.set("large-v2")
        self.whisper_model_option.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        # Language
        ctk.CTkLabel(options_frame, text="Language (e.g., en, es):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.language_entry = ctk.CTkEntry(options_frame, placeholder_text="Auto-detect if empty")
        self.language_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Translate to
        ctk.CTkLabel(options_frame, text="Translate to (e.g., fr, de):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.translate_entry = ctk.CTkEntry(options_frame, placeholder_text="Leave empty for no translation")
        self.translate_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Colorize
        self.colorize_checkbox = ctk.CTkCheckBox(options_frame, text="Enable Speaker Colorization")
        self.colorize_checkbox.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Emotion Detection
        self.emotion_detection_checkbox = ctk.CTkCheckBox(options_frame, text="Enable Emotion Detection")
        self.emotion_detection_checkbox.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Process Button
        self.process_button = ctk.CTkButton(self, text="Start Processing", command=self._start_processing)
        self.process_button.pack(padx=10, pady=10, fill="x")

        # Status Output
        self.status_text = ctk.CTkTextbox(self, height=150)
        self.status_text.pack(padx=10, pady=10, fill="both", expand=True)
        self.status_text.insert("end", "Welcome to Subtitle AI Suite!\n")
        self.status_text.configure(state="disabled")

    def _browse_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, file_path)

    def _browse_output_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir_entry.delete(0, "end")
            self.output_dir_entry.insert(0, dir_path)

    def _log_to_status(self, message):
        self.status_text.configure(state="normal")
        self.status_text.insert("end", message + "\n")
        self.status_text.see("end")
        self.status_text.configure(state="disabled")

    def _start_processing(self):
        input_source = self.input_entry.get()
        output_dir = self.output_dir_entry.get()

        if not input_source:
            messagebox.showerror("Input Error", "Please provide an input URL or local file path.")
            return
        if not output_dir:
            messagebox.showerror("Output Error", "Please select an output directory.")
            return
        
        # Disable button during processing
        self.process_button.configure(state="disabled")
        self._log_to_status("Starting processing...")
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.configure(state="disabled")


        # Create mock args for SubtitlePipelineManager
        args = argparse.Namespace(
            input=input_source,
            output_dir=output_dir,
            whisper_model=self.whisper_model_option.get(),
            language=self.language_entry.get() if self.language_entry.get() else None,
            translate=self.translate_entry.get() if self.translate_entry.get() else None,
            colorize=self.colorize_checkbox.get(),
            enable_emotion_detection=self.emotion_detection_checkbox.get(), # Pass emotion detection flag
            speaker_threshold=0.95, # Hardcoded for now
            device="auto", # Hardcoded for now
            temp_dir="./temp", # Hardcoded for now
            keep_temp=False, # Hardcoded for now
            prefix=None, # Hardcoded for now
            format=["srt", "ass"], # Hardcoded for now
        )

        # Run processing in a separate thread to keep GUI responsive
        processing_thread = threading.Thread(target=self._run_pipeline, args=(args,))
        processing_thread.start()

    def _run_pipeline(self, args):
        try:
            pipeline_manager = SubtitlePipelineManager(args)
            result = pipeline_manager.process(args.input)

            if result["success"]:
                self._log_to_status(f"✅ Processing completed successfully for {args.input}")
                self._log_to_status(f"Output files: {', '.join(result.get('generated_files', []))}")
            else:
                self._log_to_status(f"❌ Processing failed for {args.input}: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            self.logger.error(f"Error during GUI pipeline execution: {e}", exc_info=True)
            self._log_to_status(f"❌ An unexpected error occurred: {e}")
        finally:
            self.process_button.configure(state="normal") # Re-enable button

def main():
    app = MainWindow()
    app.mainloop()

if __name__ == '__main__':
    main()
