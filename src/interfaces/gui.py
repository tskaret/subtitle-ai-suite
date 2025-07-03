"""
Basic GUI interface for Subtitle AI Suite
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.subtitle_processor import EnhancedSubtitleProcessor
from core.batch_processor import BatchProcessor
from utils.device_manager import DeviceManager
from utils.config_manager import ConfigManager
from utils.logger import setup_logging

class SubtitleGUI:
    """Basic GUI interface for subtitle processing"""
    
    def __init__(self):
        """Initialize the GUI application"""
        self.root = tk.Tk()
        self.root.title("Subtitle AI Suite")
        self.root.geometry("800x600")
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.logger = setup_logging('gui')
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
        # Create GUI elements
        self.create_widgets()
        self.setup_layout()
        
        # Start queue processor
        self.root.after(100, self.process_queue)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Single file processing tab
        self.single_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.single_tab, text="Single File")
        self.create_single_tab()
        
        # Batch processing tab
        self.batch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_tab, text="Batch Processing")
        self.create_batch_tab()
        
        # Settings tab
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="Settings")
        self.create_settings_tab()
        
        # Log tab
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Logs")
        self.create_log_tab()
    
    def create_single_tab(self):
        """Create single file processing tab"""
        # Input section
        input_frame = ttk.LabelFrame(self.single_tab, text="Input", padding=10)
        
        # File/URL input
        ttk.Label(input_frame, text="File or YouTube URL:").grid(row=0, column=0, sticky='w', pady=2)
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_var, width=50)
        self.input_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Button(input_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=2)
        
        # Output directory
        ttk.Label(input_frame, text="Output Directory:").grid(row=1, column=0, sticky='w', pady=2)
        self.output_var = tk.StringVar(value="./output")
        self.output_entry = ttk.Entry(input_frame, textvariable=self.output_var, width=50)
        self.output_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Button(input_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=2)
        
        input_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        # Options section
        options_frame = ttk.LabelFrame(self.single_tab, text="Options", padding=10)
        
        # Model selection
        ttk.Label(options_frame, text="Whisper Model:").grid(row=0, column=0, sticky='w', pady=2)
        self.model_var = tk.StringVar(value="large-v2")
        model_combo = ttk.Combobox(options_frame, textvariable=self.model_var, 
                                  values=['tiny', 'base', 'small', 'medium', 'large', 'large-v2'],
                                  state='readonly', width=15)
        model_combo.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Language
        ttk.Label(options_frame, text="Language:").grid(row=0, column=2, sticky='w', padx=(20,5), pady=2)
        self.language_var = tk.StringVar()
        language_entry = ttk.Entry(options_frame, textvariable=self.language_var, width=10)
        language_entry.grid(row=0, column=3, sticky='w', padx=5, pady=2)
        
        # Checkboxes
        self.colorize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Enable Speaker Colorization", 
                       variable=self.colorize_var).grid(row=1, column=0, columnspan=2, sticky='w', pady=2)
        
        self.denoise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Enable Audio Denoising", 
                       variable=self.denoise_var).grid(row=1, column=2, columnspan=2, sticky='w', pady=2)
        
        options_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
        
        # Processing section
        process_frame = ttk.Frame(self.single_tab)
        
        self.process_button = ttk.Button(process_frame, text="Start Processing", 
                                        command=self.start_single_processing)
        self.process_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(process_frame, text="Stop", 
                                     command=self.stop_processing, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        process_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Progress section
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(self.single_tab, textvariable=self.progress_var).grid(row=3, column=0, columnspan=2, pady=5)
        
        self.progress_bar = ttk.Progressbar(self.single_tab, mode='indeterminate')
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
    
    def create_batch_tab(self):
        """Create batch processing tab"""
        # Input section
        input_frame = ttk.LabelFrame(self.batch_tab, text="Batch Input", padding=10)
        
        ttk.Label(input_frame, text="Input Directory:").grid(row=0, column=0, sticky='w', pady=2)
        self.batch_input_var = tk.StringVar()
        self.batch_input_entry = ttk.Entry(input_frame, textvariable=self.batch_input_var, width=50)
        self.batch_input_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Button(input_frame, text="Browse", command=self.browse_batch_input).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Output Directory:").grid(row=1, column=0, sticky='w', pady=2)
        self.batch_output_var = tk.StringVar(value="./batch_output")
        self.batch_output_entry = ttk.Entry(input_frame, textvariable=self.batch_output_var, width=50)
        self.batch_output_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Button(input_frame, text="Browse", command=self.browse_batch_output).grid(row=1, column=2, padx=5, pady=2)
        
        # Options
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(input_frame, text="Include subdirectories", 
                       variable=self.recursive_var).grid(row=2, column=0, sticky='w', pady=2)
        
        ttk.Label(input_frame, text="Max Workers:").grid(row=2, column=1, sticky='w', padx=5, pady=2)
        self.workers_var = tk.StringVar(value="2")
        workers_spin = ttk.Spinbox(input_frame, from_=1, to=8, textvariable=self.workers_var, width=5)
        workers_spin.grid(row=2, column=2, sticky='w', padx=5, pady=2)
        
        input_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(self.batch_tab)
        
        ttk.Button(control_frame, text="Start Batch Processing", 
                  command=self.start_batch_processing).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Stop Batch", 
                  command=self.stop_processing).pack(side='left', padx=5)
        
        control_frame.grid(row=1, column=0, pady=10)
        
        # Progress display
        self.batch_progress_var = tk.StringVar(value="No batch jobs")
        ttk.Label(self.batch_tab, textvariable=self.batch_progress_var).grid(row=2, column=0, pady=5)
        
        self.batch_progress_bar = ttk.Progressbar(self.batch_tab, mode='determinate')
        self.batch_progress_bar.grid(row=3, column=0, sticky='ew', padx=10, pady=5)
    
    def create_settings_tab(self):
        """Create settings configuration tab"""
        settings_frame = ttk.LabelFrame(self.settings_tab, text="Configuration", padding=10)
        
        # Device selection
        ttk.Label(settings_frame, text="Processing Device:").grid(row=0, column=0, sticky='w', pady=2)
        self.device_var = tk.StringVar(value="auto")
        device_combo = ttk.Combobox(settings_frame, textvariable=self.device_var,
                                   values=['auto', 'cpu', 'cuda', 'mps'], state='readonly')
        device_combo.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Audio quality
        ttk.Label(settings_frame, text="Audio Quality:").grid(row=1, column=0, sticky='w', pady=2)
        self.quality_var = tk.StringVar(value="high")
        quality_combo = ttk.Combobox(settings_frame, textvariable=self.quality_var,
                                    values=['low', 'medium', 'high'], state='readonly')
        quality_combo.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # Speaker threshold
        ttk.Label(settings_frame, text="Speaker Threshold:").grid(row=2, column=0, sticky='w', pady=2)
        self.threshold_var = tk.StringVar(value="0.95")
        threshold_spin = ttk.Spinbox(settings_frame, from_=0.5, to=1.0, increment=0.05,
                                    textvariable=self.threshold_var, width=10)
        threshold_spin.grid(row=2, column=1, sticky='w', padx=5, pady=2)
        
        settings_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=5)
        
        # System info
        info_frame = ttk.LabelFrame(self.settings_tab, text="System Information", padding=10)
        
        # Display device info
        device_info = self.get_device_info()
        info_text = scrolledtext.ScrolledText(info_frame, height=10, width=60)
        info_text.insert('1.0', device_info)
        info_text.config(state='disabled')
        info_text.grid(row=0, column=0, sticky='nsew')
        
        info_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)
        
        # Make info frame expandable
        self.settings_tab.grid_rowconfigure(1, weight=1)
        info_frame.grid_rowconfigure(0, weight=1)
        info_frame.grid_columnconfigure(0, weight=1)
    
    def create_log_tab(self):
        """Create logging display tab"""
        self.log_text = scrolledtext.ScrolledText(self.log_tab, height=20, width=80)
        self.log_text.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Control buttons
        log_control_frame = ttk.Frame(self.log_tab)
        ttk.Button(log_control_frame, text="Clear Logs", command=self.clear_logs).pack(side='left', padx=5)
        ttk.Button(log_control_frame, text="Save Logs", command=self.save_logs).pack(side='left', padx=5)
        log_control_frame.grid(row=1, column=0, pady=5)
        
        # Make log area expandable
        self.log_tab.grid_rowconfigure(0, weight=1)
        self.log_tab.grid_columnconfigure(0, weight=1)
    
    def setup_layout(self):
        """Setup main layout"""
        self.notebook.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Make main window expandable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Configure tab expansions
        self.single_tab.grid_columnconfigure(0, weight=1)
        self.batch_tab.grid_columnconfigure(0, weight=1)
        self.settings_tab.grid_columnconfigure(0, weight=1)
    
    def browse_input_file(self):
        """Browse for input file"""
        filetypes = [
            ('Media files', '*.mp4 *.avi *.mov *.mkv *.wav *.mp3 *.flac *.m4a'),
            ('Video files', '*.mp4 *.avi *.mov *.mkv'),
            ('Audio files', '*.wav *.mp3 *.flac *.m4a'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=filetypes
        )
        
        if filename:
            self.input_var.set(filename)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_var.set(directory)
    
    def browse_batch_input(self):
        """Browse for batch input directory"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.batch_input_var.set(directory)
    
    def browse_batch_output(self):
        """Browse for batch output directory"""
        directory = filedialog.askdirectory(title="Select Batch Output Directory")
        if directory:
            self.batch_output_var.set(directory)
    
    def get_device_info(self) -> str:
        """Get system device information"""
        try:
            device = DeviceManager.get_optimal_device()
            info = [f"Optimal Device: {device}"]
            
            if device.type == 'cuda':
                import torch
                info.append(f"GPU: {torch.cuda.get_device_name(0)}")
                info.append(f"CUDA Version: {torch.version.cuda}")
                info.append(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Python and system info
            import sys, platform
            info.append(f"Python: {sys.version}")
            info.append(f"Platform: {platform.platform()}")
            
            return "\n".join(info)
            
        except Exception as e:
            return f"Error getting device info: {e}"
    
    def build_config(self) -> Dict[str, Any]:
        """Build processing configuration from GUI settings"""
        return {
            'whisper_model': self.model_var.get(),
            'language': self.language_var.get() or None,
            'device': self.device_var.get(),
            'audio_quality': self.quality_var.get(),
            'speaker_colorization': {
                'enabled': self.colorize_var.get(),
                'threshold': float(self.threshold_var.get())
            },
            'audio_processing': {
                'denoise': self.denoise_var.get(),
                'normalize': True
            }
        }
    
    def start_single_processing(self):
        """Start single file processing in background thread"""
        if self.is_processing:
            return
        
        input_path = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()
        
        if not input_path:
            messagebox.showerror("Error", "Please select an input file or enter a URL")
            return
        
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        # Start processing in background
        self.is_processing = True
        self.process_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar.start()
        
        # Start processing thread
        thread = threading.Thread(target=self.process_single_file, 
                                 args=(input_path, output_dir))
        thread.daemon = True
        thread.start()
    
    def process_single_file(self, input_path: str, output_dir: str):
        """Process single file in background thread"""
        try:
            config = self.build_config()
            config['output_dir'] = output_dir
            
            self.processing_queue.put(("status", "Processing started..."))
            
            processor = EnhancedSubtitleProcessor(config)
            result = processor.process_audio(input_path)
            
            self.processing_queue.put(("complete", "Processing completed successfully!"))
            
        except Exception as e:
            self.processing_queue.put(("error", f"Processing failed: {e}"))
        
        finally:
            self.processing_queue.put(("finished", None))
    
    def start_batch_processing(self):
        """Start batch processing"""
        if self.is_processing:
            return
        
        input_dir = self.batch_input_var.get().strip()
        output_dir = self.batch_output_var.get().strip()
        
        if not input_dir or not os.path.exists(input_dir):
            messagebox.showerror("Error", "Please select a valid input directory")
            return
        
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        # Start batch processing
        self.is_processing = True
        
        thread = threading.Thread(target=self.process_batch_files,
                                 args=(input_dir, output_dir))
        thread.daemon = True
        thread.start()
    
    def process_batch_files(self, input_dir: str, output_dir: str):
        """Process batch files in background thread"""
        try:
            config = self.build_config()
            max_workers = int(self.workers_var.get())
            
            processor = BatchProcessor(config, max_workers=max_workers)
            
            # Add directory jobs
            jobs_added = processor.add_directory(input_dir, output_dir, 
                                               recursive=self.recursive_var.get())
            
            if jobs_added == 0:
                self.processing_queue.put(("error", "No media files found in directory"))
                return
            
            self.processing_queue.put(("status", f"Starting batch processing of {jobs_added} files..."))
            
            # Process with progress callback
            def progress_callback(job, completed, failed):
                total = completed + failed
                progress = (total / jobs_added) * 100
                self.processing_queue.put(("batch_progress", (progress, completed, failed, jobs_added)))
            
            result = processor.process_all(progress_callback=progress_callback)
            
            self.processing_queue.put(("batch_complete", result))
            
        except Exception as e:
            self.processing_queue.put(("error", f"Batch processing failed: {e}"))
        
        finally:
            self.processing_queue.put(("finished", None))
    
    def stop_processing(self):
        """Stop current processing"""
        # Note: This is a simple implementation
        # In a real application, you'd need proper thread cancellation
        self.is_processing = False
        self.processing_queue.put(("stopped", "Processing stopped by user"))
    
    def process_queue(self):
        """Process messages from background threads"""
        try:
            while True:
                message_type, data = self.processing_queue.get_nowait()
                
                if message_type == "status":
                    self.progress_var.set(data)
                    self.log_message(data)
                
                elif message_type == "complete":
                    self.progress_var.set(data)
                    self.log_message(data)
                    messagebox.showinfo("Success", data)
                
                elif message_type == "error":
                    self.progress_var.set(data)
                    self.log_message(f"ERROR: {data}")
                    messagebox.showerror("Error", data)
                
                elif message_type == "batch_progress":
                    progress, completed, failed, total = data
                    self.batch_progress_bar.config(value=progress)
                    status = f"Progress: {completed}/{total} completed, {failed} failed"
                    self.batch_progress_var.set(status)
                
                elif message_type == "batch_complete":
                    result = data
                    success_rate = result['success_rate']
                    message = f"Batch complete: {result['completed']}/{result['total_jobs']} files ({success_rate:.1f}% success)"
                    self.batch_progress_var.set(message)
                    messagebox.showinfo("Batch Complete", message)
                
                elif message_type == "finished":
                    self.is_processing = False
                    self.process_button.config(state='normal')
                    self.stop_button.config(state='disabled')
                    self.progress_bar.stop()
                
                elif message_type == "stopped":
                    self.progress_var.set(data)
                    self.log_message(data)
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_queue)
    
    def log_message(self, message: str):
        """Add message to log display"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert('end', log_entry)
        self.log_text.see('end')
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete('1.0', 'end')
    
    def save_logs(self):
        """Save logs to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    logs = self.log_text.get('1.0', 'end')
                    f.write(logs)
                messagebox.showinfo("Success", f"Logs saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {e}")
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

def main():
    """GUI entry point"""
    app = SubtitleGUI()
    app.run()

if __name__ == '__main__':
    main()