"""
Batch processing module for Subtitle AI Suite
"""

import os
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Update imports to use absolute paths
from src.subtitle_suite.utils.error_handler import ErrorHandler, SubtitleProcessingError
from src.subtitle_suite.utils.logger import setup_logging
# Removed: from .subtitle_processor import EnhancedSubtitleProcessor

@dataclass
class BatchJob:
    """Represents a single batch processing job"""
    id: str
    input_path: str
    output_dir: str
    config: Dict[str, Any]
    status: str = 'pending'  # pending, processing, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    output_files: List[str] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []

class BatchProcessor:
    """
    Advanced batch processing with parallel execution and progress tracking
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 max_workers: int = 2,
                 use_multiprocessing: bool = False):
        """
        Initialize batch processor
        
        Args:
            config (Dict[str, Any]): Processing configuration
            max_workers (int): Maximum number of parallel workers
            use_multiprocessing (bool): Use processes instead of threads
        """
        self.config = config
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        
        self.logger = setup_logging('batch_processor')
        # Assuming ErrorHandler is initialized with a logger
        self.error_handler = ErrorHandler(self.logger) 
        
        self.jobs: List[BatchJob] = []
        self.completed_jobs = 0
        self.failed_jobs = 0
        
    def add_job(self, 
                input_path: str, 
                output_dir: str, 
                job_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a job to the batch queue
        
        Args:
            input_path (str): Path to input media file
            output_dir (str): Output directory for this job
            job_config (Dict[str, Any], optional): Job-specific configuration
        
        Returns:
            str: Job ID
        """
        # Generate unique job ID
        job_id = f"job_{len(self.jobs):04d}_{int(time.time())}"
        
        # Merge job-specific config with global config
        final_config = self.config.copy()
        if job_config:
            final_config.update(job_config)
        
        # Create job
        job = BatchJob(
            id=job_id,
            input_path=input_path,
            output_dir=output_dir,
            config=final_config
        )
        
        self.jobs.append(job)
        self.logger.info(f"Added job {job_id}: {input_path}")
        
        return job_id
    
    def add_directory(self, 
                      input_dir: str, 
                      output_dir: str,
                      recursive: bool = True,
                      file_patterns: List[str] = None) -> int:
        """
        Add all media files from a directory to batch queue
        
        Args:
            input_dir (str): Input directory path
            output_dir (str): Output directory path
            recursive (bool): Search recursively in subdirectories
            file_patterns (List[str], optional): File patterns to match
        
        Returns:
            int: Number of jobs added
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Default media file extensions
        if file_patterns is None:
            file_patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv', 
                           '*.wav', '*.mp3', '*.flac', '*.m4a']
        
        # Find media files
        media_files = []
        for pattern in file_patterns:
            if recursive:
                media_files.extend(input_path.rglob(pattern))
            else:
                media_files.extend(input_path.glob(pattern))
        
        # Add jobs for each file
        jobs_added = 0
        for media_file in media_files:
            # Create relative output directory structure
            relative_path = media_file.relative_to(input_path).parent
            file_output_dir = Path(output_dir) / relative_path
            
            self.add_job(str(media_file), str(file_output_dir))
            jobs_added += 1
        
        self.logger.info(f"Added {jobs_added} jobs from directory: {input_dir}")
        return jobs_added
    
    def add_playlist(self, playlist_url: str, output_dir: str) -> int:
        """
        Add all videos from a playlist to batch queue
        
        Args:
            playlist_url (str): YouTube playlist URL
            output_dir (str): Output directory path
        
        Returns:
            int: Number of jobs added
        """
        try:
            import yt_dlp
            
            # Extract playlist information
            ydl_opts = {
                'extract_flat': True,
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
                
                if 'entries' not in playlist_info:
                    raise ValueError("Not a valid playlist URL")
                
                # Add job for each video
                jobs_added = 0
                for entry in playlist_info['entries']:
                    if entry:
                        video_url = entry['url']
                        video_title = entry.get('title', f"video_{jobs_added}")
                        
                        # Create clean filename
                        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                        video_output_dir = Path(output_dir) / safe_title
                        
                        self.add_job(video_url, str(video_output_dir))
                        jobs_added += 1
                
                self.logger.info(f"Added {jobs_added} jobs from playlist: {playlist_url}")
                return jobs_added
                
        except ImportError:
            raise SubtitleProcessingError("yt-dlp not available for playlist processing")
        except Exception as e:
            raise SubtitleProcessingError(f"Failed to process playlist: {e}")
    
    def process_job(self, job: BatchJob) -> BatchJob:
        """
        Process a single batch job
        
        Args:
            job (BatchJob): Job to process
        
        Returns:
            BatchJob: Updated job with results
        """
        job.status = 'processing'
        job.start_time = time.time()
        
        try:
            # Validate input
            # self.error_handler.validate_input(job.input_path) # ErrorHandler.validate_input does not exist
            
            # Create output directory
            os.makedirs(job.output_dir, exist_ok=True)
            
            # Process the file - Placeholder for SubtitlePipelineManager
            # This part will be updated to use SubtitlePipelineManager in a later step
            self.logger.info(f"Processing job {job.id} for input: {job.input_path}")
            # result = processor.process_audio(job.input_path) # Removed
            # Placeholder for actual processing result
            result = {"success": True, "generated_files": []} 
            
            # Record output files
            job.output_files = result.get("generated_files", []) # Placeholder for output files
            
            job.status = 'completed'
            job.end_time = time.time()
            
            self.logger.info(f"Job {job.id} completed in {job.end_time - job.start_time:.2f}s")
            
        except Exception as e:
            job.status = 'failed'
            job.end_time = time.time()
            job.error_message = str(e)
            
            self.logger.error(f"Job {job.id} failed: {e}")
            self.error_handler.handle_error(e, f"Batch job {job.id}", reraise=False)
        
        return job
    
    def _find_output_files(self, output_dir: str) -> List[str]:
        """Find all output files in the output directory (simplified for BatchJob placeholder)"""
        output_path = Path(output_dir)
        if not output_path.exists():
            return []
        
        output_files = []
        for file_path in output_path.iterdir():
            if file_path.is_file():
                output_files.append(str(file_path))
        
        return output_files
    
    def process_all(self, 
                    progress_callback: Optional[Callable] = None,
                    save_progress: bool = True) -> Dict[str, Any]:
        """
        Process all jobs in the batch queue
        
        Args:
            progress_callback (Callable, optional): Callback for progress updates
            save_progress (bool): Save progress to file
        
        Returns:
            Dict containing batch processing results
        """
        if not self.jobs:
            self.logger.warning("No jobs to process")
            return {'total_jobs': 0, 'completed': 0, 'failed': 0}
        
        self.logger.info(f"Starting batch processing of {len(self.jobs)} jobs")
        start_time = time.time()
        
        # Choose executor based on configuration
        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.process_job, job): job 
                for job in self.jobs
            }
            
            # Process completed jobs with progress bar
            with tqdm(total=len(self.jobs), desc="Processing") as pbar:
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    
                    try:
                        updated_job = future.result()
                        
                        # Update job in list
                        job_index = next(i for i, j in enumerate(self.jobs) if j.id == updated_job.id)
                        self.jobs[job_index] = updated_job
                        
                        if updated_job.status == 'completed':
                            self.completed_jobs += 1
                        else:
                            self.failed_jobs += 1
                        
                        pbar.update(1)
                        
                        # Call progress callback
                        if progress_callback:
                            progress_callback(updated_job, self.completed_jobs, self.failed_jobs)
                        
                        # Save progress periodically
                        if save_progress and (self.completed_jobs + self.failed_jobs) % 5 == 0:
                            self.save_progress()
                            
                    except Exception as e:
                        self.logger.error(f"Error processing job {job.id}: {e}")
                        self.failed_jobs += 1
                        pbar.update(1)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Generate final report
        report = {
            'total_jobs': len(self.jobs),
            'completed': self.completed_jobs,
            'failed': self.failed_jobs,
            'success_rate': self.completed_jobs / len(self.jobs) * 100,
            'total_time': total_time,
            'average_time_per_job': total_time / len(self.jobs),
            'jobs': [asdict(job) for job in self.jobs]
        }
        
        self.logger.info(f"Batch processing completed: {self.completed_jobs}/{len(self.jobs)} jobs successful")
        
        if save_progress:
            self.save_report(report)
        
        return report
    
    def save_progress(self, filename: Optional[str] = None):
        """Save current progress to file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"batch_progress_{timestamp}.json"
        
        progress_data = {
            'timestamp': time.time(),
            'total_jobs': len(self.jobs),
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'jobs': [asdict(job) for job in self.jobs]
        }
        
        progress_path = Path(self.config.get('output_dir', './output')) / filename
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        self.logger.debug(f"Progress saved to: {progress_path}")
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save final batch report"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"batch_report_{timestamp}.json"
        
        report_path = Path(self.config.get('output_dir', './output')) / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Batch report saved to: {report_path}")
    
    def load_progress(self, filename: str):
        """Load progress from file and resume processing"""
        progress_path = Path(filename)
        if not progress_path.exists():
            raise FileNotFoundError(f"Progress file not found: {filename}")
        
        with open(progress_path, 'r') as f:
            progress_data = json.load(f)
        
        # Restore jobs
        self.jobs = [BatchJob(**job_data) for job_data in progress_data['jobs']]
        self.completed_jobs = progress_data['completed_jobs']
        self.failed_jobs = progress_data['failed_jobs']
        
        self.logger.info(f"Progress loaded from: {filename}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current batch processing status"""
        return {
            'total_jobs': len(self.jobs),
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'pending_jobs': len([j for j in self.jobs if j.status == 'pending']),
            'processing_jobs': len([j for j in self.jobs if j.status == 'processing']),
            'success_rate': self.completed_jobs / len(self.jobs) * 100 if self.jobs else 0
        }