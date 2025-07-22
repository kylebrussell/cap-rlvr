#!/usr/bin/env python3
"""
Robust CAP dataset downloader with error handling, retries, and progress monitoring.
"""
import time
import sys
import os
from pathlib import Path
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_cap_dataset(output_dir: str = "data_raw/cap", max_retries: int = 3):
    """
    Download CAP dataset with robust error handling and retries.
    
    Args:
        output_dir: Directory to save the dataset
        max_retries: Maximum number of retry attempts
    """
    output_path = Path(output_dir)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Starting CAP dataset download...")
            
            # Clear any partial downloads
            if output_path.exists():
                logger.info(f"Removing partial download at {output_path}")
                import shutil
                shutil.rmtree(output_path)
            
            # Download with progress tracking
            logger.info("Loading dataset from HuggingFace...")
            dataset = load_dataset(
                "common-pile/caselaw_access_project", 
                split="train",
                streaming=False,  # Disable streaming for better error handling
                trust_remote_code=False,
                download_mode="force_redownload" if attempt > 0 else "reuse_cache_if_exists"
            )
            
            logger.info(f"Dataset loaded: {len(dataset)} examples")
            logger.info(f"Saving dataset to {output_path}...")
            
            # Save with progress monitoring
            dataset.save_to_disk(str(output_path))
            
            # Verify the saved dataset
            logger.info("Verifying saved dataset...")
            from datasets import load_from_disk
            verified_dataset = load_from_disk(str(output_path))
            logger.info(f"Verification successful: {len(verified_dataset)} examples")
            
            logger.info("Dataset download and save completed successfully!")
            return True
            
        except KeyboardInterrupt:
            logger.info("Download interrupted by user")
            sys.exit(1)
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 60  # Exponential backoff: 1min, 2min, 3min
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
                # Clean up HuggingFace cache to force fresh download
                logger.info("Cleaning HuggingFace cache...")
                try:
                    import huggingface_hub
                    huggingface_hub.scan_cache_dir().delete_revisions("*").execute()
                except Exception as cache_error:
                    logger.warning(f"Could not clean cache: {cache_error}")
            else:
                logger.error("All retry attempts failed")
                return False
    
    return False

def check_disk_space(required_gb: int = 80):
    """Check if enough disk space is available."""
    import shutil
    
    free_bytes = shutil.disk_usage('.').free
    free_gb = free_bytes / (1024**3)
    
    logger.info(f"Available disk space: {free_gb:.1f} GB")
    
    if free_gb < required_gb:
        logger.error(f"Insufficient disk space. Need {required_gb} GB, have {free_gb:.1f} GB")
        return False
    
    return True

def main():
    """Main download function with pre-checks."""
    logger.info("Starting robust CAP dataset download")
    
    # Pre-flight checks
    if not check_disk_space():
        sys.exit(1)
    
    # Set working directory
    os.chdir(os.path.expanduser("~/cap_rlvr"))
    
    # Download dataset
    success = download_cap_dataset()
    
    if success:
        logger.info("Download completed successfully!")
        
        # Final verification
        dataset_path = Path("data_raw/cap")
        if dataset_path.exists():
            size_mb = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file()) / (1024**2)
            logger.info(f"Dataset size on disk: {size_mb:.1f} MB")
        
        sys.exit(0)
    else:
        logger.error("Download failed after all retries")
        sys.exit(1)

if __name__ == "__main__":
    main()