#!/usr/bin/env python3
"""
Alternative CAP dataset download using HuggingFace CLI for maximum robustness.
"""
import subprocess
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_via_cli():
    """
    Download using HuggingFace CLI which has better retry/resume capabilities.
    """
    os.chdir(os.path.expanduser("~/cap_rlvr"))
    
    # Create data directory
    Path("data_raw").mkdir(exist_ok=True)
    
    logger.info("Downloading CAP dataset via HuggingFace CLI...")
    
    try:
        # Use huggingface-cli to download with built-in resume capability
        cmd = [
            "huggingface-cli",
            "download", 
            "common-pile/caselaw_access_project",
            "--repo-type=dataset",
            "--local-dir=data_raw/cap_raw",
            "--resume-download"  # Key: Built-in resume capability
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        
        logger.info("CLI download completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"CLI download failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("huggingface-cli not found. Install with: pip install huggingface_hub[cli]")
        return False

if __name__ == "__main__":
    success = download_via_cli()
    sys.exit(0 if success else 1)