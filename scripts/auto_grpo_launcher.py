#!/usr/bin/env python3
"""
Auto GRPO Pipeline Launcher

Monitors GRPO data generation completion and automatically launches
the full training pipeline with correct parameters.

Usage:
    python auto_grpo_launcher.py --watch_dir data_grpo_subset --sft_model models/sft_qwen3_14b_lora_30k
"""

import argparse
import os
import time
import subprocess
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_grpo_launcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GRPOAutoLauncher:
    """Monitors data generation and auto-launches GRPO training pipeline"""
    
    REQUIRED_TASKS = ['bluebook', 'holding', 'summarise', 'retrieval', 'entail']
    CHECK_INTERVAL = 60  # Check every 60 seconds
    
    def __init__(self, watch_dir: str, sft_model_path: str, output_dir: str = "models/grpo_auto"):
        """
        Initialize auto launcher.
        
        Args:
            watch_dir: Directory to monitor for GRPO data completion
            sft_model_path: Path to SFT model for training
            output_dir: Output directory for trained models
        """
        self.watch_dir = Path(watch_dir)
        self.sft_model_path = sft_model_path
        self.output_dir = output_dir
        self.start_time = datetime.now()
        
        logger.info(f"Auto GRPO Launcher initialized")
        logger.info(f"Watching directory: {self.watch_dir}")
        logger.info(f"SFT model: {self.sft_model_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def check_data_completion(self) -> bool:
        """
        Check if all required GRPO data files exist and are valid.
        
        Returns:
            True if all data is ready for training
        """
        if not self.watch_dir.exists():
            logger.debug(f"Watch directory {self.watch_dir} does not exist yet")
            return False
        
        # Check for required task directories and files
        missing_files = []
        
        for task in self.REQUIRED_TASKS:
            task_dir = self.watch_dir / task
            train_file = task_dir / 'train_grpo.json'
            eval_file = task_dir / 'eval_grpo.json'
            
            if not task_dir.exists():
                missing_files.append(f"{task}/")
                continue
                
            if not train_file.exists():
                missing_files.append(f"{task}/train_grpo.json")
            elif train_file.stat().st_size == 0:
                missing_files.append(f"{task}/train_grpo.json (empty)")
                
            if not eval_file.exists():
                missing_files.append(f"{task}/eval_grpo.json")
            elif eval_file.stat().st_size == 0:
                missing_files.append(f"{task}/eval_grpo.json (empty)")
        
        if missing_files:
            logger.debug(f"Missing files: {missing_files}")
            return False
        
        # Validate JSON format of key files
        try:
            for task in self.REQUIRED_TASKS:
                train_file = self.watch_dir / task / 'train_grpo.json'
                with open(train_file, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list) or len(data) == 0:
                        logger.warning(f"Invalid or empty data in {train_file}")
                        return False
                    
                    # Check first sample has required fields
                    sample = data[0]
                    required_fields = ['query', 'responses', 'scores']
                    if not all(field in sample for field in required_fields):
                        logger.warning(f"Missing required fields in {train_file}")
                        return False
                        
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.debug(f"Data validation error: {e}")
            return False
        
        logger.info("✅ All GRPO data files are ready!")
        return True
    
    def check_generation_process(self) -> bool:
        """
        Check if GRPO data generation process is still running.
        
        Returns:
            True if generation is still in progress
        """
        try:
            # Check for running prep_grpo processes
            result = subprocess.run(
                ['pgrep', '-f', 'prep_grpo_dataset'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                logger.debug(f"Found {len(pids)} GRPO generation processes running")
                return True
            else:
                logger.debug("No GRPO generation processes found")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking generation process: {e}")
            return False
    
    def launch_grpo_pipeline(self) -> bool:
        """
        Launch the GRPO training pipeline with correct parameters.
        
        Returns:
            True if launch was successful
        """
        logger.info("🚀 Launching GRPO training pipeline...")
        
        cmd = [
            'python', 'scripts/orchestrate_grpo_training.py',
            '--sft_model_path', self.sft_model_path,
            '--data_dir', str(self.watch_dir),
            '--output_dir', self.output_dir,
            '--start_stage', '0'
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            # Launch in background and redirect output to log file
            log_file = f"grpo_auto_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd='.'
                )
            
            logger.info(f"✅ GRPO pipeline launched successfully (PID: {process.pid})")
            logger.info(f"📋 Training logs: {log_file}")
            logger.info(f"🎯 Output directory: {self.output_dir}")
            
            # Save launch info
            launch_info = {
                'timestamp': datetime.now().isoformat(),
                'pid': process.pid,
                'command': cmd,
                'log_file': log_file,
                'sft_model': self.sft_model_path,
                'data_dir': str(self.watch_dir),
                'output_dir': self.output_dir
            }
            
            with open('grpo_auto_launch_info.json', 'w') as f:
                json.dump(launch_info, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to launch GRPO pipeline: {e}")
            return False
    
    def monitor_and_launch(self) -> None:
        """
        Main monitoring loop that watches for completion and launches pipeline.
        """
        logger.info("🔍 Starting monitoring loop...")
        logger.info(f"Checking every {self.CHECK_INTERVAL} seconds")
        
        check_count = 0
        
        while True:
            check_count += 1
            elapsed = datetime.now() - self.start_time
            
            logger.info(f"Check #{check_count} (elapsed: {elapsed})")
            
            # Check if data generation is complete
            if self.check_data_completion():
                logger.info("🎉 GRPO data generation completed!")
                
                # Launch the pipeline
                if self.launch_grpo_pipeline():
                    logger.info("✅ Auto launcher completed successfully")
                    break
                else:
                    logger.error("❌ Failed to launch pipeline")
                    sys.exit(1)
            
            # Check if generation process is still running
            elif not self.check_generation_process():
                logger.warning("⚠️  No generation process found, but data incomplete")
                logger.warning("This may indicate the generation failed or was interrupted")
                
                # Wait a bit longer in case it's just finishing up
                logger.info(f"Waiting additional {self.CHECK_INTERVAL*2} seconds...")
                time.sleep(self.CHECK_INTERVAL * 2)
                
                # Check one more time
                if not self.check_data_completion():
                    logger.error("❌ Data generation appears to have failed")
                    sys.exit(1)
            
            # Wait before next check
            logger.debug(f"Waiting {self.CHECK_INTERVAL} seconds...")
            time.sleep(self.CHECK_INTERVAL)

def main():
    parser = argparse.ArgumentParser(description='Auto GRPO Pipeline Launcher')
    
    parser.add_argument('--watch_dir', required=True,
                       help='Directory to monitor for GRPO data completion')
    parser.add_argument('--sft_model_path', required=True,
                       help='Path to SFT model for training')
    parser.add_argument('--output_dir', default='models/grpo_auto',
                       help='Output directory for trained models (default: models/grpo_auto)')
    parser.add_argument('--check_interval', type=int, default=60,
                       help='Check interval in seconds (default: 60)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.sft_model_path).exists():
        parser.error(f"SFT model path does not exist: {args.sft_model_path}")
    
    # Create launcher and start monitoring
    launcher = GRPOAutoLauncher(
        watch_dir=args.watch_dir,
        sft_model_path=args.sft_model_path,
        output_dir=args.output_dir
    )
    
    # Override check interval if specified
    if args.check_interval != 60:
        launcher.CHECK_INTERVAL = args.check_interval
        logger.info(f"Using custom check interval: {args.check_interval} seconds")
    
    try:
        launcher.monitor_and_launch()
    except KeyboardInterrupt:
        logger.info("❌ Monitoring interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise

if __name__ == '__main__':
    main()