#!/usr/bin/env python3
"""
GRPO Training Orchestration Script

Automates the multi-stage GRPO training pipeline with automatic 
stage progression validation and model management.

Usage:
    python orchestrate_grpo_training.py --start_stage 0 --sft_model_path models/sft
    python orchestrate_grpo_training.py --start_stage 1 --base_model_path models/grpo/stage0_complete
"""

import argparse
import json
import logging
import subprocess
import time
import signal
import psutil
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grpo_orchestration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GRPOTrainingOrchestrator:
    """Orchestrates multi-stage GRPO training with automatic progression"""
    
    TASK_NAMES = ['bluebook', 'holding', 'summarise', 'retrieval', 'entail']
    
    STAGE_CONFIG = {
        0: {
            'name': 'Individual Task Mastery',
            'tasks': TASK_NAMES,
            'sequential': True,
            'epochs_per_task': 5,
            'learning_rate': 1e-5,
            'threshold': 0.8
        },
        1: {
            'name': 'Multi-Task Integration',
            'tasks': ['all'],
            'sequential': False,
            'max_iterations': 3,
            'epochs_per_iteration': 3,
            'learning_rate': 5e-6,
            'threshold': 0.75
        },
        2: {
            'name': 'Curriculum Refinement',
            'tasks': ['all'],
            'sequential': False,
            'max_iterations': 2,
            'epochs_per_iteration': 2,
            'learning_rate': 3e-6,
            'threshold': 0.85
        },
        3: {
            'name': 'Production Optimization',
            'tasks': ['all'],
            'sequential': False,
            'max_iterations': 2,
            'epochs_per_iteration': 2,
            'learning_rate': 1e-6,
            'threshold': 0.90
        }
    }
    
    def __init__(self, base_output_dir: str = "models/grpo", max_retries: int = 2):
        """
        Initialize GRPO training orchestrator.
        
        Args:
            base_output_dir: Base directory for model outputs
            max_retries: Maximum retries per stage before giving up
        """
        self.base_output_dir = Path(base_output_dir)
        self.max_retries = max_retries
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.current_process = None
        self.shutdown_requested = False
        self.progress_file = self.base_output_dir / "training_progress.json"
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"GRPO Training Orchestrator initialized")
        logger.info(f"Output directory: {base_output_dir}")
        logger.info(f"Max retries per stage: {max_retries}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
        if self.current_process and self.current_process.poll() is None:
            logger.info("Terminating current training process...")
            try:
                # Try graceful termination first
                self.current_process.terminate()
                
                # Wait up to 30 seconds for graceful shutdown
                try:
                    self.current_process.wait(timeout=30)
                    logger.info("Training process terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Graceful termination failed, forcing kill...")
                    self.current_process.kill()
                    self.current_process.wait()
                    logger.info("Training process killed")
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
        
        logger.info("Orchestrator shutdown complete")
        sys.exit(0)
    
    def save_progress(self, stage: int, task: str = None, status: str = "in_progress", model_path: str = None):
        """Save current training progress to disk"""
        progress = {
            'timestamp': datetime.now().isoformat(),
            'current_stage': stage,
            'current_task': task,
            'status': status,
            'model_path': model_path,
            'completed_stages': [],
            'failed_attempts': []
        }
        
        # Load existing progress if available
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    existing = json.load(f)
                    progress['completed_stages'] = existing.get('completed_stages', [])
                    progress['failed_attempts'] = existing.get('failed_attempts', [])
            except:
                pass
        
        # Save updated progress
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def run_training_command(self, cmd: List[str], stage: int, task: str = None) -> bool:
        """
        Execute training command with enhanced monitoring for long-duration runs.
        
        Args:
            cmd: Command to execute
            stage: Current stage number
            task: Task name (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if self.shutdown_requested:
            logger.info("Shutdown requested, skipping training command")
            return False
            
        task_desc = f" ({task})" if task else ""
        logger.info(f"Stage {stage}{task_desc}: Starting training at {datetime.now()}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            # Start process with streaming output
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            start_time = time.time()
            last_heartbeat = start_time
            heartbeat_interval = 600  # 10 minutes
            
            # Monitor process with periodic heartbeats
            while self.current_process.poll() is None:
                if self.shutdown_requested:
                    logger.info("Shutdown requested during training")
                    return False
                
                current_time = time.time()
                
                # Heartbeat logging every 10 minutes
                if current_time - last_heartbeat >= heartbeat_interval:
                    elapsed_hours = (current_time - start_time) / 3600
                    logger.info(f"Stage {stage}{task_desc}: Training in progress ({elapsed_hours:.1f}h elapsed)")
                    
                    # Log system resources if available
                    try:
                        gpu_mem = psutil.virtual_memory().percent
                        logger.info(f"System memory usage: {gpu_mem:.1f}%")
                    except:
                        pass
                    
                    last_heartbeat = current_time
                
                # Check for timeout (6 hours for multi-hour training)
                if current_time - start_time > 3600 * 6:
                    logger.error(f"Stage {stage}{task_desc}: Training exceeded 6-hour limit")
                    self.current_process.terminate()
                    try:
                        self.current_process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.current_process.kill()
                        self.current_process.wait()
                    return False
                
                # Short sleep to avoid busy waiting
                time.sleep(30)
            
            # Get final result
            return_code = self.current_process.returncode
            total_duration = time.time() - start_time
            
            # Read any remaining output
            if self.current_process.stdout:
                try:
                    remaining_output = self.current_process.stdout.read()
                    if remaining_output.strip():
                        logger.debug(f"Final output: {remaining_output}")
                except:
                    pass
            
            if return_code == 0:
                logger.info(f"Stage {stage}{task_desc}: Completed successfully in {total_duration/3600:.2f}h")
                return True
            else:
                logger.error(f"Stage {stage}{task_desc}: Failed after {total_duration/3600:.2f}h (exit code: {return_code})")
                return False
                
        except Exception as e:
            logger.error(f"Stage {stage}{task_desc}: Training error: {e}")
            return False
        finally:
            self.current_process = None
    
    def validate_stage_completion(self, stage: int, model_path: str, task: str = None) -> bool:
        """
        Validate that stage completion requirements are met.
        
        Args:
            stage: Stage number to validate
            model_path: Path to model to validate
            task: Specific task to validate (for Stage 0)
            
        Returns:
            True if validation passes
        """
        logger.info(f"Validating Stage {stage} completion...")
        
        cmd = [
            'python', 'validate_stage_progression.py',
            '--stage', str(stage),
            '--model_path', model_path
        ]
        
        if stage == 0 and task:
            cmd.extend(['--task', task])
        elif stage == 0:
            cmd.append('--check_all_tasks')
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                logger.info(f"Stage {stage} validation: PASSED")
                return True
            else:
                logger.warning(f"Stage {stage} validation: FAILED")
                logger.warning(f"Validation output: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Stage {stage} validation timed out")
            return False
        except Exception as e:
            logger.error(f"Stage {stage} validation error: {e}")
            return False
    
    def execute_stage_0(self, sft_model_path: str) -> Optional[str]:
        """
        Execute Stage 0: Individual Task Mastery.
        
        Args:
            sft_model_path: Path to SFT model
            
        Returns:
            Path to final Stage 0 model or None if failed
        """
        stage = 0
        config = self.STAGE_CONFIG[stage]
        logger.info(f"Starting {config['name']}")
        
        current_model_path = sft_model_path
        
        for task in config['tasks']:
            if self.shutdown_requested:
                logger.info("Shutdown requested, stopping Stage 0 execution")
                return None
                
            task_output_dir = self.base_output_dir / f"{task}_grpo"
            retry_count = 0
            
            while retry_count <= self.max_retries and not self.shutdown_requested:
                logger.info(f"Stage 0 - {task}: Attempt {retry_count + 1}")
                
                # Training command
                cmd = [
                    'python', 'train_grpo.py',
                    '--task', task,
                    '--model_path', current_model_path,
                    '--data_path', f'../data_grpo/{task}/train_grpo.json',
                    '--eval_data_path', f'../data_grpo/{task}/eval_grpo.json',
                    '--output_dir', str(task_output_dir),
                    '--num_epochs', str(config['epochs_per_task']),
                    '--learning_rate', str(config['learning_rate'])
                ]
                
                # Execute training
                if self.run_training_command(cmd, stage, task):
                    # Validate completion
                    if self.validate_stage_completion(stage, str(task_output_dir), task):
                        logger.info(f"Stage 0 - {task}: Completed successfully")
                        current_model_path = str(task_output_dir)
                        break
                    else:
                        logger.warning(f"Stage 0 - {task}: Validation failed, retrying...")
                
                retry_count += 1
                
                if retry_count > self.max_retries:
                    logger.error(f"Stage 0 - {task}: Failed after {self.max_retries} retries")
                    return None
        
        # Final Stage 0 validation (all tasks)
        if self.validate_stage_completion(stage, str(self.base_output_dir), None):
            stage0_complete_path = self.base_output_dir / "stage0_complete"
            
            # Copy final model
            import shutil
            if stage0_complete_path.exists():
                shutil.rmtree(stage0_complete_path)
            shutil.copytree(current_model_path, stage0_complete_path)
            
            logger.info(f"Stage 0 completed successfully: {stage0_complete_path}")
            return str(stage0_complete_path)
        else:
            logger.error("Stage 0 final validation failed")
            return None
    
    def execute_multi_task_stage(self, stage: int, input_model_path: str) -> Optional[str]:
        """
        Execute multi-task stages (1, 2, 3).
        
        Args:
            stage: Stage number (1, 2, or 3)
            input_model_path: Path to input model
            
        Returns:
            Path to final stage model or None if failed
        """
        config = self.STAGE_CONFIG[stage]
        logger.info(f"Starting Stage {stage}: {config['name']}")
        
        current_model_path = input_model_path
        
        for iteration in range(config['max_iterations']):
            iteration_output_dir = self.base_output_dir / f"stage{stage}_iter{iteration + 1}"
            retry_count = 0
            
            while retry_count <= self.max_retries:
                logger.info(f"Stage {stage} - Iteration {iteration + 1}: Attempt {retry_count + 1}")
                
                # Training command
                cmd = [
                    'python', 'train_grpo.py',
                    '--task', 'all',
                    '--multi_task',
                    '--model_path', current_model_path,
                    '--data_path', '../data_grpo/unified/train_grpo.json',
                    '--eval_data_path', '../data_grpo/unified/eval_grpo.json',
                    '--output_dir', str(iteration_output_dir),
                    '--num_epochs', str(config['epochs_per_iteration']),
                    '--learning_rate', str(config['learning_rate'])
                ]
                
                # Stage-specific adjustments
                if stage == 2:
                    cmd.extend(['--beta', '0.15'])  # Higher KL penalty for stability
                elif stage == 3:
                    cmd.extend(['--beta', '0.05'])  # Lower KL penalty for fine-tuning
                
                # Execute training
                if self.run_training_command(cmd, stage, f"iter{iteration + 1}"):
                    # Validate completion
                    if self.validate_stage_completion(stage, str(iteration_output_dir)):
                        logger.info(f"Stage {stage} - Iteration {iteration + 1}: Completed successfully")
                        current_model_path = str(iteration_output_dir)
                        break
                    else:
                        logger.warning(f"Stage {stage} - Iteration {iteration + 1}: Validation failed")
                        
                        # If validation failed but we're not at max iterations, continue
                        if iteration < config['max_iterations'] - 1:
                            logger.info("Continuing to next iteration...")
                            current_model_path = str(iteration_output_dir)
                            break
                
                retry_count += 1
                
                if retry_count > self.max_retries:
                    if iteration < config['max_iterations'] - 1:
                        logger.warning(f"Stage {stage} - Iteration {iteration + 1}: Failed, trying next iteration")
                        break
                    else:
                        logger.error(f"Stage {stage}: Failed after {self.max_retries} retries")
                        return None
        
        # Final stage completion path
        stage_complete_path = self.base_output_dir / f"stage{stage}_complete"
        
        # Copy final model
        import shutil
        if stage_complete_path.exists():
            shutil.rmtree(stage_complete_path)
        shutil.copytree(current_model_path, stage_complete_path)
        
        logger.info(f"Stage {stage} completed successfully: {stage_complete_path}")
        return str(stage_complete_path)
    
    def execute_full_pipeline(self, sft_model_path: str, start_stage: int = 0) -> Optional[str]:
        """
        Execute the complete GRPO training pipeline.
        
        Args:
            sft_model_path: Path to SFT model
            start_stage: Stage to start from (0-3)
            
        Returns:
            Path to final production model or None if failed
        """
        logger.info("Starting GRPO Training Pipeline")
        logger.info(f"SFT Model: {sft_model_path}")
        logger.info(f"Starting Stage: {start_stage}")
        
        current_model_path = sft_model_path
        
        for stage in range(start_stage, 4):
            stage_start_time = time.time()
            
            if stage == 0:
                current_model_path = self.execute_stage_0(current_model_path)
            else:
                current_model_path = self.execute_multi_task_stage(stage, current_model_path)
            
            if current_model_path is None:
                logger.error(f"Pipeline failed at Stage {stage}")
                return None
            
            stage_duration = time.time() - stage_start_time
            logger.info(f"Stage {stage} completed in {stage_duration/3600:.2f} hours")
        
        # Final production model
        production_model_path = self.base_output_dir / "production_ready"
        
        import shutil
        if production_model_path.exists():
            shutil.rmtree(production_model_path)
        shutil.copytree(current_model_path, production_model_path)
        
        logger.info(f"🎉 GRPO Training Pipeline completed successfully!")
        logger.info(f"Production model: {production_model_path}")
        
        return str(production_model_path)

def main():
    parser = argparse.ArgumentParser(description='Orchestrate multi-stage GRPO training')
    
    parser.add_argument('--sft_model_path', 
                       help='Path to SFT model (required for stage 0)')
    parser.add_argument('--base_model_path', 
                       help='Path to base model for resuming from later stages')
    parser.add_argument('--start_stage', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Stage to start training from (default: 0)')
    parser.add_argument('--output_dir', default='models/grpo',
                       help='Output directory for models (default: models/grpo)')
    parser.add_argument('--max_retries', type=int, default=2,
                       help='Maximum retries per stage (default: 2)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show planned execution without running')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_stage == 0 and not args.sft_model_path:
        parser.error("--sft_model_path required when starting from stage 0")
    
    if args.start_stage > 0 and not args.base_model_path:
        parser.error("--base_model_path required when starting from stage > 0")
    
    base_model = args.sft_model_path if args.start_stage == 0 else args.base_model_path
    
    if not Path(base_model).exists():
        parser.error(f"Base model path does not exist: {base_model}")
    
    # Initialize orchestrator
    orchestrator = GRPOTrainingOrchestrator(
        base_output_dir=args.output_dir,
        max_retries=args.max_retries
    )
    
    if args.dry_run:
        logger.info("DRY RUN MODE - Showing planned execution:")
        for stage in range(args.start_stage, 4):
            config = orchestrator.STAGE_CONFIG[stage]
            logger.info(f"Stage {stage}: {config['name']}")
            if stage == 0:
                for task in config['tasks']:
                    logger.info(f"  - Train {task} task ({config['epochs_per_task']} epochs)")
            else:
                logger.info(f"  - {config['max_iterations']} iterations of multi-task training")
            logger.info(f"  - Validation threshold: {config['threshold']:.1%}")
        return
    
    # Execute pipeline
    try:
        pipeline_start_time = time.time()
        
        final_model_path = orchestrator.execute_full_pipeline(
            sft_model_path=base_model,
            start_stage=args.start_stage
        )
        
        total_duration = time.time() - pipeline_start_time
        
        if final_model_path:
            logger.info(f"🎉 Pipeline completed successfully in {total_duration/3600:.2f} hours")
            logger.info(f"Production model ready: {final_model_path}")
        else:
            logger.error("❌ Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

if __name__ == '__main__':
    main()