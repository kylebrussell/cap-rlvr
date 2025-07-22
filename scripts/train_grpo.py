#!/usr/bin/env python3
"""
GRPO Training Script for CAP RLVR Legal Reasoning

This script implements Group Relative Policy Optimization (GRPO) training
for legal reasoning tasks using the TRL library.

Usage:
    python train_grpo.py --task bluebook --model_path models/sft --data_path data_grpo/bluebook/train_grpo.json
    python train_grpo.py --task all --model_path models/sft --multi_task
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import warnings

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    set_seed
)

# TRL imports
try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    warnings.warn("TRL library not available. Please install: pip install trl")

# Add current directory to path for reward functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rewards import UnifiedRewardFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grpo_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GRPOLegalTrainer:
    """GRPO trainer specialized for legal reasoning tasks"""
    
    def __init__(
        self,
        model_path: str,
        task_name: str = None,
        faiss_index_path: str = None,
        output_dir: str = "models/grpo",
        device: str = None
    ):
        """
        Initialize GRPO Legal Trainer.
        
        Args:
            model_path: Path to fine-tuned SFT model
            task_name: Specific task name (bluebook, holding, etc.) or None for multi-task
            faiss_index_path: Path to FAISS index for retrieval tasks
            output_dir: Directory to save trained model
            device: Device to use for training ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.task_name = task_name
        self.output_dir = Path(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize reward function
        self.reward_fn = UnifiedRewardFunction(faiss_index_path=faiss_index_path)
        
        # Model and tokenizer will be loaded during training
        self.model = None
        self.tokenizer = None
        
        logger.info(f"GRPO Legal Trainer initialized")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Task: {task_name or 'multi-task'}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {output_dir}")
    
    def load_model_and_tokenizer(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info("Loading model and tokenizer...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model with appropriate settings for training
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Enable gradient computation
            self.model.train()
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model and tokenizer: {e}")
            raise
    
    def prepare_dataset(self, grpo_data: Dict[str, Any]) -> Dataset:
        """
        Convert GRPO dataset format to HuggingFace Dataset format.
        
        Args:
            grpo_data: GRPO dataset with metadata and samples
            
        Returns:
            HuggingFace Dataset ready for GRPO training
        """
        logger.info("Preparing dataset for GRPO training...")
        
        samples = grpo_data.get('samples', [])
        if not samples:
            raise ValueError("No samples found in GRPO dataset")
            
        # Convert to format expected by GRPOTrainer
        dataset_samples = []
        
        for sample in samples:
            # Each sample should have query, responses, and scores
            query = sample.get('query', '')
            responses = sample.get('responses', [])
            scores = sample.get('scores', [])
            
            if not query or not responses or not scores:
                logger.warning(f"Skipping incomplete sample: {sample.get('sample_id', 'unknown')}")
                continue
                
            if len(responses) != len(scores):
                logger.warning(f"Responses/scores length mismatch in sample: {sample.get('sample_id', 'unknown')}")
                continue
            
            # Format for GRPO trainer
            dataset_sample = {
                'query': query,
                'responses': responses,
                'scores': scores,
                'sample_id': sample.get('sample_id', ''),
                'metadata': sample.get('metadata', {})
            }
            
            dataset_samples.append(dataset_sample)
        
        logger.info(f"Prepared {len(dataset_samples)} samples for training")
        
        return Dataset.from_list(dataset_samples)
    
    def create_reward_function(self) -> Callable:
        """
        Create reward function for GRPO trainer.
        
        Returns:
            Callable reward function compatible with GRPO trainer
        """
        def reward_function(samples: List[Dict], **kwargs) -> List[List[float]]:
            """
            Compute rewards for a batch of samples.
            
            Args:
                samples: List of sample dictionaries with 'query' and 'responses'
                
            Returns:
                List of reward lists (one list of scores per sample)
            """
            batch_rewards = []
            
            for sample in samples:
                query = sample.get('query', '')
                responses = sample.get('responses', [])
                
                if not responses:
                    batch_rewards.append([0.0])
                    continue
                
                sample_rewards = []
                for response in responses:
                    try:
                        # Use unified reward function with auto task detection
                        reward = self.reward_fn.reward(
                            {'inputs': query, 'metadata': sample.get('metadata', {})},
                            response,
                            task_type=self.task_name
                        )
                        sample_rewards.append(float(reward))
                    except Exception as e:
                        logger.warning(f"Error computing reward for response: {e}")
                        sample_rewards.append(0.0)
                
                batch_rewards.append(sample_rewards)
            
            return batch_rewards
        
        return reward_function
    
    def setup_training_config(self, **kwargs) -> GRPOConfig:
        """
        Setup GRPO training configuration.
        
        Args:
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured GRPOConfig object
        """
        if not TRL_AVAILABLE:
            raise ImportError("TRL library is required for GRPO training")
            
        # Default configuration optimized for legal reasoning
        default_config = {
            'output_dir': str(self.output_dir / f"{self.task_name or 'multi_task'}_grpo"),
            'per_device_train_batch_size': 2,  # Conservative for memory
            'per_device_eval_batch_size': 4,
            'num_train_epochs': 3,
            'learning_rate': 1e-5,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'gradient_accumulation_steps': 8,
            'dataloader_num_workers': 4,
            'remove_unused_columns': False,
            'save_strategy': 'steps',
            'save_steps': 500,
            'eval_strategy': 'steps',
            'eval_steps': 500,
            'logging_strategy': 'steps',
            'logging_steps': 50,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval/rewards/mean',
            'greater_is_better': True,
            'seed': 42,
            'data_seed': 42,
            'bf16': True,  # Use bf16 for better numerical stability
            'report_to': None,  # Disable wandb by default
            'push_to_hub': False,
            
            # GRPO-specific parameters
            'temperature': 0.7,
            'beta': 0.1,  # KL penalty coefficient
            'num_generations': 4,  # Number of responses to generate per query
            'max_prompt_length': 1024,
            'max_completion_length': 512,
        }
        
        # Override with provided kwargs
        config_dict = {**default_config, **kwargs}
        
        # Create output directory
        Path(config_dict['output_dir']).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"GRPO Config: {json.dumps(config_dict, indent=2)}")
        
        return GRPOConfig(**config_dict)
    
    def train(
        self, 
        grpo_dataset_path: str,
        eval_dataset_path: str = None,
        **training_kwargs
    ) -> GRPOTrainer:
        """
        Execute GRPO training.
        
        Args:
            grpo_dataset_path: Path to GRPO training dataset JSON file
            eval_dataset_path: Optional path to evaluation dataset
            **training_kwargs: Additional training configuration parameters
            
        Returns:
            Trained GRPOTrainer instance
        """
        if not TRL_AVAILABLE:
            raise ImportError("TRL library is required for GRPO training")
            
        logger.info("Starting GRPO training...")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load training dataset
        logger.info(f"Loading training dataset from: {grpo_dataset_path}")
        with open(grpo_dataset_path, 'r', encoding='utf-8') as f:
            grpo_data = json.load(f)
        
        train_dataset = self.prepare_dataset(grpo_data)
        
        # Load evaluation dataset if provided
        eval_dataset = None
        if eval_dataset_path and Path(eval_dataset_path).exists():
            logger.info(f"Loading evaluation dataset from: {eval_dataset_path}")
            with open(eval_dataset_path, 'r', encoding='utf-8') as f:
                eval_grpo_data = json.load(f)
            eval_dataset = self.prepare_dataset(eval_grpo_data)
        
        # Setup training configuration
        training_config = self.setup_training_config(**training_kwargs)
        
        # Create reward function
        reward_function = self.create_reward_function()
        
        # Initialize GRPO trainer
        logger.info("Initializing GRPO trainer...")
        trainer = GRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reward_funcs=reward_function,
        )
        
        # Add custom callbacks for legal-specific metrics
        self.add_legal_callbacks(trainer)
        
        # Start training
        logger.info("Beginning GRPO training...")
        try:
            training_result = trainer.train()
            logger.info("GRPO training completed successfully")
            
            # Save final model
            final_output_dir = training_config.output_dir
            trainer.save_pretrained(final_output_dir)
            self.tokenizer.save_pretrained(final_output_dir)
            
            # Save training metadata
            metadata = {
                'task_name': self.task_name,
                'model_path': self.model_path,
                'training_result': training_result.__dict__ if hasattr(training_result, '__dict__') else str(training_result),
                'dataset_size': len(train_dataset),
                'eval_dataset_size': len(eval_dataset) if eval_dataset else 0,
                'training_config': training_config.__dict__ if hasattr(training_config, '__dict__') else {}
            }
            
            with open(Path(final_output_dir) / 'training_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to: {final_output_dir}")
            return trainer
            
        except Exception as e:
            logger.error(f"GRPO training failed: {e}")
            raise
    
    def add_legal_callbacks(self, trainer: GRPOTrainer):
        """Add legal-specific training callbacks"""
        from transformers import TrainerCallback
        
        class LegalMetricsCallback(TrainerCallback):
            def __init__(self, reward_fn, task_name):
                self.reward_fn = reward_fn
                self.task_name = task_name
                
            def on_evaluate(self, args, state, control, model, tokenizer, eval_dataloader, **kwargs):
                """Log legal-specific metrics during evaluation"""
                logger.info(f"Evaluation step {state.global_step} - Task: {self.task_name}")
                
            def on_log(self, args, state, control, logs=None, **kwargs):
                """Enhanced logging for legal reasoning metrics"""
                if logs and 'eval/rewards/mean' in logs:
                    logger.info(f"Step {state.global_step} - Mean Reward: {logs['eval/rewards/mean']:.4f}")
        
        trainer.add_callback(LegalMetricsCallback(self.reward_fn, self.task_name))

def main():
    parser = argparse.ArgumentParser(description='GRPO training for legal reasoning')
    
    parser.add_argument('--task', required=True, 
                       choices=['bluebook', 'holding', 'summarise', 'retrieval', 'entail', 'all'],
                       help='Task to train on')
    parser.add_argument('--model_path', required=True,
                       help='Path to SFT model')
    parser.add_argument('--data_path', required=True,
                       help='Path to GRPO dataset JSON file')
    parser.add_argument('--eval_data_path', default=None,
                       help='Path to evaluation GRPO dataset JSON file')
    parser.add_argument('--output_dir', default='models/grpo',
                       help='Output directory for trained model')
    parser.add_argument('--faiss_index', default=None,
                       help='Path to FAISS index for retrieval task')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size per device')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='GRPO KL penalty coefficient')
    parser.add_argument('--multi_task', action='store_true',
                       help='Train on multiple tasks (requires task=all)')
    parser.add_argument('--resume_from_checkpoint', default=None,
                       help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.task == 'all' and not args.multi_task:
        parser.error("--multi_task flag required when task=all")
    
    if not Path(args.model_path).exists():
        parser.error(f"Model path does not exist: {args.model_path}")
    
    if not Path(args.data_path).exists():
        parser.error(f"Data path does not exist: {args.data_path}")
    
    # Auto-detect FAISS index for retrieval task
    if args.task in ['retrieval', 'all'] and args.faiss_index is None:
        potential_faiss = Path('data_tasks/retrieval/embeddings.faiss')
        if potential_faiss.exists():
            args.faiss_index = str(potential_faiss)
            logger.info(f"Auto-detected FAISS index: {args.faiss_index}")
    
    # Initialize trainer
    task_name = None if args.multi_task else args.task
    trainer = GRPOLegalTrainer(
        model_path=args.model_path,
        task_name=task_name,
        faiss_index_path=args.faiss_index,
        output_dir=args.output_dir
    )
    
    # Training configuration
    training_config = {
        'per_device_train_batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_train_epochs': args.num_epochs,
        'beta': args.beta,
    }
    
    if args.resume_from_checkpoint:
        training_config['resume_from_checkpoint'] = args.resume_from_checkpoint
    
    # Start training
    try:
        grpo_trainer = trainer.train(
            grpo_dataset_path=args.data_path,
            eval_dataset_path=args.eval_data_path,
            **training_config
        )
        logger.info("GRPO training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main()