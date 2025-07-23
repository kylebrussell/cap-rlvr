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
from model_utils import generate_output_path, extract_model_size

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
        
        # Generate model-size-aware output directory
        if task_name:
            self.output_dir = generate_output_path(output_dir, model_path, task_name)
        else:
            self.output_dir = generate_output_path(output_dir, model_path, stage="multi_task")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize reward function
        self.reward_fn = UnifiedRewardFunction(faiss_index_path=faiss_index_path)
        
        # Model and tokenizer will be loaded during training
        self.model = None
        self.tokenizer = None
        
        # Extract model size for logging
        model_size = extract_model_size(model_path)
        size_info = f" ({model_size})" if model_size else ""
        
        logger.info(f"GRPO Legal Trainer initialized")
        logger.info(f"Model path: {model_path}{size_info}")
        logger.info(f"Task: {task_name or 'multi-task'}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
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
            # Each sample should have prompt, chosen, and rejected
            prompt = sample.get('prompt', '')
            chosen = sample.get('chosen', '')
            rejected = sample.get('rejected', '')
            
            if not prompt or not chosen:
                logger.warning(f"Skipping incomplete sample: {sample.get('sample_id', 'unknown')}")
                continue
            
            # Format for GRPO trainer
            dataset_sample = {
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
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
        eval_only: bool = False,
        **training_kwargs
    ) -> GRPOTrainer:
        """
        Execute GRPO training or evaluation.
        
        Args:
            grpo_dataset_path: Path to GRPO training dataset JSON file
            eval_dataset_path: Optional path to evaluation dataset
            eval_only: If True, run evaluation only without training
            **training_kwargs: Additional training configuration parameters
            
        Returns:
            GRPOTrainer instance or evaluation results
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
        
        # Check if it's a HuggingFace dataset or local file
        if grpo_dataset_path.startswith('kylebrussell/') or '/' in grpo_dataset_path and not os.path.exists(grpo_dataset_path):
            # Load from HuggingFace
            from datasets import load_dataset
            hf_dataset = load_dataset(grpo_dataset_path)['train']
            
            # Filter by task if specified (only for unified datasets, not task-specific ones)
            if self.task_name and 'cap-rlvr-sft' in grpo_dataset_path:
                # Only filter if using the unified SFT dataset
                grpo_data = [sample for sample in hf_dataset if sample.get('task') == self.task_name]
            else:
                # For task-specific datasets, use reasonable sample size
                grpo_data = list(hf_dataset)[:50]  # Limit for memory efficiency
                
            # Convert HuggingFace format to GRPO format
            formatted_samples = []
            for sample in grpo_data:
                if isinstance(sample, dict):
                    # Extract data from HuggingFace format
                    query = sample.get('inputs', sample.get('prompt', ''))
                    ground_truth = sample.get('ground_truth', sample.get('completion', ''))
                    sample_id = sample.get('case_id', sample.get('sample_id', ''))
                    metadata = sample.get('metadata', {})
                    
                    # Convert to GRPO format (prompt/chosen/rejected)
                    if query and ground_truth:
                        formatted_samples.append({
                            'prompt': query,
                            'chosen': ground_truth,
                            'rejected': "I don't know.",  # Generic rejected response
                            'sample_id': sample_id,
                            'metadata': metadata
                        })
                        
            # Wrap in expected dictionary format
            grpo_data = {'samples': formatted_samples}
        else:
            # Load from local JSON file
            with open(grpo_dataset_path, 'r', encoding='utf-8') as f:
                grpo_data = json.load(f)
        
        train_dataset = self.prepare_dataset(grpo_data)
        
        # For evaluation-only mode, use the same dataset as eval dataset
        if eval_only:
            eval_dataset = train_dataset
            logger.info("Using training dataset as evaluation dataset for eval-only mode")
        else:
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
        
        # Use sequential approach for both training and evaluation to save memory
        logger.info("Using sequential GRPO approach to save GPU memory...")
        return self.run_sequential_grpo(
            train_dataset, eval_dataset, reward_function, eval_only, **training_kwargs
        )
        
    def run_sequential_grpo(self, train_dataset, eval_dataset, reward_function, eval_only, **training_kwargs):
        """
        Memory-efficient sequential GRPO that loads reference and main models separately.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset  
            reward_function: Reward function
            eval_only: Whether to run evaluation only
            **training_kwargs: Additional training arguments
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting sequential GRPO approach...")
        
        # Use eval dataset for evaluation, train dataset for training
        dataset_to_use = eval_dataset if eval_only else train_dataset
        
        if not dataset_to_use:
            raise ValueError("No dataset provided for sequential GRPO")
            
        # Step 1: Generate reference model responses
        logger.info("Phase 1: Generating reference model responses...")
        reference_outputs = self.generate_reference_responses(dataset_to_use)
        
        # Clear GPU memory
        import torch
        torch.cuda.empty_cache()
        
        # Step 2: Generate main model responses and compute metrics
        logger.info("Phase 2: Generating main model responses and computing metrics...")
        results = self.evaluate_with_reference(dataset_to_use, reference_outputs, reward_function)
        
        if eval_only:
            return {'eval_results': results}
        else:
            # For training mode, we'd implement training steps here
            logger.info("Training mode not yet implemented in sequential approach")
            return {'eval_results': results}
    
    def generate_reference_responses(self, dataset):
        """Generate responses using the reference model (same as base model)."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("Loading reference model...")
        # Use the same model as reference (frozen)
        ref_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if ref_tokenizer.pad_token is None:
            ref_tokenizer.pad_token = ref_tokenizer.eos_token
            
        reference_outputs = {}
        
        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                logger.info(f"Generating reference responses: {i}/{len(dataset)}")
                
            prompt = sample['prompt']
            
            # Generate reference response
            inputs = ref_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(ref_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = ref_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=ref_tokenizer.eos_token_id
                )
            
            # Decode reference response
            ref_response = ref_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            reference_outputs[i] = {
                'prompt': prompt,
                'response': ref_response,
                'sample_id': sample.get('sample_id', ''),
                'metadata': sample.get('metadata', {})
            }
        
        # Clean up reference model
        del ref_model
        del ref_tokenizer
        torch.cuda.empty_cache()
        
        logger.info(f"Generated {len(reference_outputs)} reference responses")
        return reference_outputs
    
    def evaluate_with_reference(self, dataset, reference_outputs, reward_function):
        """Evaluate main model against reference outputs."""
        import torch
        
        logger.info("Evaluating main model responses...")
        
        # Model should already be loaded
        if not hasattr(self, 'model') or self.model is None:
            self.load_model_and_tokenizer()
        
        eval_rewards = []
        
        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                logger.info(f"Evaluating main model: {i}/{len(dataset)}")
                
            prompt = sample['prompt']
            ground_truth = sample['chosen']
            
            # Generate main model response
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode main model response
            main_response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Create reward sample
            reward_sample = {
                'inputs': prompt,
                'ground_truth': ground_truth,
                'metadata': sample.get('metadata', {})
            }
            
            # Compute reward for main model response
            try:
                reward = reward_function.reward(reward_sample, main_response, self.task_name)
                eval_rewards.append(float(reward))
            except Exception as e:
                logger.warning(f"Error computing reward for sample {i}: {e}")
                eval_rewards.append(0.0)
        
        # Compute evaluation metrics
        if eval_rewards:
            mean_reward = sum(eval_rewards) / len(eval_rewards)
            reward_std = (sum((r - mean_reward) ** 2 for r in eval_rewards) / len(eval_rewards)) ** 0.5
        else:
            mean_reward = 0.0
            reward_std = 0.0
        
        results = {
            'eval/rewards/mean': mean_reward,
            'eval/rewards/std': reward_std,
            'eval/num_samples': len(eval_rewards)
        }
        
        logger.info(f"Sequential evaluation complete: mean_reward={mean_reward:.3f}, std={reward_std:.3f}")
        return results
    
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
    parser.add_argument('--eval_only', action='store_true',
                       help='Run evaluation only without training')
    
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
    
    # Start training or evaluation
    try:
        result = trainer.train(
            grpo_dataset_path=args.data_path,
            eval_dataset_path=args.eval_data_path,
            eval_only=args.eval_only,
            **training_config
        )
        
        if args.eval_only:
            logger.info("GRPO evaluation completed successfully!")
            logger.info(f"Results: {result}")
        else:
            logger.info("GRPO training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main()