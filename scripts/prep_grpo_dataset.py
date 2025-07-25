#!/usr/bin/env python3
"""
GRPO Dataset Preparation Script

Generates multiple candidate responses per query for GRPO training.
This script is required for process supervision and group relative policy optimization.

Usage:
    python prep_grpo_dataset.py --task bluebook --model_path models/sft --num_candidates 4
    python prep_grpo_dataset.py --task all --model_path models/sft --subset 1000
"""

import json
import argparse
import pathlib
from typing import List, Dict, Any
import sys
import os
from tqdm import tqdm
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Using mock generation mode.")

from rewards import UnifiedRewardFunction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GRPODatasetGenerator:
    """Generates GRPO training datasets with multiple scored responses per query"""
    
    def __init__(self, model_path: str, faiss_index_path: str = None):
        """
        Initialize GRPO dataset generator.
        
        Args:
            model_path: Path to SFT model for response generation
            faiss_index_path: Path to FAISS index for retrieval task
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize reward function
        self.reward_fn = UnifiedRewardFunction(faiss_index_path=faiss_index_path)
        
        logger.info(f"GRPO Generator initialized with model: {model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"FAISS index: {faiss_index_path or 'Not provided'}")
    
    def load_model(self):
        """Load the SFT model for response generation"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - using mock mode")
            return
            
        try:
            logger.info("Loading model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to mock generation mode")
            self.model = None
            self.tokenizer = None
    
    def generate_responses(self, query: str, num_candidates: int = 4) -> List[str]:
        """
        Generate multiple candidate responses for a single query.
        
        Args:
            query: Input query/prompt
            num_candidates: Number of responses to generate
            
        Returns:
            List of candidate response strings
        """
        if self.model is None or not TRANSFORMERS_AVAILABLE:
            # Mock responses for testing/development
            return [
                f"Mock response {i+1} for query",
                f"Alternative mock response {i+1}",
                f"Another mock answer {i+1}",
                f"Different mock completion {i+1}"
            ][:num_candidates]
        
        candidates = []
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(query, return_tensors='pt', truncate=True, max_length=2048)
            inputs = inputs.to(self.device)
            
            # Generate multiple responses with different sampling parameters
            generation_configs = [
                {"temperature": 0.7, "top_p": 0.9, "do_sample": True},
                {"temperature": 0.8, "top_p": 0.85, "do_sample": True},
                {"temperature": 0.6, "top_p": 0.95, "do_sample": True},
                {"temperature": 0.9, "top_p": 0.8, "do_sample": True}
            ]
            
            for i in range(num_candidates):
                config = generation_configs[i % len(generation_configs)]
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=200,  # Reasonable length for legal responses
                        pad_token_id=self.tokenizer.eos_token_id,
                        **config
                    )
                
                # Decode only the new tokens (response)
                response = self.tokenizer.decode(
                    outputs[0][inputs.shape[-1]:], 
                    skip_special_tokens=True
                ).strip()
                
                candidates.append(response)
                
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            # Fallback to mock responses
            candidates = [f"Fallback response {i+1}" for i in range(num_candidates)]
        
        return candidates
    
    def enhance_prompt(self, original_prompt: str, task_type: str) -> str:
        """
        Enhance prompts to include complete choice sets, fixing ground truth vs choice mismatches.
        
        Args:
            original_prompt: Original prompt from SFT dataset
            task_type: Task type ('entail', 'holding', etc.)
            
        Returns:
            Enhanced prompt with complete choice set
        """
        if task_type == 'entail':
            # Fix incomplete entail task choices - original missing FOLLOWS and CITES_POSITIVELY
            old_choices = """Choose from:
- OVERRULES: The citing case overrules or abrogates the cited case
- DISTINGUISHES: The citing case distinguishes itself from the cited case  
- AFFIRMS: The citing case affirms or follows the cited case
- NONE: No clear relationship is established"""
            
            new_choices = """Choose from:
- OVERRULES: The citing case overrules or abrogates the cited case
- DISTINGUISHES: The citing case distinguishes itself from the cited case  
- AFFIRMS: The citing case affirms or follows the cited case
- FOLLOWS: The citing case follows the precedent established by the cited case
- CITES_POSITIVELY: The citing case cites the cited case in support of its reasoning
- NONE: No clear relationship is established"""
            
            if old_choices in original_prompt:
                return original_prompt.replace(old_choices, new_choices)
        
        # Add enhancements for other task types as needed
        # TODO: Audit holding, bluebook, summarise, retrieval tasks
        
        return original_prompt
    
    def process_task_file(self, task_file: pathlib.Path, num_candidates: int = 4, subset_size: int = None) -> List[Dict[str, Any]]:
        """
        Process a task file and generate GRPO dataset.
        
        Args:
            task_file: Path to task JSONL file
            num_candidates: Number of response candidates per query
            subset_size: Limit to first N samples (for development)
            
        Returns:
            List of GRPO training samples
        """
        logger.info(f"Processing {task_file}")
        
        grpo_samples = []
        
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if subset_size:
                lines = lines[:subset_size]
                logger.info(f"Using subset of {len(lines)} samples")
            
            for idx, line in enumerate(tqdm(lines, desc=f"Processing {task_file.name}")):
                try:
                    sample = json.loads(line.strip())
                    query = sample.get('inputs', '')
                    
                    if not query:
                        logger.warning(f"Empty query at line {idx+1}, skipping")
                        continue
                    
                    # Auto-detect task type from file path
                    task_type = None
                    if 'bluebook' in str(task_file):
                        task_type = 'bluebook'
                    elif 'holding' in str(task_file):
                        task_type = 'holding'
                    elif 'summarise' in str(task_file):
                        task_type = 'summarise'
                    elif 'retrieval' in str(task_file):
                        task_type = 'retrieval'
                    elif 'entail' in str(task_file):
                        task_type = 'entail'
                    
                    # Enhance prompt to include complete choice sets
                    enhanced_query = self.enhance_prompt(query, task_type) if task_type else query
                    
                    # Generate multiple candidate responses
                    candidates = self.generate_responses(enhanced_query, num_candidates)
                    
                    # Score each candidate using unified reward function
                    scores = []
                    for candidate in candidates:
                        try:
                            # Use original sample for reward computation to ensure correct field format
                            score = self.reward_fn.reward(sample, candidate, task_type=task_type)
                            scores.append(float(score))
                            
                        except Exception as e:
                            logger.warning(f"Error scoring candidate: {e}")
                            scores.append(0.0)
                    
                    grpo_sample = {
                        'query': enhanced_query,
                        'responses': candidates,
                        'scores': scores,
                        'metadata': sample.get('metadata', {}),
                        'original_sample': sample,
                        'sample_id': sample.get('case_id') or sample.get('pair_id') or f"sample_{idx}"
                    }
                    
                    grpo_samples.append(grpo_sample)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at line {idx+1}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {idx+1}: {e}")
                    continue
                    
        except FileNotFoundError:
            logger.error(f"Task file not found: {task_file}")
            return []
        
        logger.info(f"Generated {len(grpo_samples)} GRPO samples from {task_file.name}")
        return grpo_samples
    
    def save_grpo_dataset(self, grpo_samples: List[Dict[str, Any]], output_file: pathlib.Path):
        """Save GRPO dataset to JSON file"""
        try:
            # Calculate statistics
            total_samples = len(grpo_samples)
            avg_max_score = sum(max(sample['scores']) for sample in grpo_samples) / total_samples if total_samples > 0 else 0
            avg_score_range = sum(max(sample['scores']) - min(sample['scores']) for sample in grpo_samples) / total_samples if total_samples > 0 else 0
            
            # Create metadata
            metadata = {
                'total_samples': total_samples,
                'num_candidates_per_query': len(grpo_samples[0]['responses']) if grpo_samples else 0,
                'avg_max_score': avg_max_score,
                'avg_score_range': avg_score_range,
                'generation_model': self.model_path
            }
            
            output_data = {
                'metadata': metadata,
                'samples': grpo_samples
            }
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"GRPO dataset saved to {output_file}")
            logger.info(f"Dataset stats: {total_samples} samples, avg max score: {avg_max_score:.3f}, avg range: {avg_score_range:.3f}")
            
        except Exception as e:
            logger.error(f"Error saving GRPO dataset: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate GRPO training dataset')
    parser.add_argument('--task', required=True, 
                       choices=['bluebook', 'holding', 'summarise', 'retrieval', 'entail', 'all'],
                       help='Task type to process or "all" for all tasks')
    parser.add_argument('--model_path', required=True, 
                       help='Path to SFT model for response generation')
    parser.add_argument('--num_candidates', type=int, default=4,
                       help='Number of response candidates per query (default: 4)')
    parser.add_argument('--subset', type=int, default=None,
                       help='Process only first N samples (for development/testing)')
    parser.add_argument('--faiss_index', type=str, default=None,
                       help='Path to FAISS index for retrieval task')
    parser.add_argument('--data_root', type=str, default='../data_tasks',
                       help='Root directory for task data (default: ../data_tasks)')
    parser.add_argument('--output_root', type=str, default='../data_grpo',
                       help='Root directory for GRPO output (default: ../data_grpo)')
    parser.add_argument('--mock_mode', action='store_true',
                       help='Use mock responses instead of actual model generation')
    parser.add_argument('--unified_output', action='store_true',
                       help='Create unified multi-task dataset for Stage 1+ training')
    
    args = parser.parse_args()
    
    # Resolve paths
    data_root = pathlib.Path(args.data_root)
    output_root = pathlib.Path(args.output_root)
    
    # Auto-detect FAISS index if not provided
    if args.faiss_index is None and args.task in ['retrieval', 'all']:
        potential_faiss = data_root / 'retrieval' / 'embeddings.faiss'
        if potential_faiss.exists():
            args.faiss_index = str(potential_faiss)
            logger.info(f"Auto-detected FAISS index: {args.faiss_index}")
    
    # Initialize generator
    generator = GRPODatasetGenerator(
        model_path=args.model_path,
        faiss_index_path=args.faiss_index
    )
    
    # Load model unless in mock mode
    if not args.mock_mode:
        generator.load_model()
    
    # Define tasks to process
    if args.task == 'all':
        tasks = ['bluebook', 'holding', 'summarise', 'retrieval', 'entail']
    else:
        tasks = [args.task]
    
    # Process each task
    all_grpo_samples = []  # For unified dataset
    
    for task in tasks:
        logger.info(f"Processing task: {task}")
        
        # Find task file
        task_file = data_root / task / 'train.jsonl'
        if not task_file.exists():
            logger.warning(f"Task file not found: {task_file}")
            continue
        
        # Generate GRPO dataset
        grpo_samples = generator.process_task_file(
            task_file, 
            num_candidates=args.num_candidates,
            subset_size=args.subset
        )
        
        if not grpo_samples:
            logger.warning(f"No GRPO samples generated for task: {task}")
            continue
        
        # Add task identifier to samples for unified dataset
        if args.unified_output or args.task == 'all':
            for sample in grpo_samples:
                sample['task_type'] = task
            all_grpo_samples.extend(grpo_samples)
        
        # Save individual task dataset (always)
        output_file = output_root / task / 'train_grpo.json'
        generator.save_grpo_dataset(grpo_samples, output_file)
    
    # Create unified multi-task dataset if requested
    if (args.unified_output or args.task == 'all') and all_grpo_samples:
        logger.info(f"Creating unified multi-task dataset with {len(all_grpo_samples)} samples")
        
        # Shuffle samples for better task mixing
        import random
        random.shuffle(all_grpo_samples)
        
        # Save unified dataset
        unified_output_file = output_root / 'unified' / 'train_grpo.json'
        generator.save_grpo_dataset(all_grpo_samples, unified_output_file)
        
        # Create balanced evaluation dataset (smaller subset)
        eval_samples = all_grpo_samples[:min(len(all_grpo_samples) // 10, 1000)]
        eval_output_file = output_root / 'unified' / 'eval_grpo.json'
        generator.save_grpo_dataset(eval_samples, eval_output_file)
        
        logger.info(f"Unified datasets created: {len(all_grpo_samples)} train, {len(eval_samples)} eval")
    
    logger.info("GRPO dataset generation completed")

if __name__ == '__main__':
    main()