#!/usr/bin/env python3
"""
GRPO Dataset Preparation Script - HuggingFace Version

Generates multiple candidate responses per query for GRPO training using HuggingFace datasets.
This script is required for process supervision and group relative policy optimization.

Usage:
    python prep_grpo_dataset_hf.py --task all --model_path models/sft_qwen3_14b_lora_30k --num_candidates 4 --subset 1000
"""

import json
import argparse
import pathlib
from typing import List, Dict, Any
import sys
import os
from tqdm import tqdm
import logging
import random

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from datasets import load_dataset
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
        if self.model is None or self.tokenizer is None:
            # Mock generation for testing
            return [f"Mock response {i+1} for query" for i in range(num_candidates)]
        
        responses = []
        
        try:
            # Tokenize input
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate multiple responses with different sampling parameters
            generation_configs = [
                {"temperature": 0.7, "do_sample": True, "top_p": 0.9},
                {"temperature": 0.9, "do_sample": True, "top_p": 0.95},
                {"temperature": 1.0, "do_sample": True, "top_k": 50},
                {"temperature": 0.8, "do_sample": True, "top_p": 0.85}
            ]
            
            with torch.no_grad():
                for i in range(num_candidates):
                    config = generation_configs[i % len(generation_configs)]
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **config
                    )
                    
                    # Decode only the generated part (excluding input)
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    responses.append(generated_text.strip())
                    
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            # Fallback to mock responses
            return [f"Error response {i+1}" for i in range(num_candidates)]
        
        return responses
    
    def process_hf_dataset(self, task_filter: str = 'all', num_candidates: int = 4, subset: int = None) -> List[Dict[str, Any]]:
        """
        Process HuggingFace dataset to generate GRPO samples.
        
        Args:
            task_filter: Task to process ('all' or specific task)
            num_candidates: Number of response candidates per query
            subset: Limit to first N samples
            
        Returns:
            List of GRPO samples with multiple scored responses
        """
        logger.info("Loading HuggingFace dataset...")
        dataset = load_dataset('kylebrussell/cap-rlvr-sft')['train']
        
        # Filter by task if specified
        if task_filter != 'all':
            dataset = [sample for sample in dataset if sample['task'] == task_filter]
            logger.info(f"Filtered to {len(dataset)} samples for task: {task_filter}")
        
        # Apply subset limit
        if subset:
            dataset = random.sample(list(dataset), min(subset, len(dataset)))
            logger.info(f"Using subset of {len(dataset)} samples")
        
        grpo_samples = []
        
        logger.info(f"Processing {len(dataset)} samples...")
        for idx, sample in enumerate(tqdm(dataset)):
            try:
                query = sample['prompt']
                expected = sample['completion']
                task_type = sample['task']
                
                # Generate multiple candidate responses
                candidates = self.generate_responses(query, num_candidates)
                
                # Score each candidate
                scores = []
                for candidate in candidates:
                    try:
                        # Create a mock sample for reward function
                        mock_sample = {
                            'task': task_type,
                            'prompt': query,
                            'completion': expected
                        }
                        
                        score = self.reward_fn.reward(mock_sample, candidate, task_type=task_type)
                        scores.append(float(score))
                        
                    except Exception as e:
                        logger.warning(f"Error scoring candidate: {e}")
                        scores.append(0.0)
                
                grpo_sample = {
                    'query': query,
                    'responses': candidates,
                    'scores': scores,
                    'task': task_type,
                    'expected_completion': expected,
                    'sample_id': f"hf_sample_{idx}"
                }
                
                grpo_samples.append(grpo_sample)
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue
        
        logger.info(f"Generated {len(grpo_samples)} GRPO samples")
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
                'generation_model': self.model_path,
                'dataset_source': 'kylebrussell/cap-rlvr-sft'
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
    parser = argparse.ArgumentParser(description='Generate GRPO training dataset from HuggingFace')
    parser.add_argument('--task', default='all',
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
    parser.add_argument('--output_root', type=str, default='data_grpo',
                       help='Root directory for GRPO output (default: data_grpo)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = GRPODatasetGenerator(args.model_path, args.faiss_index)
    generator.load_model()
    
    # Process dataset
    grpo_samples = generator.process_hf_dataset(
        task_filter=args.task,
        num_candidates=args.num_candidates,
        subset=args.subset
    )
    
    if not grpo_samples:
        logger.error("No GRPO samples generated - exiting")
        return
    
    # Determine output file
    output_root = pathlib.Path(args.output_root)
    if args.task == 'all':
        output_file = output_root / 'unified' / 'train_grpo.json'
    else:
        output_file = output_root / args.task / 'train_grpo.json'
    
    # Save dataset
    generator.save_grpo_dataset(grpo_samples, output_file)
    
    logger.info("GRPO dataset generation completed successfully")

if __name__ == '__main__':
    main()