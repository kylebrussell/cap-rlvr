#!/usr/bin/env python3
"""
Create GRPO input files from HuggingFace dataset

Converts the HF SFT dataset to individual task JSONL files 
for use with the regular prep_grpo_dataset.py script.
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset
import random

def create_grpo_input_files(dataset_name: str, output_dir: str, subset_size: int = None):
    """
    Create individual task JSONL files from HF dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Output directory for JSONL files
        subset_size: Optional limit on samples per task
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)['train']
    
    # Group samples by task
    task_samples = {}
    for sample in dataset:
        task = sample['task']
        if task not in task_samples:
            task_samples[task] = []
        task_samples[task].append(sample)
    
    print("Creating JSONL files by task:")
    for task, samples in task_samples.items():
        print(f"  {task}: {len(samples)} samples")
        
        # Apply subset limit if specified
        if subset_size and len(samples) > subset_size:
            samples = random.sample(samples, subset_size)
            print(f"    Reduced to {len(samples)} samples (subset)")
        
        # Create task directory and file
        task_dir = output_path / task
        task_dir.mkdir(exist_ok=True)
        
        output_file = task_dir / f"{task}_subset.jsonl"
        
        with open(output_file, 'w') as f:
            for sample in samples:
                # Convert to format expected by regular script
                jsonl_sample = {
                    'inputs': sample['prompt'],
                    'expected_completion': sample['completion'],
                    'task': sample['task'],
                    'metadata': {}
                }
                f.write(json.dumps(jsonl_sample) + '\n')
        
        print(f"    Created: {output_file}")
    
    print(f"\nGRPO input files created in: {output_dir}")
    print("Files are ready for use with prep_grpo_dataset.py")

def main():
    parser = argparse.ArgumentParser(description='Create GRPO input files from HF dataset')
    parser.add_argument('--dataset', default='kylebrussell/cap-rlvr-sft',
                       help='HuggingFace dataset name (default: kylebrussell/cap-rlvr-sft)')
    parser.add_argument('--output_dir', default='data_grpo_input',
                       help='Output directory for JSONL files (default: data_grpo_input)')
    parser.add_argument('--subset_size', type=int, default=200,
                       help='Max samples per task (default: 200 for fast testing)')
    
    args = parser.parse_args()
    
    create_grpo_input_files(args.dataset, args.output_dir, args.subset_size)

if __name__ == '__main__':
    main()