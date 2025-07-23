# Simple SFT Evaluation Script

## Overview
The simple_eval_sft.py script evaluates CAP RLVR models using the SFT-formatted dataset by comparing model outputs directly with expected completions.

## Key Features
- Direct Completion Comparison: Compares model-generated text with expected completions
- Task-Specific Evaluation: Uses different evaluation strategies for each task
- Simple Accuracy Metrics: Calculates percentage of correct matches
- No Dependency on Reward Functions: Removes complex reward function dependencies

## Usage Examples

Basic usage:
python3 scripts/simple_eval_sft.py --model_path models/sft_qwen3_14b_lora_10k

Specific task:
python3 scripts/simple_eval_sft.py --model_path models/sft_qwen3_14b_lora_10k --task holding --num_samples 50

Background execution:
nohup python3 scripts/simple_eval_sft.py --model_path models/sft_qwen3_14b_lora_10k > eval.log 2>&1 &

## Arguments
- --model_path: Path to model (required)
- --task: Task to evaluate (default: all)
- --num_samples: Number of samples per task (default: 100) 
- --dataset: Dataset name (default: kylebrussell/cap-rlvr-sft)

The script provides accuracy metrics and pass/fail status with 80% threshold.
