#!/usr/bin/env python3
"""
Simple evaluation script for CAP RLVR models using HuggingFace datasets.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import sys
import os

# Add scripts directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rewards import UnifiedRewardFunction

def evaluate_model(model_path, task_filter=None, num_samples=100, dataset_name="kylebrussell/cap-rlvr-sft"):
    """Evaluate a model on CAP RLVR tasks"""
    
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)['train']
    
    # Filter by task if specified
    if task_filter:
        if task_filter == "all":
            tasks = ["bluebook", "holding", "summarise", "retrieval", "entail"]
        else:
            tasks = [task_filter]
    else:
        tasks = ["bluebook", "holding", "summarise", "retrieval", "entail"]
    
    # Initialize reward function
    reward_fn = UnifiedRewardFunction()
    
    results = {}
    
    for task in tasks:
        print(f"\nEvaluating task: {task}")
        
        # Filter dataset for this task
        task_samples = [sample for sample in dataset if sample['task'] == task]
        
        if not task_samples:
            print(f"No samples found for task {task}")
            continue
            
        # Sample random subset
        eval_samples = random.sample(task_samples, min(num_samples, len(task_samples)))
        print(f"Evaluating {len(eval_samples)} samples")
        
        task_rewards = []
        
        for i, sample in enumerate(eval_samples):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(eval_samples)}")
            
            prompt = sample['prompt']
            ground_truth = sample['completion']
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Compute reward
            reward_sample = {
                'inputs': prompt,
                'ground_truth': ground_truth,
                'metadata': sample.get('metadata', {})
            }
            
            try:
                reward = reward_fn.reward(reward_sample, response, task)
                task_rewards.append(reward)
            except Exception as e:
                print(f"Error computing reward: {e}")
                task_rewards.append(0.0)
        
        # Calculate statistics
        if task_rewards:
            mean_reward = sum(task_rewards) / len(task_rewards)
            results[task] = {
                'mean_reward': mean_reward,
                'num_samples': len(task_rewards),
                'passed_threshold': mean_reward >= 0.8  # Stage 0 threshold
            }
            print(f"Task {task}: Mean reward = {mean_reward:.3f} ({'PASS' if mean_reward >= 0.8 else 'FAIL'})")
        else:
            results[task] = {'mean_reward': 0.0, 'num_samples': 0, 'passed_threshold': False}
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Simple CAP RLVR model evaluation')
    parser.add_argument('--model_path', required=True, help='Path to model')
    parser.add_argument('--task', default='all', help='Task to evaluate (all, bluebook, holding, etc.)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate per task')
    parser.add_argument('--dataset', default='kylebrussell/cap-rlvr-sft', help='Dataset name')
    
    args = parser.parse_args()
    
    results = evaluate_model(args.model_path, args.task, args.num_samples, args.dataset)
    
    # Print summary
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"Stage 0 Threshold: 80% reward")
    print()
    
    total_passed = 0
    total_tasks = 0
    
    for task, result in results.items():
        status = "✅ PASS" if result['passed_threshold'] else "❌ FAIL"
        print(f"{status} {task.title()}: {result['mean_reward']:.3f} ({result['num_samples']} samples)")
        if result['num_samples'] > 0:
            total_passed += 1 if result['passed_threshold'] else 0
            total_tasks += 1
    
    print()
    if total_tasks > 0:
        print(f"Overall: {total_passed}/{total_tasks} tasks passed")
        print(f"Ready for Stage 1: {'✅ YES' if total_passed == total_tasks else '❌ NO'}")
    else:
        print("No tasks evaluated")

if __name__ == '__main__':
    main()