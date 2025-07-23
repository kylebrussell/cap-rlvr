#\!/usr/bin/env python3
"""
Simple evaluation script for CAP RLVR models using SFT-formatted dataset.
Compares model outputs directly with expected completions using string matching.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import sys
import os
from difflib import SequenceMatcher
import re

def clean_text(text):
    """Clean text for comparison by removing extra whitespace and normalizing"""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase for case-insensitive comparison
    return text.lower()

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts using SequenceMatcher"""
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    if not clean1 and not clean2:
        return 1.0
    if not clean1 or not clean2:
        return 0.0
        
    return SequenceMatcher(None, clean1, clean2).ratio()

def exact_match(text1, text2):
    """Check if two texts match exactly after cleaning"""
    return clean_text(text1) == clean_text(text2)

def evaluate_task_specific(task, generated, expected):
    """Task-specific evaluation logic"""
    
    if task == "bluebook":
        # For citation format, use exact match
        return 1.0 if exact_match(generated, expected) else 0.0
    
    elif task == "holding":
        # For multiple choice, check if the answer letter appears in generated text
        expected_clean = clean_text(expected)
        generated_clean = clean_text(generated)
        
        # Look for answer patterns like "A)", "(A)", "A.", or just "A"
        answer_patterns = [
            r'^\s*([A-E])\s*[\)\.]\s*',  # A) or A.
            r'^\s*\(([A-E])\)\s*',       # (A)
            r'^\s*([A-E])\s*$',          # Just A
            r'answer\s*is\s*([A-E])',    # "answer is A"
            r'option\s*([A-E])',         # "option A"
        ]
        
        # Extract expected answer
        expected_answer = None
        for pattern in answer_patterns:
            match = re.search(pattern, expected_clean)
            if match:
                expected_answer = match.group(1).upper()
                break
        
        # Extract generated answer
        generated_answer = None
        for pattern in answer_patterns:
            match = re.search(pattern, generated_clean)
            if match:
                generated_answer = match.group(1).upper()
                break
        
        # If we found both answers, compare them
        if expected_answer and generated_answer:
            return 1.0 if expected_answer == generated_answer else 0.0
        
        # Fallback to exact match
        return 1.0 if exact_match(generated, expected) else 0.0
    
    elif task == "entail":
        # For entailment, look for key words like "entails", "contradicts", "neutral"
        expected_clean = clean_text(expected)
        generated_clean = clean_text(generated)
        
        # Common entailment labels
        entailment_labels = ["entails", "contradicts", "neutral", "supports", "opposes"]
        
        expected_label = None
        generated_label = None
        
        for label in entailment_labels:
            if label in expected_clean:
                expected_label = label
                break
        
        for label in entailment_labels:
            if label in generated_clean:
                generated_label = label
                break
        
        if expected_label and generated_label:
            return 1.0 if expected_label == generated_label else 0.0
        
        # Fallback to similarity
        return calculate_similarity(generated, expected)
    
    elif task == "summarise":
        # For IRAC summaries, use similarity scoring (more flexible)
        similarity = calculate_similarity(generated, expected)
        # Consider it correct if similarity > 0.7
        return 1.0 if similarity > 0.7 else 0.0
    
    elif task == "retrieval":
        # For retrieval, check if case citations match
        expected_clean = clean_text(expected)
        generated_clean = clean_text(generated)
        
        # Extract case citations (basic pattern matching)
        citation_pattern = r'[A-Za-z\s]+v\.?\s+[A-Za-z\s]+|[A-Za-z\s]+\s+\d+\s+[A-Za-z\.]+\s+\d+'
        
        expected_citations = set(re.findall(citation_pattern, expected_clean))
        generated_citations = set(re.findall(citation_pattern, generated_clean))
        
        if expected_citations and generated_citations:
            # Check for any overlap in citations
            overlap = expected_citations.intersection(generated_citations)
            return 1.0 if overlap else 0.0
        
        # Fallback to exact match
        return 1.0 if exact_match(generated, expected) else 0.0
    
    else:
        # Default: use similarity scoring
        similarity = calculate_similarity(generated, expected)
        return 1.0 if similarity > 0.8 else 0.0

def evaluate_model(model_path, task_filter=None, num_samples=100, dataset_name="kylebrussell/cap-rlvr-sft"):
    """Evaluate a model on CAP RLVR tasks using direct completion comparison"""
    
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
        
        correct_count = 0
        total_count = 0
        task_scores = []
        
        for i, sample in enumerate(eval_samples):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(eval_samples)}")
            
            prompt = sample['prompt']
            expected_completion = sample['completion']
            
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
            generated_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Evaluate using task-specific logic
            try:
                score = evaluate_task_specific(task, generated_response, expected_completion)
                task_scores.append(score)
                
                if score >= 1.0:
                    correct_count += 1
                total_count += 1
                
                # Debug output for first few samples
                if i < 3:
                    print(f"    Sample {i+1}:")
                    print(f"      Expected: {expected_completion[:100]}...")
                    print(f"      Generated: {generated_response[:100]}...")
                    print(f"      Score: {score:.3f}")
                    
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                task_scores.append(0.0)
                total_count += 1
        
        # Calculate statistics
        if task_scores:
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            mean_score = sum(task_scores) / len(task_scores)
            
            results[task] = {
                'accuracy': accuracy,
                'mean_score': mean_score,
                'correct': correct_count,
                'total': total_count,
                'passed_threshold': accuracy >= 0.8  # 80% accuracy threshold
            }
            
            print(f"Task {task}: Accuracy = {accuracy:.3f} ({correct_count}/{total_count}) ({'PASS' if accuracy >= 0.8 else 'FAIL'})")
            print(f"           Mean Score = {mean_score:.3f}")
        else:
            results[task] = {'accuracy': 0.0, 'mean_score': 0.0, 'correct': 0, 'total': 0, 'passed_threshold': False}
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Simple CAP RLVR model evaluation with SFT dataset')
    parser.add_argument('--model_path', required=True, help='Path to model')
    parser.add_argument('--task', default='all', help='Task to evaluate (all, bluebook, holding, etc.)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate per task')
    parser.add_argument('--dataset', default='kylebrussell/cap-rlvr-sft', help='Dataset name')
    
    args = parser.parse_args()
    
    results = evaluate_model(args.model_path, args.task, args.num_samples, args.dataset)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SFT EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Accuracy Threshold: 80%")
    print()
    
    total_passed = 0
    total_tasks = 0
    overall_correct = 0
    overall_total = 0
    
    for task, result in results.items():
        status = "✅ PASS" if result['passed_threshold'] else "❌ FAIL"
        print(f"{status} {task.title()}: {result['accuracy']:.3f} ({result['correct']}/{result['total']} correct)")
        print(f"      Mean Score: {result['mean_score']:.3f}")
        
        if result['total'] > 0:
            total_passed += 1 if result['passed_threshold'] else 0
            total_tasks += 1
            overall_correct += result['correct']
            overall_total += result['total']
    
    print()
    if total_tasks > 0:
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
        print(f"Overall Accuracy: {overall_accuracy:.3f} ({overall_correct}/{overall_total})")
        print(f"Tasks Passed: {total_passed}/{total_tasks}")
        print(f"Ready for Next Stage: {'✅ YES' if total_passed == total_tasks else '❌ NO'}")
    else:
        print("No tasks evaluated")

if __name__ == '__main__':
    main()
