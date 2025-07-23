#!/usr/bin/env python3
"""Scan entire dataset for anomalies that could break DataCollatorForLanguageModeling"""

from datasets import load_dataset
from transformers import AutoTokenizer
import sys
from collections import defaultdict

def scan_for_anomalies(max_samples=None):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load dataset
    print("Loading dataset...")
    if max_samples:
        dataset = load_dataset("kylebrussell/cap-rlvr-sft", split=f"train[:{max_samples}]")
    else:
        dataset = load_dataset("kylebrussell/cap-rlvr-sft", split="train")
    
    print(f"Dataset size: {len(dataset)} examples")
    
    # Statistics
    stats = defaultdict(int)
    anomalies = []
    length_distribution = defaultdict(int)
    
    def tokenize_single(prompt, completion):
        """Tokenize a single example"""
        prompt = prompt or ""
        completion = completion or ""
        text = f"{prompt}\n{completion}{tokenizer.eos_token}"
        
        result = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors=None
        )
        return result
    
    # Scan dataset
    print("\nScanning for anomalies...")
    batch_size = 100
    
    for i in range(0, len(dataset), batch_size):
        if i % 10000 == 0:
            print(f"Progress: {i}/{len(dataset)} ({i/len(dataset)*100:.1f}%)")
        
        batch_end = min(i + batch_size, len(dataset))
        batch = dataset[i:batch_end]
        
        for j, (prompt, completion) in enumerate(zip(batch['prompt'], batch['completion'])):
            idx = i + j
            
            # Check for None/empty values
            if prompt is None:
                anomalies.append(f"Example {idx}: prompt is None")
                stats['none_prompt'] += 1
            if completion is None:
                anomalies.append(f"Example {idx}: completion is None")
                stats['none_completion'] += 1
            
            # Try tokenization
            try:
                tokenized = tokenize_single(prompt, completion)
                
                # Check tokenization output
                if 'input_ids' not in tokenized:
                    anomalies.append(f"Example {idx}: Missing input_ids")
                    stats['missing_input_ids'] += 1
                    continue
                
                input_ids = tokenized['input_ids']
                
                # Check for empty sequences
                if len(input_ids) == 0:
                    anomalies.append(f"Example {idx}: Empty input_ids")
                    stats['empty_sequence'] += 1
                
                # Check for nested lists
                if isinstance(input_ids, list) and len(input_ids) > 0:
                    if isinstance(input_ids[0], list):
                        anomalies.append(f"Example {idx}: Nested list in input_ids")
                        stats['nested_list'] += 1
                    elif not isinstance(input_ids[0], int):
                        anomalies.append(f"Example {idx}: Non-integer in input_ids: {type(input_ids[0])}")
                        stats['non_integer'] += 1
                
                # Track length distribution
                length_bin = (len(input_ids) // 100) * 100
                length_distribution[length_bin] += 1
                
                # Check for very long sequences
                if len(input_ids) >= 1024:
                    stats['max_length_sequences'] += 1
                
            except Exception as e:
                anomalies.append(f"Example {idx}: Tokenization failed - {type(e).__name__}: {str(e)}")
                stats['tokenization_failed'] += 1
    
    # Report findings
    print("\n=== ANOMALY REPORT ===")
    print(f"Total examples scanned: {len(dataset)}")
    print(f"Total anomalies found: {len(anomalies)}")
    
    print("\nStatistics:")
    for stat, count in sorted(stats.items()):
        print(f"  {stat}: {count}")
    
    print("\nLength distribution:")
    for length, count in sorted(length_distribution.items()):
        print(f"  {length}-{length+99}: {count} examples")
    
    print("\nFirst 20 anomalies:")
    for anomaly in anomalies[:20]:
        print(f"  {anomaly}")
    
    # Try batch collation test
    if len(anomalies) == 0:
        print("\nNo anomalies found! Testing batch collation with random samples...")
        from transformers import DataCollatorForLanguageModeling
        import random
        
        # Tokenize random batch
        indices = random.sample(range(len(dataset)), min(32, len(dataset)))
        test_batch = []
        
        for idx in indices:
            tokenized = tokenize_single(dataset[idx]['prompt'], dataset[idx]['completion'])
            test_batch.append(tokenized)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        try:
            collated = data_collator(test_batch)
            print("✓ Random batch collation successful!")
        except Exception as e:
            print(f"✗ Random batch collation failed: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    # Run with optional sample limit
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, help="Maximum samples to scan")
    args = parser.parse_args()
    
    scan_for_anomalies(args.max_samples)