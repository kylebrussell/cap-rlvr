#!/usr/bin/env python3
"""Scan specific region of dataset where crash occurs"""

from datasets import load_dataset
from transformers import AutoTokenizer
import sys

def scan_region(start_idx=350000, end_idx=370000):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load specific region of dataset
    print(f"Loading dataset region {start_idx}-{end_idx}...")
    dataset = load_dataset("kylebrussell/cap-rlvr-sft", split=f"train[{start_idx}:{end_idx}]")
    
    print(f"Dataset region size: {len(dataset)} examples")
    
    def tokenize_function(examples):
        """Mirror the exact tokenization function from training"""
        texts = []
        for i in range(len(examples['prompt'])):
            prompt = examples['prompt'][i] or ""
            completion = examples['completion'][i] or ""
            text = f"{prompt}\n{completion}{tokenizer.eos_token}"
            texts.append(text)
        
        result = tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors=None
        )
        
        return result
    
    # Scan each example individually
    print("\nScanning individual examples...")
    anomalies = []
    
    for idx in range(len(dataset)):
        global_idx = start_idx + idx
        if idx % 1000 == 0:
            print(f"Progress: {idx}/{len(dataset)} (global index: {global_idx})")
        
        try:
            # Get single example
            prompt = dataset[idx]['prompt']
            completion = dataset[idx]['completion']
            
            # Check for basic issues
            if prompt is None:
                anomalies.append(f"Example {global_idx}: prompt is None")
            if completion is None:
                anomalies.append(f"Example {global_idx}: completion is None")
            
            # Tokenize single example
            single_batch = {
                'prompt': [prompt],
                'completion': [completion]
            }
            
            tokenized = tokenize_function(single_batch)
            
            # Check tokenized output
            if 'input_ids' not in tokenized:
                anomalies.append(f"Example {global_idx}: Missing input_ids")
                continue
                
            input_ids = tokenized['input_ids'][0]  # Get first (only) example
            
            # Detailed checks
            if not isinstance(input_ids, list):
                anomalies.append(f"Example {global_idx}: input_ids is not a list, type={type(input_ids)}")
            elif len(input_ids) == 0:
                anomalies.append(f"Example {global_idx}: Empty input_ids")
            elif any(not isinstance(token, int) for token in input_ids):
                non_ints = [(i, type(token)) for i, token in enumerate(input_ids) if not isinstance(token, int)]
                anomalies.append(f"Example {global_idx}: Non-integer tokens at positions {non_ints[:5]}")
            
            # Also try batch tokenization to simulate training
            if idx % 100 == 0 and idx > 0:
                batch_start = max(0, idx - 10)
                batch_end = min(len(dataset), idx + 10)
                batch = dataset[batch_start:batch_end]
                
                try:
                    batch_tokenized = tokenize_function(batch)
                    # Success
                except Exception as e:
                    anomalies.append(f"Batch tokenization failed around {global_idx}: {type(e).__name__}: {str(e)}")
                    
        except Exception as e:
            anomalies.append(f"Example {global_idx}: Error - {type(e).__name__}: {str(e)}")
            print(f"\nError at {global_idx}: {e}")
            # Print the problematic data
            try:
                print(f"Prompt: {repr(dataset[idx]['prompt'][:200])}...")
                print(f"Completion: {repr(dataset[idx]['completion'][:200])}...")
            except:
                print("Could not print example data")
    
    # Report findings
    print(f"\n=== SCAN COMPLETE ===")
    print(f"Total anomalies found: {len(anomalies)}")
    
    if anomalies:
        print("\nAnomalies found:")
        for anomaly in anomalies[:20]:
            print(f"  {anomaly}")
    else:
        print("\nNo anomalies found in this region!")
        
        # Try to tokenize the entire region as a batch
        print("\nTrying to tokenize entire region as batch...")
        try:
            full_tokenized = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing region"
            )
            print(f"✓ Full region tokenization successful! Got {len(full_tokenized)} examples")
            
            # Test with collator
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # Test batch from middle
            test_batch = [full_tokenized[i] for i in range(min(4, len(full_tokenized)))]
            collated = data_collator(test_batch)
            print("✓ Collation test successful!")
            
        except Exception as e:
            print(f"✗ Batch processing failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=350000, help="Start index")
    parser.add_argument("--end", type=int, default=370000, help="End index")
    args = parser.parse_args()
    
    scan_region(args.start, args.end)