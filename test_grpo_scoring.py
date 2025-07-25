#!/usr/bin/env python3
"""
Quick test script to debug GRPO scoring issues
"""
import json
import sys
import os
sys.path.append('scripts')

from rewards import UnifiedRewardFunction
from datasets import load_dataset

def test_reward_scoring():
    print("Testing GRPO reward scoring...")
    
    # Load original SFT dataset
    print("\n1. Loading SFT dataset...")
    dataset = load_dataset("kylebrussell/cap-rlvr-sft", split="train")
    
    # Get an entail sample
    entail_samples = [s for s in dataset if s.get("task") == "entail"][:5]
    print(f"Found {len(entail_samples)} entail samples")
    
    # Initialize reward function
    reward_fn = UnifiedRewardFunction()
    
    print("\n2. Testing reward function...")
    for i, sample in enumerate(entail_samples):
        print(f"\nSample {i+1}:")
        print(f"  Keys: {list(sample.keys())}")
        
        # Test different responses
        test_responses = ["AFFIRMS", "OVERRULES", "DISTINGUISHES", "NONE"]
        
        for response in test_responses:
            try:
                score = reward_fn.reward(sample, response, task_type="entail")
                print(f"  Response '{response}' -> Score: {score}")
                if score > 0:
                    print(f"    SUCCESS: Got non-zero score!")
                    return  # Found working example
            except Exception as e:
                print(f"  Response '{response}' -> ERROR: {e}")
        
        if i >= 2:  # Test only first 3 samples
            break
    
    print("\n3. All tests completed")

if __name__ == "__main__":
    test_reward_scoring()