#!/usr/bin/env python3
"""
Test script to verify streaming functionality in train_sft_robust.py
"""

import os
import sys
import logging
import time
import gc
from datasets import load_dataset
from transformers import AutoTokenizer

# Add parent dir to path so we can import from train_sft_robust
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_sft_robust import prepare_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_memory():
    """Get current memory usage (simplified without psutil)"""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB on Linux, already MB on Mac
    except:
        return 0  # Fallback if resource monitoring not available

def test_streaming_dataset():
    """Test that streaming mode works correctly"""
    logger.info("=== Testing Streaming Dataset Functionality ===")
    
    # Use a small, fast tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Test parameters
    dataset_name = "kylebrussell/cap-rlvr-sft"
    max_samples = 1000  # Small number for testing
    max_length = 512
    
    # Monitor initial memory
    initial_memory = monitor_memory()
    logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Test dataset preparation with streaming
    logger.info("Testing dataset preparation with streaming...")
    start_time = time.time()
    
    try:
        dataset = prepare_dataset(dataset_name, tokenizer, max_samples, max_length)
        prep_time = time.time() - start_time
        memory_after_prep = monitor_memory()
        
        logger.info(f"Dataset preparation completed in {prep_time:.2f} seconds")
        logger.info(f"Memory after prep: {memory_after_prep:.1f} MB")
        logger.info(f"Memory increase: {memory_after_prep - initial_memory:.1f} MB")
        
        # Verify dataset types
        from datasets import IterableDataset
        train_is_iterable = isinstance(dataset['train'], IterableDataset)
        val_is_iterable = isinstance(dataset['validation'], IterableDataset)
        
        logger.info(f"Train dataset is IterableDataset: {train_is_iterable}")
        logger.info(f"Validation dataset is IterableDataset: {val_is_iterable}")
        
        if not (train_is_iterable and val_is_iterable):
            logger.error("ERROR: Datasets are not streaming (IterableDataset)!")
            return False
            
        # Test data iteration and tokenization
        logger.info("Testing data iteration...")
        train_iter = iter(dataset['train'])
        
        memory_samples = []
        for i in range(min(10, max_samples)):
            try:
                sample = next(train_iter)
                current_memory = monitor_memory()
                memory_samples.append(current_memory)
                
                if i == 0:
                    # Log first sample structure
                    logger.info(f"First sample keys: {list(sample.keys())}")
                    logger.info(f"Input IDs length: {len(sample['input_ids'])}")
                    
                if i % 100 == 0:
                    logger.info(f"Processed {i} samples, memory: {current_memory:.1f} MB")
                    
            except StopIteration:
                logger.info(f"Dataset exhausted after {i} samples")
                break
        
        # Check memory stability
        if len(memory_samples) > 1:
            memory_growth = memory_samples[-1] - memory_samples[0]
            logger.info(f"Memory growth during iteration: {memory_growth:.1f} MB")
            
            if memory_growth > 50:  # Allow some growth, but not excessive
                logger.warning(f"WARNING: Significant memory growth detected ({memory_growth:.1f} MB)")
            else:
                logger.info("✓ Memory usage appears stable during iteration")
        
        logger.info("✓ Streaming dataset test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"ERROR during streaming test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_integration():
    """Test that streaming dataset works with Trainer"""
    logger.info("\n=== Testing Trainer Integration ===")
    
    try:
        from transformers import (
            AutoModelForCausalLM, 
            TrainingArguments, 
            Trainer,
            DataCollatorForLanguageModeling
        )
        
        # Use tiny model for speed
        model_name = "sshleifer/tiny-gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Prepare small streaming dataset
        dataset = prepare_dataset("kylebrussell/cap-rlvr-sft", tokenizer, 50, 256)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        # Minimal training args
        training_args = TrainingArguments(
            output_dir="/tmp/test_streaming",
            per_device_train_batch_size=2,
            max_steps=3,  # Just a few steps
            logging_steps=1,
            save_steps=10,
            remove_unused_columns=False,
            report_to="none",
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=tokenizer,
        )
        
        logger.info("Testing trainer setup and first batch...")
        
        # Try to get one batch to verify integration
        train_dataloader = trainer.get_train_dataloader()
        first_batch = next(iter(train_dataloader))
        
        logger.info(f"✓ Successfully created batch with keys: {list(first_batch.keys())}")
        logger.info(f"✓ Batch size: {first_batch['input_ids'].shape}")
        
        logger.info("✓ Trainer integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"ERROR during trainer integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all streaming functionality tests"""
    logger.info("Starting streaming functionality tests...")
    
    test_results = []
    
    # Test 1: Basic streaming dataset functionality
    test_results.append(test_streaming_dataset())
    
    # Test 2: Trainer integration
    test_results.append(test_trainer_integration())
    
    # Summary
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Streaming dataset test: {'PASS' if test_results[0] else 'FAIL'}")
    logger.info(f"Trainer integration test: {'PASS' if test_results[1] else 'FAIL'}")
    
    if all(test_results):
        logger.info("✓ ALL TESTS PASSED - Streaming implementation is ready for deployment!")
        return 0
    else:
        logger.error("✗ SOME TESTS FAILED - Fix issues before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())