#!/usr/bin/env python3
"""Debug script to inspect tokenization output structure"""

from datasets import load_dataset
from transformers import AutoTokenizer

def inspect_dataset():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load a small sample of the dataset
    print("Loading dataset...")
    dataset = load_dataset("kylebrussell/cap-rlvr-sft", split="train[:10]")
    
    print(f"\nRaw dataset columns: {dataset.column_names}")
    print(f"First example raw:")
    print(f"  prompt: {dataset[0]['prompt'][:100]}...")
    print(f"  completion: {dataset[0]['completion'][:100]}...")
    
    # Test tokenization function
    def tokenize_function(examples):
        texts = []
        for i in range(len(examples['prompt'])):
            prompt = examples['prompt'][i] or ""
            completion = examples['completion'][i] or ""
            text = f"{prompt}\n{completion}{tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize
        result = tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors=None
        )
        
        return result
    
    # Tokenize a batch
    print("\n\nTokenizing dataset...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print(f"\nTokenized dataset columns: {tokenized.column_names}")
    print(f"\nFirst tokenized example structure:")
    
    # Inspect the structure
    first_example = tokenized[0]
    for key, value in first_example.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, list):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
                print(f"  First 10 elements: {value[:10]}")
                # Check for nested lists
                if isinstance(value[0], list):
                    print(f"  WARNING: Nested list detected!")
                    print(f"  First nested list: {value[0][:10]}")
    
    # Test with DataCollatorForLanguageModeling
    print("\n\nTesting DataCollatorForLanguageModeling...")
    from transformers import DataCollatorForLanguageModeling
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Try to collate a batch
    try:
        batch = [tokenized[i] for i in range(min(4, len(tokenized)))]
        print(f"Batch size: {len(batch)}")
        print(f"Batch[0] keys: {batch[0].keys()}")
        
        collated = data_collator(batch)
        print("✓ Collation successful!")
        print(f"Collated batch keys: {collated.keys()}")
        print(f"Collated shapes: {[(k, v.shape) for k, v in collated.items()]}")
    except Exception as e:
        print(f"✗ Collation failed with error: {type(e).__name__}: {e}")
        print("\nDetailed batch structure:")
        for i, example in enumerate(batch):
            print(f"\nExample {i}:")
            for key, value in example.items():
                print(f"  {key}: type={type(value)}, len={len(value) if isinstance(value, list) else 'N/A'}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"    First element: type={type(value[0])}, value={value[0] if not isinstance(value[0], list) else 'NESTED LIST'}")

if __name__ == "__main__":
    inspect_dataset()