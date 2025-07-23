#!/usr/bin/env python3
"""
Upload CAP RLVR SFT datasets to HuggingFace Hub

Uploads the unified SFT training datasets to HuggingFace for easy access
from better GPU instances.
"""

import os
import argparse
from datasets import load_dataset
from huggingface_hub import HfApi

def upload_sft_datasets(repo_name: str, data_dir: str, token: str = None):
    """Upload SFT datasets to HuggingFace Hub"""
    
    print(f"🚀 Uploading CAP RLVR SFT datasets to {repo_name}")
    
    # Paths to the SFT files
    train_file = os.path.join(data_dir, "train_sft_unified.jsonl")
    eval_file = os.path.join(data_dir, "eval_sft_unified.jsonl") 
    test_file = os.path.join(data_dir, "test_sft_unified.jsonl")
    
    # Verify files exist
    for file_path in [train_file, eval_file, test_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
        print(f"✅ Found: {file_path} ({os.path.getsize(file_path) / 1e9:.1f}GB)")
    
    # Load the dataset
    print("📊 Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": eval_file,
            "test": test_file
        }
    )
    
    # Print dataset info
    print(f"📈 Train samples: {len(dataset['train']):,}")
    print(f"📊 Eval samples: {len(dataset['validation']):,}")
    print(f"🧪 Test samples: {len(dataset['test']):,}")
    
    # Sample data inspection
    print("\n📋 Sample record:")
    sample = dataset['train'][0]
    print(f"  Task: {sample.get('task', 'unknown')}")
    print(f"  Prompt length: {len(sample['prompt'])} chars")
    print(f"  Completion length: {len(sample['completion'])} chars")
    
    # Upload to HuggingFace Hub
    print(f"\n📤 Uploading to HuggingFace Hub: {repo_name}")
    dataset.push_to_hub(
        repo_name,
        token=token,
        private=False  # Make it public for easy access
    )
    
    print(f"✅ Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_name}")
    
    # Create dataset card with metadata
    readme_content = f"""# CAP RLVR Legal Reasoning SFT Dataset

This dataset contains supervised fine-tuning (SFT) data for legal reasoning tasks, derived from the Caselaw Access Project (CAP).

## Dataset Overview

- **Train samples**: {len(dataset['train']):,}
- **Validation samples**: {len(dataset['validation']):,}  
- **Test samples**: {len(dataset['test']):,}
- **Total size**: ~{(os.path.getsize(train_file) + os.path.getsize(eval_file) + os.path.getsize(test_file)) / 1e9:.1f}GB

## Legal Tasks Included

1. **Holding Selection**: Multiple-choice questions identifying correct legal holdings
2. **Bluebook Citations**: Fill-in-the-blank citation format completion
3. **IRAC Summaries**: Structured case summarization using Issue-Rule-Application-Conclusion
4. **Case Retrieval**: Finding analogous cases based on legal concepts
5. **Relationship Classification**: Determining how cases relate (overrule, distinguish, affirm, etc.)

## Data Format

Each record contains:
- `prompt`: The instruction and context for the legal reasoning task
- `completion`: The expected response/answer
- `task`: The type of legal reasoning task (holding, bluebook, summarise, retrieval, entail)
- `metadata`: Additional context and identifiers
- `split`: Data split (train/validation/test)
- `source_line`: Original line number from processing

## Usage

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("{repo_name}")

# Load specific split
train_data = load_dataset("{repo_name}", split="train")

# Use with transformers for training
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

def preprocess(examples):
    inputs = [p + c for p, c in zip(examples["prompt"], examples["completion"])]
    return tokenizer(inputs, truncation=True, padding=True, max_length=1024)

train_data = train_data.map(preprocess, batched=True)
```

## Training Recommendations

- **Model**: Qwen3-14B or similar legal-focused models
- **Approach**: LoRA fine-tuning for memory efficiency
- **Batch Size**: 1-4 with gradient accumulation
- **Learning Rate**: 1e-4 to 5e-5
- **Max Length**: 1024-2048 tokens

## Source

Generated from the [CAP RLVR project](https://github.com/kylebrussell/cap-rlvr) using the Caselaw Access Project corpus.

## License

The dataset follows the terms of the original Caselaw Access Project data.
"""
    
    # Upload README
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset"
    )
    
    print(f"📝 Uploaded dataset card to README.md")

def main():
    parser = argparse.ArgumentParser(description="Upload CAP RLVR SFT datasets to HuggingFace")
    parser.add_argument("--repo_name", default="kylebrussell/cap-rlvr-sft", 
                       help="HuggingFace dataset repository name")
    parser.add_argument("--data_dir", default="data_tasks/sft_formatted/unified",
                       help="Directory containing SFT files")
    parser.add_argument("--token", 
                       help="HuggingFace API token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("❌ HuggingFace token required. Set HF_TOKEN env var or use --token")
        return 1
    
    try:
        upload_sft_datasets(args.repo_name, args.data_dir, token)
        print("🎉 Upload completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())