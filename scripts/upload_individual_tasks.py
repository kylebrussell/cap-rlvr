#!/usr/bin/env python3
"""
Upload individual CAP RLVR task datasets to HuggingFace Hub

Uploads each legal reasoning task as a separate dataset for flexible training
and development.
"""

import os
import argparse
from datasets import load_dataset
from huggingface_hub import HfApi
from typing import List

TASK_INFO = {
    "holding": {
        "name": "Holding Selection",
        "description": "Multiple-choice questions identifying correct legal holdings from case law",
        "size_estimate": "29MB",
        "sample_count": "~100K"
    },
    "bluebook": {
        "name": "Bluebook Citation Completion", 
        "description": "Fill-in-the-blank citation format completion following Bluebook standards",
        "size_estimate": "97MB",
        "sample_count": "~50K"
    },
    "summarise": {
        "name": "IRAC Case Summarization",
        "description": "Structured case summarization using Issue-Rule-Application-Conclusion format",
        "size_estimate": "14GB", 
        "sample_count": "~30K"
    },
    "entail": {
        "name": "Case Relationship Classification",
        "description": "Determining how cases relate (overrule, distinguish, affirm, etc.)",
        "size_estimate": "5GB",
        "sample_count": "~40K"
    }
}

def upload_task_dataset(task_name: str, data_dir: str, username: str, token: str = None):
    """Upload a single task dataset to HuggingFace Hub"""
    
    task_dir = os.path.join(data_dir, task_name)
    repo_name = f"{username}/cap-rlvr-{task_name}"
    
    print(f"🚀 Uploading {task_name} dataset to {repo_name}")
    
    # Paths to the task files
    train_file = os.path.join(task_dir, "train.jsonl")
    eval_file = os.path.join(task_dir, "eval.jsonl") 
    test_file = os.path.join(task_dir, "test.jsonl")
    
    # Verify files exist
    for file_path in [train_file, eval_file, test_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
        print(f"✅ Found: {file_path} ({os.path.getsize(file_path) / 1e6:.1f}MB)")
    
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
    for key, value in sample.items():
        if isinstance(value, str):
            print(f"  {key}: {value[:100]}{'...' if len(value) > 100 else ''}")
        else:
            print(f"  {key}: {value}")
    
    # Upload to HuggingFace Hub
    print(f"\n📤 Uploading to HuggingFace Hub: {repo_name}")
    dataset.push_to_hub(
        repo_name,
        token=token,
        private=False
    )
    
    # Create task-specific dataset card
    task_info = TASK_INFO.get(task_name, {})
    readme_content = f"""# CAP RLVR {task_info.get('name', task_name.title())} Dataset

{task_info.get('description', f'Legal reasoning dataset for {task_name} tasks')}

## Dataset Overview

- **Task Type**: {task_info.get('name', task_name.title())}
- **Train samples**: {len(dataset['train']):,}
- **Validation samples**: {len(dataset['validation']):,}  
- **Test samples**: {len(dataset['test']):,}
- **Estimated size**: {task_info.get('size_estimate', 'Unknown')}

## Task Description

{task_info.get('description', f'This dataset contains {task_name} tasks derived from legal case law.')}

## Data Format

Each record contains:
- `inputs`: The question or prompt for the legal reasoning task
- `outputs`: The expected answer or completion
- Additional metadata specific to {task_name} tasks

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_name}")

# Load specific split
train_data = load_dataset("{repo_name}", split="train")

# Example usage
for example in train_data.take(1):
    print("Input:", example["inputs"])
    print("Output:", example["outputs"])
```

## Training Recommendations

- **Model**: Qwen3-14B or similar legal-focused models
- **Task-specific training**: Can be used individually or combined with other legal reasoning tasks
- **Evaluation**: Use test split for final evaluation, validation for hyperparameter tuning

## Related Datasets

This is part of the CAP RLVR project. See also:
- [`{username}/cap-rlvr-sft`](https://huggingface.co/datasets/{username}/cap-rlvr-sft) - Unified SFT dataset
- [`{username}/cap-rlvr-holding`](https://huggingface.co/datasets/{username}/cap-rlvr-holding) - Holding selection tasks
- [`{username}/cap-rlvr-bluebook`](https://huggingface.co/datasets/{username}/cap-rlvr-bluebook) - Citation completion tasks
- [`{username}/cap-rlvr-summarise`](https://huggingface.co/datasets/{username}/cap-rlvr-summarise) - IRAC summarization tasks
- [`{username}/cap-rlvr-entail`](https://huggingface.co/datasets/{username}/cap-rlvr-entail) - Case relationship classification

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
    
    print(f"📝 Uploaded dataset card for {task_name}")
    return repo_name

def upload_all_tasks(tasks: List[str], data_dir: str, username: str, token: str = None):
    """Upload all specified task datasets"""
    
    print(f"🎯 Uploading {len(tasks)} task datasets to HuggingFace")
    uploaded_repos = []
    
    for task in tasks:
        try:
            repo_name = upload_task_dataset(task, data_dir, username, token)
            uploaded_repos.append(repo_name)
            print(f"✅ Successfully uploaded: {repo_name}")
        except Exception as e:
            print(f"❌ Failed to upload {task}: {e}")
            continue
    
    print(f"\n🎉 Upload summary:")
    print(f"✅ Successful: {len(uploaded_repos)}")
    print(f"❌ Failed: {len(tasks) - len(uploaded_repos)}")
    
    if uploaded_repos:
        print(f"\n📂 Uploaded datasets:")
        for repo in uploaded_repos:
            print(f"  - https://huggingface.co/datasets/{repo}")
    
    return uploaded_repos

def main():
    parser = argparse.ArgumentParser(description="Upload individual CAP RLVR task datasets to HuggingFace")
    parser.add_argument("--tasks", nargs="+", 
                       choices=["holding", "bluebook", "summarise", "entail", "all"],
                       default=["all"],
                       help="Tasks to upload (default: all)")
    parser.add_argument("--data_dir", default="data_tasks",
                       help="Directory containing task subdirectories")
    parser.add_argument("--username", default="kylebrussell",
                       help="HuggingFace username for repository names")
    parser.add_argument("--token", 
                       help="HuggingFace API token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Expand "all" to specific tasks
    if "all" in args.tasks:
        tasks = ["holding", "bluebook", "summarise", "entail"]
    else:
        tasks = args.tasks
    
    # Get token from environment if not provided
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("❌ HuggingFace token required. Set HF_TOKEN env var or use --token")
        return 1
    
    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return 1
    
    # Check which tasks are available
    available_tasks = []
    for task in tasks:
        task_dir = os.path.join(args.data_dir, task)
        if os.path.exists(task_dir):
            available_tasks.append(task)
            print(f"✅ Found task directory: {task_dir}")
        else:
            print(f"⚠️  Task directory not found: {task_dir}")
    
    if not available_tasks:
        print("❌ No task directories found")
        return 1
    
    try:
        uploaded_repos = upload_all_tasks(available_tasks, args.data_dir, args.username, token)
        if uploaded_repos:
            print("🎉 Upload completed successfully!")
            return 0
        else:
            print("❌ No datasets were uploaded successfully")
            return 1
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())