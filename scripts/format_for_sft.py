#!/usr/bin/env python3
"""
SFT Dataset Formatter for CAP RLVR Tasks
Converts task-specific JSONL data into TRL-compatible prompt-completion format.
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

class SFTDataFormatter:
    def __init__(self, output_dir="data_tasks/sft_formatted"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Task-specific prompt templates
        self.templates = {
            'holding': self._format_holding,
            'bluebook': self._format_bluebook,
            'summarise': self._format_summarise,
            'retrieval': self._format_retrieval,
            'entail': self._format_entail
        }
        
    def _format_holding(self, sample: Dict) -> Dict:
        """Format holding selection task for SFT"""
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(sample['choices'])])
        
        prompt = f"""You are a legal expert. Given a case scenario, select the correct legal holding from the multiple choices provided.

Case Context: Based on the legal case provided, choose the most accurate holding statement.

Multiple Choice Question:
{choices_text}

Select the letter corresponding to the correct holding:"""

        # Convert answer_idx to letter (0->A, 1->B, etc.)
        correct_letter = chr(65 + sample['answer_idx'])
        completion = correct_letter
        
        return {
            "prompt": prompt,
            "completion": completion,
            "task": "holding",
            "metadata": {
                "case_id": sample.get('case_id', ''),
                "num_choices": len(sample['choices'])
            }
        }
        
    def _format_bluebook(self, sample: Dict) -> Dict:
        """Format bluebook citation task for SFT"""
        prompt = f"""You are a legal citation expert. Complete the following legal citation according to proper Bluebook format.

Citation to complete: {sample['inputs']}

Provide the complete, properly formatted citation:"""

        completion = sample['ground_truth']
        
        return {
            "prompt": prompt,
            "completion": completion,
            "task": "bluebook",
            "metadata": sample.get('metadata', {})
        }
        
    def _format_summarise(self, sample: Dict) -> Dict:
        """Format IRAC summarization task for SFT"""
        prompt = f"""You are a legal writing expert. Create a concise legal case summary using the IRAC method (Issue, Rule, Analysis, Conclusion).

{sample['inputs']}

Provide a structured summary following IRAC format:"""

        # Extract ground truth components
        gt = sample['ground_truth']
        if isinstance(gt, dict):
            completion = f"**Issue**: {gt.get('syllabus', 'Legal issue regarding the case facts.')}\n"
            completion += f"**Parties**: {', '.join(gt.get('key_parties', []))}\n"
            completion += f"**Analysis**: Based on the case facts and applicable legal principles.\n"
            completion += f"**Conclusion**: The court's ruling addresses the core legal issue presented."
        else:
            completion = str(gt)
            
        return {
            "prompt": prompt,
            "completion": completion,
            "task": "summarise",
            "metadata": {
                "case_id": sample.get('case_id', '')
            }
        }
        
    def _format_retrieval(self, sample: Dict) -> Dict:
        """Format case retrieval task for SFT"""
        prompt = f"""You are a legal research expert. Given the case facts below, identify analogous cases that are legally relevant.

{sample['inputs']}

List the case IDs of analogous cases, one per line:"""

        # Format positive case IDs
        positives = sample.get('positives', [])
        completion = "\n".join(positives[:5])  # Limit to top 5 for training
        
        return {
            "prompt": prompt,
            "completion": completion,
            "task": "retrieval",
            "metadata": {
                "case_id": sample.get('case_id', ''),
                "num_positives": len(positives)
            }
        }
        
    def _format_entail(self, sample: Dict) -> Dict:
        """Format case relationship/entailment task for SFT"""
        prompt = f"""You are a legal precedent expert. Analyze the relationship between legal cases based on the provided context.

Context: {sample['inputs']}

Classify the relationship between the cases. Choose from:
- OVERRULES: The citing case overrules or abrogates the cited case
- DISTINGUISHES: The citing case distinguishes itself from the cited case  
- AFFIRMS: The citing case affirms or follows the cited case
- NONE: No clear relationship is established

Relationship classification:"""

        completion = sample['label']
        
        return {
            "prompt": prompt,
            "completion": completion,
            "task": "entail",
            "metadata": {
                "pair_id": sample.get('pair_id', '')
            }
        }
        
    def format_task_data(self, task_name: str, input_file: Path, split: str = "train") -> List[Dict]:
        """Format a single task's data"""
        if task_name not in self.templates:
            raise ValueError(f"Unknown task: {task_name}")
            
        formatted_data = []
        formatter = self.templates[task_name]
        
        print(f"Formatting {task_name} {split} data from {input_file}")
        
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    formatted_sample = formatter(sample)
                    
                    # Add split and line info
                    formatted_sample['split'] = split
                    formatted_sample['source_line'] = line_num
                    
                    formatted_data.append(formatted_sample)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    print(f"Warning: Error formatting line {line_num}: {e}")
                    
        print(f"Formatted {len(formatted_data)} samples for {task_name} {split}")
        return formatted_data
        
    def format_all_tasks(self, data_dir: Path, tasks: List[str] = None) -> Dict[str, Dict]:
        """Format all specified tasks"""
        if tasks is None:
            tasks = ['holding', 'bluebook', 'summarise', 'retrieval', 'entail']
            
        all_formatted_data = {}
        
        for task in tasks:
            task_dir = data_dir / task
            if not task_dir.exists():
                print(f"Warning: Task directory {task_dir} does not exist, skipping")
                continue
                
            task_data = {}
            
            # Process each split
            for split in ['train', 'eval', 'test']:
                split_file = task_dir / f"{split}.jsonl"
                if split_file.exists():
                    task_data[split] = self.format_task_data(task, split_file, split)
                else:
                    print(f"Warning: {split_file} does not exist, skipping")
                    
            all_formatted_data[task] = task_data
            
        return all_formatted_data
        
    def save_formatted_data(self, formatted_data: Dict[str, Dict], output_format: str = "separate"):
        """Save formatted data in specified format"""
        
        if output_format == "separate":
            # Save each task separately
            for task, splits in formatted_data.items():
                task_dir = self.output_dir / task
                task_dir.mkdir(exist_ok=True)
                
                for split, samples in splits.items():
                    output_file = task_dir / f"{split}_sft.jsonl"
                    with open(output_file, 'w') as f:
                        for sample in samples:
                            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    print(f"Saved {len(samples)} samples to {output_file}")
                    
        elif output_format == "unified":
            # Save all tasks together by split
            unified_dir = self.output_dir / "unified"
            unified_dir.mkdir(exist_ok=True)
            
            # Combine all tasks by split
            combined_splits = defaultdict(list)
            for task, splits in formatted_data.items():
                for split, samples in splits.items():
                    combined_splits[split].extend(samples)
                    
            for split, samples in combined_splits.items():
                # Shuffle for better training dynamics
                random.shuffle(samples)
                
                output_file = unified_dir / f"{split}_sft_unified.jsonl"
                with open(output_file, 'w') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                print(f"Saved {len(samples)} unified samples to {output_file}")
                
        elif output_format == "chat":
            # Save in chat message format for newer models
            chat_dir = self.output_dir / "chat_format"
            chat_dir.mkdir(exist_ok=True)
            
            for task, splits in formatted_data.items():
                task_dir = chat_dir / task
                task_dir.mkdir(exist_ok=True)
                
                for split, samples in splits.items():
                    chat_samples = []
                    for sample in samples:
                        chat_sample = {
                            "messages": [
                                {"role": "user", "content": sample["prompt"]},
                                {"role": "assistant", "content": sample["completion"]}
                            ],
                            "task": sample["task"],
                            "metadata": sample.get("metadata", {})
                        }
                        chat_samples.append(chat_sample)
                        
                    output_file = task_dir / f"{split}_chat.jsonl"
                    with open(output_file, 'w') as f:
                        for sample in chat_samples:
                            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    print(f"Saved {len(chat_samples)} chat-format samples to {output_file}")
                    
    def generate_statistics(self, formatted_data: Dict[str, Dict]) -> Dict:
        """Generate statistics about the formatted dataset"""
        stats = {}
        
        for task, splits in formatted_data.items():
            task_stats = {}
            
            for split, samples in splits.items():
                if not samples:
                    continue
                    
                prompt_lengths = [len(s['prompt'].split()) for s in samples]
                completion_lengths = [len(s['completion'].split()) for s in samples]
                
                split_stats = {
                    'num_samples': len(samples),
                    'avg_prompt_length': sum(prompt_lengths) / len(prompt_lengths),
                    'avg_completion_length': sum(completion_lengths) / len(completion_lengths),
                    'max_prompt_length': max(prompt_lengths),
                    'max_completion_length': max(completion_lengths),
                }
                
                task_stats[split] = split_stats
                
            stats[task] = task_stats
            
        return stats
        
    def print_statistics(self, stats: Dict):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("SFT DATASET STATISTICS")
        print("="*60)
        
        for task, task_stats in stats.items():
            print(f"\n📊 {task.upper()} TASK:")
            for split, split_stats in task_stats.items():
                print(f"  {split}: {split_stats['num_samples']} samples")
                print(f"    Avg prompt length: {split_stats['avg_prompt_length']:.1f} words")
                print(f"    Avg completion length: {split_stats['avg_completion_length']:.1f} words")
                print(f"    Max prompt/completion: {split_stats['max_prompt_length']}/{split_stats['max_completion_length']} words")

def main():
    parser = argparse.ArgumentParser(description="Format CAP RLVR task data for SFT training")
    parser.add_argument("--data-dir", type=Path, default="data_tasks", 
                       help="Directory containing task data")
    parser.add_argument("--output-dir", type=Path, default="data_tasks/sft_formatted",
                       help="Output directory for formatted data")
    parser.add_argument("--tasks", nargs="+", 
                       choices=['holding', 'bluebook', 'summarise', 'retrieval', 'entail'],
                       help="Tasks to format (default: all)")
    parser.add_argument("--format", choices=['separate', 'unified', 'chat'], default='separate',
                       help="Output format: separate files per task, unified file, or chat format")
    parser.add_argument("--stats-only", action='store_true',
                       help="Only generate and print statistics, don't save files")
    
    args = parser.parse_args()
    
    formatter = SFTDataFormatter(output_dir=args.output_dir)
    
    print("🚀 Starting SFT data formatting...")
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output format: {args.format}")
    
    # Format all tasks
    formatted_data = formatter.format_all_tasks(args.data_dir, args.tasks)
    
    if not formatted_data:
        print("❌ No data was formatted. Check your data directory and task files.")
        return
    
    # Generate and print statistics
    stats = formatter.generate_statistics(formatted_data)
    formatter.print_statistics(stats)
    
    if not args.stats_only:
        # Save formatted data
        formatter.save_formatted_data(formatted_data, args.format)
        print(f"\n✅ SFT formatting completed! Data saved to {args.output_dir}")
        
        # Print usage instructions
        print("\n📋 USAGE INSTRUCTIONS:")
        print("For TRL SFTTrainer, use the formatted data as follows:")
        
        if args.format == 'separate':
            print("  # Train on individual tasks:")
            print("  dataset = load_dataset('json', data_files='data_tasks/sft_formatted/bluebook/train_sft.jsonl')")
        elif args.format == 'unified':
            print("  # Train on all tasks together:")
            print("  dataset = load_dataset('json', data_files='data_tasks/sft_formatted/unified/train_sft_unified.jsonl')")
        elif args.format == 'chat':
            print("  # Use chat format with apply_chat_template:")
            print("  dataset = load_dataset('json', data_files='data_tasks/sft_formatted/chat_format/bluebook/train_chat.jsonl')")
            
        print("  trainer = SFTTrainer(model=model, train_dataset=dataset, dataset_text_field='text')")
    else:
        print("\n📊 Statistics only mode - no files were saved.")

if __name__ == "__main__":
    main()