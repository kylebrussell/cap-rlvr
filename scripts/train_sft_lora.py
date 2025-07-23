#!/usr/bin/env python3
"""
LoRA SFT Training Script for CAP RLVR Legal Reasoning

Optimized for A6000 GPU with memory-efficient LoRA fine-tuning of Qwen3-14B.
Uses PEFT library for parameter-efficient training on legal reasoning tasks.

Usage:
    python train_sft_lora.py --model_name Qwen/Qwen3-14B --output_dir models/sft_qwen3_14b_lora
"""

import argparse
import json
import logging
import sys
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sft_lora_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LegalReasoningLoRATrainer:
    """LoRA trainer optimized for legal reasoning tasks on A6000 GPU"""
    
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"🚀 Initializing LoRA trainer for {model_name}")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"🔧 Device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def setup_model_and_tokenizer(self):
        """Load model and tokenizer with LoRA configuration"""
        logger.info("📦 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("🔧 Set pad_token to eos_token")
        
        logger.info("🧠 Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False,
            trust_remote_code=True
        )
        
        # LoRA configuration optimized for legal reasoning
        lora_config = LoraConfig(
            r=16,  # Rank - balance between efficiency and performance
            lora_alpha=32,  # Scaling factor
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj"      # MLP layers
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        logger.info("🔗 Applying LoRA configuration...")
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"📊 Trainable parameters: {trainable_params:,}")
        logger.info(f"📊 Total parameters: {total_params:,}")
        logger.info(f"📊 Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    def load_dataset_streaming(self, train_file: str, eval_file: str, max_samples: int = None):
        """Load datasets with memory-efficient streaming approach"""
        from datasets import load_dataset
        
        logger.info(f"📊 Loading training data from {train_file} (streaming mode)")
        
        # Use streaming dataset to avoid loading everything into memory
        train_dataset = load_dataset(
            "json", 
            data_files=train_file, 
            split="train",
            streaming=True
        )
        
        eval_dataset = load_dataset(
            "json", 
            data_files=eval_file, 
            split="train", 
            streaming=True
        )
        
        # Take only the requested number of samples
        if max_samples:
            train_dataset = train_dataset.take(max_samples)
            eval_dataset = eval_dataset.take(max_samples // 10)
            logger.info(f"📈 Train samples: {max_samples:,} (streaming)")
            logger.info(f"📊 Eval samples: {max_samples // 10:,} (streaming)")
        else:
            logger.info(f"📈 Using full dataset (streaming mode)")
        
        return train_dataset, eval_dataset
    
    def tokenize_function(self, examples):
        """Tokenize text with proper truncation and padding"""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=1024,  # Reduced for memory efficiency
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def setup_training_args(self):
        """Configure training arguments optimized for A6000"""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Minimal batch size for memory
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Larger accumulation for effective batch size
            learning_rate=1e-4,  # Slightly higher for LoRA
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
            save_steps=1000,
            eval_steps=1000,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            dataloader_num_workers=0,  # No multiprocessing to save memory
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing=True,  # Additional memory savings
        )
    
    def train(self, train_file: str, eval_file: str, max_samples: int = None):
        """Execute LoRA training pipeline"""
        logger.info("🎯 Starting LoRA SFT training pipeline...")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Load datasets with streaming for memory efficiency
        train_dataset, eval_dataset = self.load_dataset_streaming(train_file, eval_file, max_samples)
        
        logger.info("🔤 Tokenizing datasets...")
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        tokenized_eval = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        logger.info("⚡ Starting training...")
        
        # Train the model
        trainer.train()
        
        logger.info("💾 Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save LoRA adapter separately for easy loading
        self.model.save_pretrained(f"{self.output_dir}/lora_adapter")
        
        logger.info("✅ LoRA SFT training completed successfully!")
        logger.info(f"📂 Model saved to: {self.output_dir}")
        logger.info(f"🔗 LoRA adapter saved to: {self.output_dir}/lora_adapter")

def main():
    parser = argparse.ArgumentParser(description="LoRA SFT Training for Legal Reasoning")
    
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B",
                       help="Base model name from HuggingFace")
    parser.add_argument("--train_file", 
                       default="data_tasks/sft_formatted/unified/train_sft_unified.jsonl",
                       help="Training data file")
    parser.add_argument("--eval_file",
                       default="data_tasks/sft_formatted/unified/eval_sft_unified.jsonl", 
                       help="Evaluation data file")
    parser.add_argument("--output_dir", default="models/sft_qwen3_14b_lora",
                       help="Output directory for trained model")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of training samples (for testing)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run trainer
    trainer = LegalReasoningLoRATrainer(args.model_name, args.output_dir)
    trainer.train(args.train_file, args.eval_file, args.max_samples)

if __name__ == "__main__":
    main()