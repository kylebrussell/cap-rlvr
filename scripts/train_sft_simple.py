#!/usr/bin/env python3
"""
Simplified FP16 LoRA SFT Training for H100s
Avoids TensorFlow conflicts by using PyTorch-only components
"""

import os
import argparse
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(model_name):
    """Load model and tokenizer with FP16 LoRA configuration"""
    
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA Configuration optimized for H100s
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling parameter
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Load model in FP16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # Automatic multi-GPU placement
        trust_remote_code=True,
        use_cache=False  # Disable caching for training
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer

def prepare_dataset(dataset_name, tokenizer, max_samples=None, max_length=1024):
    """Load and tokenize datasets for SFT training"""
    
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Limit samples if specified
    if max_samples:
        train_size = min(max_samples, len(dataset['train']))
        eval_size = min(max_samples // 10, len(dataset['validation']))
        dataset['train'] = dataset['train'].select(range(train_size))
        dataset['validation'] = dataset['validation'].select(range(eval_size))
    
    logger.info(f"Train samples: {len(dataset['train']):,}")
    logger.info(f"Validation samples: {len(dataset['validation']):,}")
    
    # Tokenize function for SFT format
    def tokenize_function(examples):
        # Combine prompt and completion for SFT
        texts = []
        for prompt, completion in zip(examples['prompt'], examples['completion']):
            # Handle None/empty completions gracefully
            if completion is None:
                completion = ""
            if prompt is None:
                prompt = ""
            text = f"{prompt}\n{completion}{tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize with error handling
        try:
            return tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            # Return empty tokenization if failed
            return {'input_ids': [], 'attention_mask': []}
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['task', 'metadata', 'split', 'source_line'],  # Keep prompt/completion for tokenization
        desc="Tokenizing"
    )
    
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description="Simplified H100 LoRA SFT Training")
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B", help="Base model name")
    parser.add_argument("--dataset_name", default="kylebrussell/cap-rlvr-sft", help="HuggingFace dataset name")
    parser.add_argument("--output_dir", default="models/sft_qwen3_14b_lora", help="Output directory")
    parser.add_argument("--max_samples", type=int, help="Max samples to use (for testing)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU (H100 optimized)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation (H100 optimized)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save frequency")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging frequency")
    
    args = parser.parse_args()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    
    # Prepare dataset
    dataset = prepare_dataset(args.dataset_name, tokenizer, args.max_samples, args.max_length)
    
    # Training arguments optimized for H100s
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        
        # Batch configuration for H100s
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Precision settings
        fp16=True,
        dataloader_pin_memory=True,
        
        # Learning settings
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        
        # Logging and saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        
        # Memory optimization for H100s
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        weight_decay=0.01,
        
        # Multi-GPU
        ddp_find_unused_parameters=False,
        
        # Disable unnecessary features
        report_to="none",
        load_best_model_at_end=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Calculate effective batch size
    num_gpus = torch.cuda.device_count()
    effective_batch_size = args.per_device_train_batch_size * num_gpus * args.gradient_accumulation_steps
    
    logger.info(f"Training Configuration:")
    logger.info(f"  GPUs: {num_gpus}")
    logger.info(f"  Per-device batch size: {args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Max length: {args.max_length}")
    
    # Import Trainer here to avoid TF conflicts
    from transformers import Trainer
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training summary
    summary = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "train_samples": len(dataset['train']),
        "validation_samples": len(dataset['validation']),
        "effective_batch_size": effective_batch_size,
        "num_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length
    }
    
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()