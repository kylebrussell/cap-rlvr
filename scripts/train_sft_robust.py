#!/usr/bin/env python3
"""
Robust SFT training script for H100 GPUs - simplified and bulletproof
"""

import os
import argparse
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(model_name):
    """Setup model and tokenizer with proper configuration"""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset(dataset_name, tokenizer, max_samples, max_length):
    """Prepare dataset with robust tokenization"""
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Limit samples if specified
    if max_samples and max_samples < len(dataset['train']):
        train_size = min(max_samples, len(dataset['train']))
        eval_size = min(max_samples // 10, len(dataset['validation']))
        dataset['train'] = dataset['train'].select(range(train_size))
        dataset['validation'] = dataset['validation'].select(range(eval_size))
    
    logger.info(f"Train samples: {len(dataset['train']):,}")
    logger.info(f"Validation samples: {len(dataset['validation']):,}")
    
    def tokenize_function(examples):
        """Robust tokenization function with defensive handling"""
        # Combine prompt and completion
        texts = []
        for i in range(len(examples['prompt'])):
            prompt = examples['prompt'][i] or ""
            completion = examples['completion'][i] or ""
            
            # Ensure we never have empty text - always have at least eos_token
            text = f"{prompt}\n{completion}".strip()
            if not text:
                text = tokenizer.eos_token
            else:
                text = f"{text}{tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize - let the collator handle padding and label creation
        result = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # Don't pad here, let collator do it
            return_tensors=None
        )
        
        # Defensive: ensure output structure is correct
        # Each value should be a list of lists (batch of sequences)
        for key in result:
            if isinstance(result[key], list):
                # Ensure no empty sequences that could cause nesting issues
                result[key] = [seq if seq else [tokenizer.eos_token_id] for seq in result[key]]
        
        # Don't create labels here - DataCollatorForLanguageModeling will handle it
        return result
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )
    
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    args = parser.parse_args()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    
    # Prepare dataset
    dataset = prepare_dataset(args.dataset_name, tokenizer, args.max_samples, args.max_length)
    
    # Data collator - this handles padding and label creation
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8,  # Optimize for tensor cores
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        
        # Batch configuration
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Training configuration
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        
        # Memory optimization
        fp16=True,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        
        # Logging and saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=args.save_steps,
        
        # Other
        remove_unused_columns=False,
        report_to="none",
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
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()

if __name__ == "__main__":
    main()