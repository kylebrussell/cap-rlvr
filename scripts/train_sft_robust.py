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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

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
    
    # Enable gradient checkpointing BEFORE preparing for PEFT
    model.gradient_checkpointing_enable()
    
    # Prepare model for PEFT training (required for gradient checkpointing)
    model = prepare_model_for_kbit_training(model)
    
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
    """Prepare dataset with robust tokenization using streaming to handle large datasets."""
    logger.info(f"Loading dataset: {dataset_name} in streaming mode")
    
    # Load the dataset in streaming mode, which returns an IterableDatasetDict
    dataset = load_dataset(dataset_name, streaming=True)

    # The tokenize_function remains the same, as it processes batches
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
        
        # Tokenize - let collator handle padding for better efficiency
        result = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # Let DataCollator handle padding dynamically
            return_tensors=None  # Return lists, not tensors
        )
        
        # Explicitly create labels for causal language modeling
        # Labels should be the same as input_ids for next-token prediction
        result["labels"] = result["input_ids"].copy()
        
        return result

    # Get column names from streaming dataset by taking one sample
    sample = next(iter(dataset['train']))
    original_columns = list(sample.keys())
    logger.info(f"Original dataset columns: {original_columns}")
    
    # We need to remove ALL original columns since tokenize_function creates new ones
    # The tokenizer will create: input_ids, attention_mask (and labels via data collator)
    columns_to_remove = original_columns
    logger.info(f"Removing columns: {columns_to_remove}")
    
    logger.info("Tokenizing datasets on the fly...")
    tokenized_train = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove
    )
    
    tokenized_validation = dataset['validation'].map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove
    )

    # Apply max_samples limit after tokenization
    if max_samples:
        logger.info(f"Limiting train samples to {max_samples}")
        tokenized_train = tokenized_train.take(max_samples)
        
        eval_samples = max(1, max_samples // 10)
        logger.info(f"Limiting validation samples to {eval_samples}")
        tokenized_validation = tokenized_validation.take(eval_samples)

    logger.info("Dataset preparation complete. Training will now use the streamed dataset.")
    
    return {"train": tokenized_train, "validation": tokenized_validation}

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
    
    # Use a different data collator that handles padding for labels correctly
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,  # Optimize for tensor cores
    )
    
    # Calculate max_steps for streaming dataset (since it doesn't have __len__)
    num_gpus = torch.cuda.device_count()
    effective_batch_size = args.per_device_train_batch_size * num_gpus * args.gradient_accumulation_steps
    
    # Estimate steps: if we have max_samples, use that; otherwise use reasonable default
    if args.max_samples:
        estimated_steps = (args.max_samples // effective_batch_size) * args.num_train_epochs
    else:
        # Default for 1M samples with 1 epoch
        estimated_steps = (1000000 // effective_batch_size) * args.num_train_epochs
    
    logger.info(f"Estimated training steps: {estimated_steps}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        
        # Batch configuration
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Training configuration
        max_steps=estimated_steps,  # Required for streaming datasets
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
        remove_unused_columns=True,  # Let Trainer automatically remove unused columns
        report_to="none",
    )
    
    logger.info(f"Training Configuration:")
    logger.info(f"  GPUs: {num_gpus}")
    logger.info(f"  Per-device batch size: {args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Max steps: {estimated_steps}")
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