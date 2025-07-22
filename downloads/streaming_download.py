#!/usr/bin/env python3
"""
Optimized CAP dataset downloader using streaming and chunked processing.
"""
import time
import sys
import os
from pathlib import Path
from datasets import load_dataset
import logging
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streaming_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_cap_streaming(output_dir: str = "data_raw/cap", chunk_size: int = 10000):
    """
    Download CAP dataset using streaming mode with chunked processing.
    
    Args:
        output_dir: Directory to save the dataset
        chunk_size: Number of examples to process at a time
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("Loading CAP dataset in streaming mode...")
        
        # Use streaming mode to avoid loading entire dataset into memory
        dataset = load_dataset(
            "common-pile/caselaw_access_project", 
            split="train",
            streaming=True,  # Key: Use streaming mode
            trust_remote_code=False
        )
        
        logger.info("Starting chunked download and save...")
        
        # Process in chunks to avoid memory issues
        chunk_files = []
        chunk_num = 0
        current_chunk = []
        total_processed = 0
        
        # Create progress bar (we don't know total size in streaming)
        pbar = tqdm(desc="Processing examples", unit="examples")
        
        for example in dataset:
            current_chunk.append(example)
            pbar.update(1)
            total_processed += 1
            
            # Save chunk when it reaches target size
            if len(current_chunk) >= chunk_size:
                chunk_file = output_path / f"chunk_{chunk_num:06d}.jsonl"
                save_chunk(current_chunk, chunk_file)
                chunk_files.append(chunk_file)
                
                logger.info(f"Saved chunk {chunk_num} with {len(current_chunk)} examples (total: {total_processed})")
                current_chunk = []
                chunk_num += 1
        
        # Save final partial chunk
        if current_chunk:
            chunk_file = output_path / f"chunk_{chunk_num:06d}.jsonl"
            save_chunk(current_chunk, chunk_file)
            chunk_files.append(chunk_file)
            logger.info(f"Saved final chunk {chunk_num} with {len(current_chunk)} examples")
        
        pbar.close()
        
        # Create metadata file
        metadata = {
            "total_examples": total_processed,
            "num_chunks": len(chunk_files),
            "chunk_files": [str(f.name) for f in chunk_files],
            "chunk_size": chunk_size
        }
        
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Download completed! {total_processed} examples in {len(chunk_files)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Streaming download failed: {str(e)}")
        return False

def save_chunk(examples, chunk_file):
    """Save a chunk of examples to JSONL format."""
    with open(chunk_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

def resume_download(output_dir: str = "data_raw/cap", chunk_size: int = 10000):
    """Resume download from where it left off."""
    output_path = Path(output_dir)
    metadata_file = output_path / "metadata.json"
    
    if not metadata_file.exists():
        logger.info("No previous download found, starting fresh")
        return download_cap_streaming(output_dir, chunk_size)
    
    # Load existing metadata
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    existing_chunks = len(metadata["chunk_files"])
    logger.info(f"Found {existing_chunks} existing chunks, resuming...")
    
    # Continue from where we left off
    try:
        dataset = load_dataset(
            "common-pile/caselaw_access_project", 
            split="train",
            streaming=True,
            trust_remote_code=False
        )
        
        # Skip already processed examples
        examples_to_skip = existing_chunks * chunk_size
        logger.info(f"Skipping first {examples_to_skip} examples...")
        
        dataset_iter = iter(dataset)
        for _ in range(examples_to_skip):
            try:
                next(dataset_iter)
            except StopIteration:
                logger.info("Reached end of dataset, download was already complete")
                return True
        
        # Continue processing from where we left off
        chunk_num = existing_chunks
        current_chunk = []
        total_new = 0
        
        pbar = tqdm(desc=f"Resuming from chunk {chunk_num}", unit="examples")
        
        for example in dataset_iter:
            current_chunk.append(example)
            pbar.update(1)
            total_new += 1
            
            if len(current_chunk) >= chunk_size:
                chunk_file = output_path / f"chunk_{chunk_num:06d}.jsonl"
                save_chunk(current_chunk, chunk_file)
                
                # Update metadata
                metadata["chunk_files"].append(chunk_file.name)
                metadata["num_chunks"] += 1
                metadata["total_examples"] += len(current_chunk)
                
                logger.info(f"Saved chunk {chunk_num} with {len(current_chunk)} examples")
                current_chunk = []
                chunk_num += 1
        
        # Save final chunk and update metadata
        if current_chunk:
            chunk_file = output_path / f"chunk_{chunk_num:06d}.jsonl"
            save_chunk(current_chunk, chunk_file)
            metadata["chunk_files"].append(chunk_file.name)
            metadata["num_chunks"] += 1
            metadata["total_examples"] += len(current_chunk)
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        pbar.close()
        logger.info(f"Resume completed! Added {total_new} new examples")
        return True
        
    except Exception as e:
        logger.error(f"Resume failed: {str(e)}")
        return False

def main():
    """Main function with resume capability."""
    logger.info("Starting optimized CAP dataset download")
    
    # Check disk space
    import shutil
    free_gb = shutil.disk_usage('.').free / (1024**3)
    logger.info(f"Available disk space: {free_gb:.1f} GB")
    
    if free_gb < 80:
        logger.error(f"Insufficient disk space. Need 80+ GB, have {free_gb:.1f} GB")
        sys.exit(1)
    
    # Set working directory
    os.chdir(os.path.expanduser("~/cap_rlvr"))
    
    # Try to resume existing download, or start new one
    success = resume_download()
    
    if success:
        logger.info("Download completed successfully!")
        
        # Show final stats
        metadata_file = Path("data_raw/cap/metadata.json")
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            logger.info(f"Final stats: {metadata['total_examples']} examples in {metadata['num_chunks']} chunks")
        
        sys.exit(0)
    else:
        logger.error("Download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()