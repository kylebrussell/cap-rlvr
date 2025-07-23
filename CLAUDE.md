# CLAUDE.md - Project Memory

This file contains important context and instructions for working with this CAP RLVR project.

## Remote System Access
SSH command to connect to the Vast.ai remote CPU instance:
```bash
ssh vast-cap
```

SSH command to connect to the Lambda Labs GPU instance:
```bash
ssh -i ~/.ssh/lambda_key_new ubuntu@104.171.203.89
```

## Data Preparation Scripts - Remote Execution Instructions

### Critical Setup
When running data prep scripts on remote CPU instance (e.g., Vast.ai), follow these exact steps:

```bash
# Always run from the scripts directory with proper environment
cd ~/cap_rlvr/scripts
source ../cap_env/bin/activate

# Run individual scripts in background with logging
python prep_holding_task.py > ../logs/holding.log 2>&1 &
python prep_bluebook_task.py > ../logs/bluebook.log 2>&1 &
python prep_summarise_task.py > ../logs/summary.log 2>&1 &
python prep_retrieval_task.py > ../logs/retrieval.log 2>&1 &
python prep_entail_task.py > ../logs/entail.log 2>&1 &
```

### Why This Matters
- **Working Directory**: Scripts expect `../data_raw/cap_raw/` path to exist
- **Environment**: Requires activated Python virtual environment with dependencies
- **CPU Utilization**: Run ALL 5 scripts simultaneously to utilize multi-core systems (15+ cores)
- **Logging**: Background processes need proper output redirection

### System Requirements
- **CPU**: 15+ cores (all prep scripts can run simultaneously)
- **Memory**: 98GB+ recommended (holding task uses ~3GB, retrieval uses ~1GB)
- **Storage**: 80GB+ for CAP dataset (23GB compressed) + output files
- **Time**: ~2-3 hours to process full CAP dataset

### Monitoring Progress
```bash
# Check running processes
ps aux | grep "python prep_" | grep -v grep

# Monitor progress logs
tail -f logs/holding.log
tail -f logs/bluebook.log

# Check system resources
top -bn1 | head -10
free -h
```

### Expected Output
Each script generates train/eval/test splits in `~/cap_rlvr/data_tasks/`:
- `holding/` - Multiple choice holding selection (expect ~100K+ questions)
- `bluebook/` - Citation format completion (expect ~50K+ questions)  
- `summarise/` - IRAC case summarization (expect ~30K+ questions)
- `retrieval/` - Analogous case finding (expect ~20K+ questions)
- `entail/` - Case relationship classification (expect ~40K+ questions)

### Common Issues Avoided
- ❌ Running from wrong directory (`~/cap_rlvr` instead of `~/cap_rlvr/scripts`)
- ❌ Not activating virtual environment
- ❌ Running scripts sequentially instead of parallel
- ❌ Log file creation failures with improper redirection

## Dataset Information
- **Source**: Caselaw Access Project (CAP) via HuggingFace
- **Size**: 78GB uncompressed, 23GB compressed (173 files)
- **Records**: ~7 million legal case documents
- **Download Method**: Use `downloads/cli_download.py` for robust acquisition with resume capability

## Frozen Embeddings Step
After data prep completes, build FAISS index for retrieval task:

```bash
cd ~/cap_rlvr/scripts
source ../cap_env/bin/activate

# Build FAISS index for retrieval evaluation
python build_faiss.py --in ../data_tasks/retrieval/train.jsonl --out ../data_tasks/retrieval/embeddings.faiss --test

# This creates:
# - embeddings.faiss (FAISS index file)
# - embeddings.metadata.json (case ID mappings)
```

**Purpose**: Creates frozen vector embeddings for efficient similarity search during retrieval task evaluation. Uses sentence-transformers to encode legal case texts and builds FAISS index for fast nearest-neighbor search.

## SFT Training with LoRA

### Memory-Optimized Training for GPU Instances
For GPU training on instances like Lambda Labs A6000 (50GB VRAM), use LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning:

```bash
cd ~/cap-rlvr
source ../cap_env/bin/activate

# Install additional dependencies for LoRA training
pip install peft

# Start LoRA SFT training (optimized for A6000)
python train_sft_lora.py \
  --model_name Qwen/Qwen3-14B \
  --train_file data_tasks/sft_formatted/unified/train_sft_unified.jsonl \
  --eval_file data_tasks/sft_formatted/unified/eval_sft_unified.jsonl \
  --output_dir models/sft_qwen3_14b_lora \
  > sft_lora_training.log 2>&1 &
```

### LoRA Configuration Details
- **Memory Usage**: ~15-25GB (vs ~94GB for full fine-tuning)
- **Trainable Parameters**: ~0.2% of total model parameters
- **Performance**: 85-95% of full fine-tuning quality for legal reasoning tasks
- **Training Speed**: 2-3x faster due to larger effective batch sizes

### GPU Requirements by Approach
| Approach | Memory Needed | A6000 Compatible | Recommended Batch Size |
|----------|---------------|------------------|------------------------|
| Full Fine-tuning | ~94GB | ❌ No | N/A |
| LoRA | ~25GB | ✅ Yes | 4-8 per device |
| QLoRA (4-bit) | ~15GB | ✅ Yes | 8-16 per device |

### Monitoring LoRA Training
```bash
# Check training progress
tail -f sft_lora_training.log

# Monitor GPU usage
nvidia-smi -l 5

# Check training process
ps aux | grep train_sft_lora
```

## Next Steps After Data Prep
1. Verify all 5 task types generated successfully
2. Build FAISS index for retrieval task (`build_faiss.py`)
3. Run reward function tests on sample outputs
4. Begin SFT warm-start training with LoRA on Qwen3-14B
5. Implement GRPO training pipeline with process supervision