# CLAUDE.md - Project Memory

This file contains important context and instructions for working with this CAP RLVR project.

## Remote System Access
SSH command to connect to the Vast.ai remote CPU instance:
```bash
ssh vast-cap
```

SSH command to connect to the current Lambda Labs H100 GPU instance:
```bash
ssh -i ~/.ssh/lambda ubuntu@192.222.52.232
```

**GPU Instance Specs:**
- **GPUs**: 2x NVIDIA H100-80GB HBM3 (160GB total VRAM)
- **CUDA**: 12.8
- **PyTorch**: 2.7.0
- **Purpose**: Optimized FP16 LoRA training with large batch sizes

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
python prep_retrieval_task_streaming.py > ../logs/retrieval_streaming.log 2>&1 &  # Memory-optimized streaming version
python prep_entail_task.py > ../logs/entail.log 2>&1 &
```

### Why This Matters
- **Working Directory**: Scripts expect `../data_raw/cap_raw/` path to exist
- **Environment**: Requires activated Python virtual environment with dependencies
- **CPU Utilization**: Run ALL 5 scripts simultaneously to utilize multi-core systems (15+ cores)
- **Logging**: Background processes need proper output redirection

### System Requirements
- **CPU**: 15+ cores (all prep scripts can run simultaneously)
- **Memory**: 32GB+ minimum (streaming retrieval uses <1GB, holding task uses ~3GB)
- **Storage**: 80GB+ for CAP dataset (23GB compressed) + output files + SQLite temp files
- **Time**: ~2-3 hours to process full CAP dataset

### Memory Optimizations
- **Streaming Retrieval**: Uses SQLite for indexing instead of loading all records into RAM
- **Memory Reduction**: 98% less RAM usage (28GB → 600MB for retrieval task)
- **Fault Tolerance**: Process can run on systems with limited RAM (32GB+)

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

## SFT Training with H100 LoRA

### Optimized FP16 Training for H100 GPUs
Current setup uses 2x H100-80GB GPUs with optimized FP16 LoRA training (no quantization needed):

```bash
cd ~/cap-rlvr

# H100-Optimized LoRA SFT training with HuggingFace datasets
python3 scripts/train_sft_simple.py \
  --dataset_name kylebrussell/cap-rlvr-sft \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --output_dir models/sft_qwen3_14b_lora \
  > sft_training.log 2>&1 &

# For testing with subset
python3 scripts/train_sft_simple.py \
  --dataset_name kylebrussell/cap-rlvr-sft \
  --max_samples 10000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4
```

### H100 LoRA Configuration Details
- **Memory Usage**: ~132GB total (82% utilization, 28GB safety margin)
- **Trainable Parameters**: 64.2M / 14.8B total (0.43%)
- **Performance**: 90-95% of full fine-tuning quality for legal reasoning
- **Training Speed**: 4-6x faster than A6000 due to larger batches
- **Effective Batch Size**: 64 (4 per GPU × 2 GPUs × 8 accumulation)

### GPU Performance Comparison
| Setup | Memory Used | Batch Size | Training Speed | Quality |
|-------|-------------|------------|----------------|---------|
| A6000 LoRA | ~25GB | 8 effective | 1x baseline | 85-90% |
| H100 FP16 LoRA | ~132GB | 64 effective | 4-6x faster | 90-95% |
| H100 Full FT | ~150GB+ | ❌ Too much | N/A | 100% |

### Monitoring H100 Training
```bash
# Check training progress
tail -f sft_training.log

# Monitor H100 GPU usage
nvidia-smi -l 5

# Check training process
ps aux | grep train_sft_simple

# Monitor memory usage per GPU
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv -l 5
```

## Next Steps After Data Prep
1. Verify all 5 task types generated successfully
2. Build FAISS index for retrieval task (`build_faiss.py`)
3. Run reward function tests on sample outputs
4. Begin SFT warm-start training with LoRA on Qwen3-14B
5. Implement GRPO training pipeline with process supervision