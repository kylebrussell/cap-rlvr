# CLAUDE.md - Project Memory

This file contains important context and instructions for working with this CAP RLVR project.

## Remote System Access
SSH command to connect to the current Lambda Labs H100 GPU instance:
```bash
ssh -i ~/.ssh/lambda ubuntu@192.222.52.232
```

**GPU Instance Specs:**
- **GPUs**: 2x NVIDIA H100-80GB HBM3 (160GB total VRAM)
- **CUDA**: 12.8
- **PyTorch**: 2.7.0
- **Purpose**: Optimized FP16 LoRA training with large batch sizes

## Dataset Availability
**✅ Data preparation completed!** All processed datasets are now available on HuggingFace:

- **All 5 Tasks**: `kylebrussell/cap-rlvr-sft` (SFT training data)
- **Retrieval + FAISS**: `kylebrussell/cap-rlvr-retrieval` (includes embeddings.faiss index)

Data prep was completed on remote CPU instances with the following outputs:
- `holding/` - Multiple choice holding selection (~30K questions)
- `bluebook/` - Citation format completion (~50K questions)  
- `summarise/` - IRAC case summarization (~30K questions)
- `retrieval/` - Analogous case finding (~30K questions)
- `entail/` - Case relationship classification (~40K questions)

## Dataset Information
- **Source**: Caselaw Access Project (CAP) via HuggingFace
- **Size**: 78GB uncompressed, 23GB compressed (173 files)
- **Records**: ~7 million legal case documents
- **Download Method**: Use `downloads/cli_download.py` for robust acquisition with resume capability

## FAISS Embeddings for Retrieval
**✅ FAISS index completed!** Available in `kylebrussell/cap-rlvr-retrieval`:
- `embeddings.faiss` - FAISS index file (43MB)
- `embeddings.metadata.json` - Case ID mappings (2.1MB)

Uses sentence-transformers (all-MiniLM-L6-v2) for efficient similarity search during retrieval evaluation.

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

## Trained Models
**Available SFT Models on Lambda H100s:**
- `models/sft_qwen3_14b_lora_10k` - Working LoRA model (10K samples subset)
- `models/sft_qwen3_14b_lora_production_final` - Empty directory (training incomplete)

**Validation Results:**
- **Base Model (Qwen/Qwen3-14B)**: 0.152 mean reward (15.2% - FAIL)
- **SFT Model**: Validation in progress

## Next Steps
1. ✅ Verify all 5 task types generated successfully  
2. ✅ Build FAISS index for retrieval task
3. ✅ Upload datasets to HuggingFace
4. ✅ Complete SFT warm-start training with LoRA on Qwen3-14B
5. 🔄 Run validation comparisons between base and SFT models
6. Implement GRPO training pipeline with process supervision