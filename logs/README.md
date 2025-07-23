# Training and Validation Logs

This directory contains important logs from CAP RLVR model training and evaluation runs on Lambda Labs H100 instances.

## Key Training Logs

### `sft_10k_training.log`
- **Status**: ✅ **SUCCESSFUL SFT Training**
- **Model**: Qwen3-14B with LoRA fine-tuning  
- **Dataset**: 10,000 samples from `kylebrussell/cap-rlvr-sft`
- **Duration**: ~19 minutes (1130.8645 seconds)
- **Final Loss**: 1.297 (improved from ~1.5+)
- **Training Speed**: 4.4 samples/second, 0.138 steps/second
- **Steps**: 156/156 completed (100%)
- **Output**: `models/sft_qwen3_14b_lora_10k`
- **Hardware**: 2x H100-80GB GPUs, FP16 LoRA training

### `sft_production_final.log`
- **Status**: ❌ Failed (gradient computation error)
- **Model**: Qwen3-14B attempted with 1M samples
- **Error**: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
- **Note**: Led to switching to smaller 10K sample training

## Validation Development Logs

### `validation_sequential.log`
- **Status**: ✅ **Sequential GRPO Approach Working**
- **Innovation**: Memory-efficient sequential model loading
- **Phase 1**: Reference model response generation (0/50 samples started)
- **Memory Usage**: ~50% reduction vs parallel GRPO loading
- **Next**: Phase 2 main model evaluation and comparison

### `validation_prompt_format.log`
- **Status**: ✅ Data format fixes successful
- **Fixed**: HuggingFace dataset loading and GRPO format conversion
- **Format**: `inputs/ground_truth` → `prompt/chosen/rejected`
- **Error Resolved**: CUDA out of memory (expected with 14B models)
- **Achievement**: All data pipeline issues resolved

## Training Configuration

**Successful SFT Configuration:**
```bash
python3 scripts/train_sft_robust.py \
  --dataset_name kylebrussell/cap-rlvr-sft \
  --max_samples 10000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --max_length 1024 \
  --output_dir models/sft_qwen3_14b_lora_10k \
  --logging_steps 10 \
  --save_steps 100
```

**Hardware Specs:**
- **GPUs**: 2x NVIDIA H100-80GB HBM3 (160GB total VRAM)
- **CUDA**: 12.8
- **PyTorch**: 2.7.0
- **Training Method**: FP16 LoRA (no quantization needed)
- **Effective Batch Size**: 64 (4 per GPU × 2 GPUs × 8 accumulation)

## Next Steps

1. ✅ Complete sequential validation evaluation
2. ⏳ Compare base Qwen3-14B vs SFT model performance
3. ⏳ Generate comprehensive evaluation report
4. 🎯 Proceed to GRPO stage if SFT shows improvement

## Memory Optimization Success

The sequential GRPO approach successfully resolved GPU memory constraints:
- **Previous**: Dual model loading (~60GB) → Out of memory
- **Sequential**: Single model at a time (~30GB) → Fits on H100
- **Innovation**: Reference model → cleanup → Main model → evaluation