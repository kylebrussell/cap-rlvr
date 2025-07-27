#!/bin/bash

# Multi-task GRPO training script using Entail GRPO as base
# Waits for dataset generation to complete, then starts training

echo "🚀 Multi-task GRPO Training Launcher"
echo "Base Model: Entail GRPO (best performing from progressive sequence)"
echo "Training Samples: 5000 (1000 per task)"
echo "Target: Stage 1 Multi-task Integration"
echo "=========================================="

# Wait for dataset generation to complete
echo "⏳ Waiting for dataset generation to complete..."
while [ ! -f data_grpo/unified/train_grpo.json ] || pgrep -f "prep_grpo_dataset.py" > /dev/null; do
    if pgrep -f "prep_grpo_dataset.py" > /dev/null; then
        echo "📊 Dataset generation in progress..."
        # Show current progress
        if [ -f logs/grpo_multitask_entail_generation.log ]; then
            tail -1 logs/grpo_multitask_entail_generation.log | grep -o "Processing.*" || echo "Still loading model..."
        fi
    else
        echo "⚠️  Dataset generation process not found, checking if data exists..."
    fi
    sleep 60  # Check every minute
done

echo "✅ Dataset generation completed!"

# Verify dataset
if [ -f data_grpo/unified/train_grpo.json ]; then
    echo "📊 Dataset verification:"
    python -c "
import json
with open('data_grpo/unified/train_grpo.json', 'r') as f:
    data = json.load(f)
metadata = data['metadata']
print(f'  Total samples: {metadata[\"total_samples\"]}')
print(f'  Generation model: {metadata[\"generation_model\"]}')
print(f'  Candidates per query: {metadata[\"num_candidates_per_query\"]}')
print(f'  Average max score: {metadata.get(\"avg_max_score\", \"N/A\")}')
"
else
    echo "❌ Dataset file not found!"
    exit 1
fi

# Start multi-task GRPO training
echo ""
echo "🔥 Starting Multi-task GRPO Training..."
echo "Base Model: models/grpo/qwen3-14b/grpo/entail"
echo "Output: models/grpo/qwen3-14b/grpo/multitask"
echo ""

python scripts/train_grpo.py \
    --task all \
    --multi_task \
    --model_path models/grpo/qwen3-14b/grpo/entail \
    --data_path data_grpo/unified/train_grpo.json \
    --output_dir models/grpo/qwen3-14b/grpo/multitask \
    --batch_size 2 \
    --learning_rate 5e-6 \
    --num_epochs 3 \
    --beta 0.1 \
    2>&1 | tee logs/grpo_multitask_training.log

echo ""
echo "🎯 Multi-task GRPO training completed!"
echo "📊 Check logs/grpo_multitask_training.log for detailed results"
echo "📁 Model saved to: models/grpo/qwen3-14b/grpo/multitask"