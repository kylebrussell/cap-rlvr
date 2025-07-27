# Progressive GRPO Model Evaluation Report

**Generated**: 2025-07-27 05:36:52
**Methodology**: 20 samples per task, 80% accuracy threshold
**Evaluation Script**: `scripts/simple_eval_sft.py`

## Executive Summary

This report evaluates the progressive GRPO training sequence against SFT baselines to measure the cumulative effect of sequential improvements.

### Progressive Training Sequence
1. **SFT Base** → **Bluebook GRPO** (citation formatting mastery)
2. **Bluebook GRPO** → **Holding GRPO** (building on citation knowledge)  
3. **Holding GRPO** → **Summarise GRPO** (adding structured reasoning)
4. **Summarise GRPO** → **Entail GRPO** (completing legal reasoning suite)

## Results Summary

| Model | Type | Overall Accuracy | Tasks Passed | Stage 1 Ready | Base Model |
|-------|------|------------------|---------------|---------------|-------------|
| SFT 30K (Baseline) | sft_baseline | 73.8% | 2/4 | ❌ NO | N/A |
| Bluebook GRPO | grpo_progressive | 76.2% | 2/4 | ❌ NO | SFT |
| Holding GRPO | grpo_progressive | 73.8% | 2/4 | ❌ NO | Bluebook GRPO |
| Summarise GRPO | grpo_progressive | 75.0% | 2/4 | ❌ NO | Holding GRPO |
| Entail GRPO | grpo_progressive | 77.5% | 2/4 | ❌ NO | Summarise GRPO |

## Detailed Task Performance

### SFT 30K (Baseline)

**Description**: Optimal SFT model (81.3% accuracy)
**Model Path**: `models/sft_qwen3_14b_lora_30k`
**Overall Performance**: 73.8% accuracy (59/80 samples)
**Stage 1 Ready**: ❌ NO

| Task | Accuracy | Correct/Total | Status | Notes |
|------|----------|---------------|--------|---------|
| Bluebook | 0.800 | 16/20 | ✅ PASS |  |
| Holding | 0.500 | 10/20 | ❌ FAIL |  |
| Summarise | 1.000 | 20/20 | ✅ PASS |  |
| Retrieval | N/A | N/A | ⚠️ NO DATA | Missing evaluation data |
| Entail | 0.650 | 13/20 | ❌ FAIL |  |

### Bluebook GRPO

**Description**: Citation formatting specialist (2,988 training pairs)
**Model Path**: `models/grpo_30k_bluebook/qwen3-14b/grpo/bluebook`
**Base Model**: SFT
**Overall Performance**: 76.2% accuracy (61/80 samples)
**Stage 1 Ready**: ❌ NO

| Task | Accuracy | Correct/Total | Status | Notes |
|------|----------|---------------|--------|---------|
| Bluebook | 0.900 | 18/20 | ✅ PASS |  |
| Holding | 0.700 | 14/20 | ❌ FAIL |  |
| Summarise | 1.000 | 20/20 | ✅ PASS |  |
| Retrieval | N/A | N/A | ⚠️ NO DATA | Missing evaluation data |
| Entail | 0.450 | 9/20 | ❌ FAIL |  |

### Holding GRPO

**Description**: Holding selection expert (3,000 training pairs)
**Model Path**: `models/grpo_30k_holding/qwen3-14b/grpo/holding`
**Base Model**: Bluebook GRPO
**Overall Performance**: 73.8% accuracy (59/80 samples)
**Stage 1 Ready**: ❌ NO

| Task | Accuracy | Correct/Total | Status | Notes |
|------|----------|---------------|--------|---------|
| Bluebook | 0.950 | 19/20 | ✅ PASS |  |
| Holding | 0.550 | 11/20 | ❌ FAIL |  |
| Summarise | 0.950 | 19/20 | ✅ PASS |  |
| Retrieval | N/A | N/A | ⚠️ NO DATA | Missing evaluation data |
| Entail | 0.500 | 10/20 | ❌ FAIL |  |

### Summarise GRPO

**Description**: IRAC summarization expert (755 training pairs)
**Model Path**: `models/grpo/qwen3-14b/grpo/summarise`
**Base Model**: Holding GRPO
**Overall Performance**: 75.0% accuracy (60/80 samples)
**Stage 1 Ready**: ❌ NO

| Task | Accuracy | Correct/Total | Status | Notes |
|------|----------|---------------|--------|---------|
| Bluebook | 0.900 | 18/20 | ✅ PASS |  |
| Holding | 0.350 | 7/20 | ❌ FAIL |  |
| Summarise | 1.000 | 20/20 | ✅ PASS |  |
| Retrieval | N/A | N/A | ⚠️ NO DATA | Missing evaluation data |
| Entail | 0.750 | 15/20 | ❌ FAIL |  |

### Entail GRPO

**Description**: Case relationship classifier (840 training pairs)
**Model Path**: `models/grpo/qwen3-14b/grpo/entail`
**Base Model**: Summarise GRPO
**Overall Performance**: 77.5% accuracy (62/80 samples)
**Stage 1 Ready**: ❌ NO

| Task | Accuracy | Correct/Total | Status | Notes |
|------|----------|---------------|--------|---------|
| Bluebook | 0.950 | 19/20 | ✅ PASS |  |
| Holding | 0.700 | 14/20 | ❌ FAIL |  |
| Summarise | 0.950 | 19/20 | ✅ PASS |  |
| Retrieval | N/A | N/A | ⚠️ NO DATA | Missing evaluation data |
| Entail | 0.500 | 10/20 | ❌ FAIL |  |

## Progressive Training Analysis

### Performance Progression

**Progressive GRPO Training Effects:**

- **Bluebook GRPO**: 76.2% accuracy, 2 tasks passed 
- **Holding GRPO**: 73.8% accuracy, 2 tasks passed (-2.5%)
- **Summarise GRPO**: 75.0% accuracy, 2 tasks passed (+1.2%)
- **Entail GRPO**: 77.5% accuracy, 2 tasks passed (+2.5%)

### Key Insights

- **Progressive Training**: Each model builds on the previous GRPO-trained model
- **Task Specialization**: Models trained on specific legal reasoning tasks
- **Cumulative Learning**: Sequential improvements in legal domain knowledge
- **Performance Tracking**: Consistent evaluation methodology with SFT baselines

## Next Steps

1. ⚠️ **Additional Training Needed**: Best model `Entail GRPO` not ready for Stage 1
2. 🔄 **Task-Specific Improvements**: Focus on failed tasks before multi-task training
3. 📊 **Update eval-tracking.md**: Add progressive model results to main tracking file
4. 🚀 **Continue Progressive Sequence**: Use evaluation insights for next training phase

---

*Report generated by `scripts/evaluate_progressive_models.py` at 2025-07-27T05:36:52.092434*
