# CAP → Qwen Reinforcement Learning with Verifiable Rewards (RLVR)

Engineering guide for turning the **Caselaw Access Project (CAP)** corpus + **Qwen-3-14B** into a continuously-improving legal LLM with fully deterministic rewards using **GRPO (Group Relative Policy Optimization)**.

## Project Structure

```
cap-rlvr/
├── README.md                    # This file
├── CLAUDE.md                    # Project instructions and SSH setup
├── requirements.txt             # Python dependencies for gym environments
├── test_gym_envs.py            # Comprehensive test suite for all environments
├── docs/
│   └── cap_rlvr_grpo_plan.md   # Comprehensive implementation plan
├── scripts/
│   ├── vast_setup.sh           # Remote system setup script
│   ├── prep_utils.py           # Shared utilities for data preparation
│   ├── prep_holding_task.py    # Generate holding selection tasks
│   ├── prep_bluebook_task.py   # Generate citation format tasks
│   ├── prep_summarise_task.py  # Generate IRAC summarization tasks
│   ├── prep_retrieval_task.py  # Generate case retrieval tasks
│   ├── prep_entail_task.py     # Generate case relationship tasks
│   ├── build_faiss.py          # Build FAISS index for retrieval evaluation
│   ├── format_for_sft.py       # ✅ Format task data for SFT training (TRL-compatible)
│   ├── migrate_to_lambda.py    # ✅ Automated Vast.ai -> Lambda Labs migration
│   ├── prep_grpo_dataset.py    # ✅ Generate GRPO training datasets with scored responses
│   ├── train_grpo.py           # ✅ Complete GRPO training implementation with eval-only mode
│   ├── validate_stage_progression.py # ✅ NEW: Validate reward thresholds for stage progression
│   ├── orchestrate_grpo_training.py  # ✅ NEW: Automated multi-stage training pipeline
│   ├── reward_holding.py       # Reward function for holding selection
│   ├── reward_bluebook.py      # Reward function for citation completion
│   ├── reward_irac.py          # Reward function for IRAC summarization
│   ├── reward_retrieval.py     # Reward function for case retrieval
│   ├── reward_entail.py        # Reward function for relationship classification
│   └── rewards.py              # Unified reward interface for all tasks
├── envs/                       # ✅ NEW: OpenAI Gym environments
│   ├── __init__.py             # Environment package initialization
│   ├── base_env.py             # Base environment class (BaseCapRLVREnv)
│   ├── holding_env.py          # Holding selection environment
│   ├── bluebook_env.py         # Bluebook citation environment
│   ├── summarise_env.py        # IRAC summary environment
│   ├── retrieval_env.py        # Case retrieval environment
│   ├── entail_env.py           # Entailment environment
│   └── README.md               # Environment documentation and usage
├── downloads/
│   ├── cli_download.py         # Robust CLI-based dataset download
│   ├── robust_download.py      # Alternative download with retry logic
│   └── streaming_download.py   # Memory-efficient streaming download
└── cap_rlvr_env/               # Virtual environment with gym dependencies
```

## Quick Start

### Local Development Setup
1. **Install Gym Environment Dependencies**:
   ```bash
   python -m venv cap_rlvr_env
   source cap_rlvr_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Test Gym Environments**:
   ```bash
   python test_gym_envs.py
   # Expected: ✅ 5/5 environments passed testing
   ```

3. **Use Individual Environments**:
   ```python
   from envs import HoldingSelectionEnv, BluebookCitationEnv
   
   # Create environment
   env = HoldingSelectionEnv(subset_size=100)
   obs = env.reset()
   reward = env.step("A")[1]  # Model chooses option A
   ```

### Remote Data Preparation (Vast.ai)
4. **Setup Environment**: Use `scripts/vast_setup.sh` on a remote system with sufficient storage (80GB+ for CAP dataset)
5. **Download Dataset**: Run `downloads/cli_download.py` for robust CAP dataset acquisition
6. **Prepare Tasks**: Execute all `scripts/prep_*.py` scripts to generate training data
7. **Build Embeddings**: Run `scripts/build_faiss.py` to create retrieval index
8. **Format for SFT**: Generate TRL-compatible datasets with `scripts/format_for_sft.py`

### Data Migration (Vast.ai → Lambda Labs)
9. **Transfer Data**: Copy prepared datasets to Lambda Labs filesystem with `scripts/migrate_to_lambda.py`

### Training Pipeline (Lambda Labs)
10. **SFT Training**: Complete supervised fine-tuning using transferred datasets
11. **Generate GRPO Data**: Create multi-response datasets with `scripts/prep_grpo_dataset.py` using SFT model
12. **GRPO Training**: Execute reinforcement learning with `scripts/train_grpo.py` using the generated datasets

## Key Features

- **✅ Complete Gym Environments**: Full OpenAI Gym interface for all 5 legal reasoning tasks
- **✅ RLHF/GRPO Ready**: Environments integrate seamlessly with reinforcement learning training
- **✅ Automated Migration Pipeline**: Seamless Vast.ai → Lambda Labs data transfer with verification
- **✅ TRL-Compatible SFT Formatting**: Ready-to-use prompt-completion datasets for supervised fine-tuning
- **Robust Dataset Download**: Multiple approaches for handling 78GB CAP dataset with resume capability
- **Multi-Task Training Data**: 5 legal reasoning tasks (holdings, citations, summaries, retrieval, relationships)
- **✅ Complete Reward System**: Deterministic scoring functions for all task types with unified interface
- **Process Supervision**: GRPO training with group-based reward comparisons
- **Production Ready**: Quantization, serving, and deployment pipeline included

## Legal Tasks Generated

1. **Holding Selection**: Multiple-choice questions identifying correct legal holdings
2. **Bluebook Citations**: Fill-in-the-blank citation format completion
3. **IRAC Summaries**: Structured case summarization using Issue-Rule-Application-Conclusion
4. **Case Retrieval**: Finding analogous cases based on legal concepts
5. **Relationship Classification**: Determining how cases relate (overrule, distinguish, affirm, etc.)

## Reward Functions

All task types have comprehensive, deterministic reward functions:

| Task | Reward Components | Score Range |
|------|-------------------|-------------|
| **Holding Selection** | Binary accuracy (correct choice = 1.0) | 0.0 - 1.0 |
| **Bluebook Citation** | Component accuracy (80%) + format validation (20%) | 0.0 - 1.0 |
| **IRAC Summary** | Structure (40%) + content (30%) + length (15%) + legal language (15%) | 0.0 - 1.0 |
| **Case Retrieval** | FAISS similarity matching + quantity bonus | 0.0 - 1.0 |
| **Relationship** | Classification accuracy (60%) + context consistency (25%) + quality (15%) | 0.0 - 1.0 |

Use the unified interface:
```python
from scripts.rewards import UnifiedRewardFunction
reward_fn = UnifiedRewardFunction()
score = reward_fn.reward(sample, model_output)  # Auto-detects task type
```

## Gym Environments

**✅ Complete Implementation**: All 5 legal reasoning tasks have been implemented as OpenAI Gym environments.

### Environment Features
- **Standard Gym Interface**: `reset()`, `step(action)`, `render()`, `close()`
- **Text-Based Actions**: Natural language model responses as actions
- **Unified Reward Integration**: Automatic scoring using the reward functions
- **Flexible Data Loading**: Support for subsets during development
- **Task-Specific Observations**: Formatted prompts and context for each legal task
- **RLHF/GRPO Ready**: Direct integration with reinforcement learning pipelines

### Available Environments
| Environment | Task | Action Format | Reward Range |
|-------------|------|---------------|--------------|
| `HoldingSelectionEnv` | Multiple choice holding selection | Letter choice (A,B,C,D) or text | 0.0 - 1.0 |
| `BluebookCitationEnv` | Legal citation completion | Complete citation string | 0.0 - 1.0 |
| `IRACsSummaryEnv` | Structured case summarization | IRAC-formatted text | 0.0 - 1.0 |
| `CaseRetrievalEnv` | Analogous case finding | List of case IDs/descriptions | 0.0 - 1.0 |
| `EntailmentEnv` | Case relationship classification | Relationship label (AFFIRMS, etc.) | 0.0 - 1.0 |

### Usage Examples
```python
# Individual task environment
from envs import HoldingSelectionEnv
env = HoldingSelectionEnv(data_path="data_tasks/holding/train.jsonl")
obs = env.reset()
reward = env.step("A")[1]

# Multi-task training setup
from envs import *
environments = {
    'holding': HoldingSelectionEnv(),
    'citation': BluebookCitationEnv(), 
    'summary': IRACsSummaryEnv(),
    'retrieval': CaseRetrievalEnv(),
    'entailment': EntailmentEnv()
}

# GRPO/RLHF training integration
for task, env in environments.items():
    obs = env.reset()
    model_response = policy.generate(obs['inputs'])
    obs, reward, done, info = env.step(model_response)
    policy.update(reward, info)
```

See `envs/README.md` for comprehensive documentation and advanced usage patterns.

## SFT Data Formatting & Migration

### TRL-Compatible Dataset Generation
The `format_for_sft.py` script converts raw task data into ready-to-use SFT datasets:

```bash
# Generate all SFT format variants
python scripts/format_for_sft.py --format separate    # Individual task files
python scripts/format_for_sft.py --format unified     # Multi-task training
python scripts/format_for_sft.py --format chat        # Chat message format

# View statistics without saving
python scripts/format_for_sft.py --stats-only
```

**Output Formats:**
- **Separate**: `data_tasks/sft_formatted/bluebook/train_sft.jsonl` (per-task training)
- **Unified**: `data_tasks/sft_formatted/unified/train_sft_unified.jsonl` (multi-task)
- **Chat**: `data_tasks/sft_formatted/chat_format/` (messages format for newer models)

### Automated Data Migration Pipeline
Transfer processed data from Vast.ai CPU instances to Lambda Labs filesystem:

```bash
# Check data readiness on Vast.ai
python scripts/migrate_to_lambda.py --check-only

# Transfer data to Lambda Labs
python scripts/migrate_to_lambda.py --lambda-host your-lambda-host

# Test migration without executing
python scripts/migrate_to_lambda.py --dry-run --lambda-host test-host
```

**Migration Features:**
- **Data Transfer Only**: Copies prepared datasets to Lambda Labs filesystem
- **Validation**: Verifies all 5 data prep tasks completed before transfer
- **Compression**: Creates efficient archive (~5-8GB from ~16GB raw)
- **Integrity**: MD5 checksums ensure data transfer accuracy  
- **Clean Transfer**: Both raw task data and SFT-formatted datasets
- **No Training Steps**: Migration script only handles data movement, not training orchestration

## GRPO Dataset Generation (Lambda Labs)

After SFT training completes, generate multi-response datasets for GRPO training using the fine-tuned model:

```bash
# Run on Lambda Labs GPU instance with SFT model
python scripts/prep_grpo_dataset.py --task all --model_path models/sft --num_candidates 4

# For development/testing with subset
python scripts/prep_grpo_dataset.py --task bluebook --model_path models/sft --subset 1000

# Mock mode for testing script without model loading
python scripts/prep_grpo_dataset.py --task bluebook --model_path models/sft --mock_mode
```

**Key Features:**
- **Multi-response generation**: Creates 4 candidate responses per query using different sampling parameters
- **Unified reward scoring**: Integrates with existing reward functions for consistent evaluation
- **GPU-optimized**: Designed to run on Lambda Labs GPU instances with model loaded in memory
- **Flexible output**: Generates JSON files with scored response groups ready for GRPO training

**Output:** Creates `data_grpo/{task}/train_grpo.json` files with multiple scored responses per query, enabling GRPO's group-based ranking approach.

## GRPO Training (Lambda Labs)

Execute reinforcement learning training using the complete GRPO implementation:

```bash
# Single task GRPO training
python scripts/train_grpo.py --task bluebook --model_path models/sft \
  --data_path data_grpo/bluebook/train_grpo.json

# Multi-task GRPO training with evaluation
python scripts/train_grpo.py --task all --multi_task --model_path models/sft \
  --data_path data_grpo/unified/train_grpo.json \
  --eval_data_path data_grpo/unified/eval_grpo.json

# Custom configuration for production training
python scripts/train_grpo.py --task holding --model_path models/sft \
  --data_path data_grpo/holding/train_grpo.json \
  --batch_size 4 --learning_rate 5e-6 --num_epochs 5
```

**Key Features:**
- **Production-Ready**: Complete error handling, checkpointing, and resumption
- **Multi-Task Support**: Train on individual tasks or combined datasets
- **Memory Optimized**: Conservative batch sizes and gradient accumulation for large models
- **Legal-Specific Metrics**: Custom callbacks and logging for legal reasoning evaluation
- **TRL Integration**: Modern TRL library compatibility with proper GRPO implementation
- **Evaluation-Only Mode**: Stage progression validation with `--eval_only` flag

## Multi-Stage Training Automation

The project includes comprehensive automation for the iterative GRPO training pipeline:

### Manual Stage Management
```bash
# Generate unified multi-task datasets
python scripts/prep_grpo_dataset.py --task all --unified_output --model_path models/sft

# Validate stage progression
python scripts/validate_stage_progression.py --stage 0 --check_all_tasks --model_path models/grpo/

# Evaluation-only mode for testing
python scripts/train_grpo.py --eval_only --task all --model_path models/grpo/current
```

### Fully Automated Pipeline
```bash
# Complete 4-stage automated training from SFT to production
python scripts/orchestrate_grpo_training.py --sft_model_path models/sft --start_stage 0

# Resume from specific stage
python scripts/orchestrate_grpo_training.py --base_model_path models/grpo/stage1_complete --start_stage 2

# Preview execution plan
python scripts/orchestrate_grpo_training.py --sft_model_path models/sft --dry_run
```

**Automation Features:**
- **4-Stage Pipeline**: Individual mastery → Multi-task integration → Curriculum refinement → Production optimization
- **Auto-Validation**: Reward thresholds checked automatically between stages
- **Smart Retry Logic**: Failed stages retry with adjusted parameters (max 2 retries per stage)
- **Multi-Hour Training Support**: Enhanced monitoring for long-duration runs (up to 6 hours per stage)
- **Graceful Shutdown**: Signal handling for clean interruption and resumption
- **Progress Persistence**: Training state saved to disk for crash recovery
- **Resource Monitoring**: Memory and system resource tracking during execution
- **Heartbeat Logging**: Progress updates every 10 minutes during long runs
- **Comprehensive Logging**: Detailed progress tracking and error reporting
- **Flexible Resumption**: Start from any stage with appropriate base model

**Stage Progression Thresholds:**
- **Stage 0**: ≥80% reward per individual task
- **Stage 1**: ≥75% reward across all tasks simultaneously
- **Stage 2**: ≥85% reward with variance <0.15
- **Stage 3**: ≥90% reward with variance <0.10

## Dataset

Based on the **Caselaw Access Project (CAP)** containing millions of US court decisions, processed into structured training tasks for legal reasoning.

See `docs/cap_rlvr_grpo_plan.md` for the complete implementation plan and training details.