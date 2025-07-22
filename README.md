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
8. **Test Rewards**: Verify reward functions with `python scripts/rewards.py`
9. **Train Model**: Follow the GRPO training pipeline in `docs/cap_rlvr_grpo_plan.md`

## Key Features

- **✅ Complete Gym Environments**: Full OpenAI Gym interface for all 5 legal reasoning tasks
- **✅ RLHF/GRPO Ready**: Environments integrate seamlessly with reinforcement learning training
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

## Dataset

Based on the **Caselaw Access Project (CAP)** containing millions of US court decisions, processed into structured training tasks for legal reasoning.

See `docs/cap_rlvr_grpo_plan.md` for the complete implementation plan and training details.