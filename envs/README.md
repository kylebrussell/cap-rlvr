# CAP RLVR Gym Environments

This directory contains OpenAI Gym-compatible environments for the five CAP RLVR legal reasoning tasks.

## Overview

The CAP RLVR project implements five distinct legal reasoning tasks, each with its own specialized gym environment:

1. **Holding Selection** (`HoldingSelectionEnv`) - Multiple choice holding statement selection
2. **Bluebook Citation** (`BluebookCitationEnv`) - Legal citation completion in Bluebook format
3. **IRAC Summarization** (`IRACsSummaryEnv`) - Case summarization using IRAC structure
4. **Case Retrieval** (`CaseRetrievalEnv`) - Finding analogous legal cases
5. **Entailment Classification** (`EntailmentEnv`) - Case relationship classification

## Installation

1. **Create a virtual environment:**
   ```bash
   python -m venv cap_rlvr_env
   source cap_rlvr_env/bin/activate  # On Windows: cap_rlvr_env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data (optional for testing):**
   ```bash
   cd scripts
   python prep_holding_task.py &
   python prep_bluebook_task.py &
   python prep_summarise_task.py &
   python prep_retrieval_task.py &
   python prep_entail_task.py &
   ```

## Quick Start

### Basic Usage Example

```python
from envs import HoldingSelectionEnv, BluebookCitationEnv

# Create environment
env = HoldingSelectionEnv(
    data_path="data_tasks/holding/train.jsonl",
    subset_size=100  # Use subset for faster testing
)

# Reset environment
observation = env.reset()
print(f"Task: {observation['inputs']}")

# Take action (model response)
model_response = "A"  # Example choice selection
obs, reward, done, info = env.step(model_response)

print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Info: {info}")

env.close()
```

### Environment Interface

All environments implement the standard gym interface:

- `reset()` → `observation`: Reset environment and get initial observation
- `step(action)` → `(observation, reward, done, info)`: Take action and get results
- `render(mode='human')`: Display current state
- `close()`: Clean up resources

### Observation Space

Each environment returns observations with the following structure:

```python
observation = {
    'inputs': str,      # Task prompt/question
    'task_type': int,   # Task type ID (0-4)
    'sample_id': str    # Unique sample identifier
}
```

### Action Space

Actions are text strings representing the model's response to the legal reasoning task.

## Environment Details

### 1. Holding Selection Environment

**Task**: Select the correct legal holding from multiple choice options.

```python
from envs import HoldingSelectionEnv

env = HoldingSelectionEnv(data_path="data_tasks/holding/train.jsonl")
```

**Action Format**: Letter choice (A, B, C, D) or descriptive text
**Reward**: 1.0 for correct choice, 0.0 for incorrect

### 2. Bluebook Citation Environment

**Task**: Complete legal citations in proper Bluebook format.

```python
from envs import BluebookCitationEnv

env = BluebookCitationEnv(data_path="data_tasks/bluebook/train.jsonl")
```

**Action Format**: Complete citation string (e.g., "123 U.S. 456 (1990)")
**Reward**: Score based on citation accuracy (0.0-1.0)

### 3. IRAC Summary Environment

**Task**: Summarize legal cases using IRAC (Issue, Rule, Application, Conclusion) format.

```python
from envs import IRACsSummaryEnv

env = IRACsSummaryEnv(data_path="data_tasks/summarise/train.jsonl")
```

**Action Format**: IRAC-structured summary text
**Reward**: Score based on content quality and structure (0.0-1.0)

### 4. Case Retrieval Environment

**Task**: Find analogous legal cases from a database.

```python
from envs import CaseRetrievalEnv

env = CaseRetrievalEnv(
    data_path="data_tasks/retrieval/train.jsonl",
    faiss_index_path="data_tasks/retrieval/embeddings.faiss"
)
```

**Action Format**: List of case IDs or descriptions
**Reward**: Score based on retrieval accuracy (0.0-1.0)

### 5. Entailment Environment

**Task**: Classify the relationship between two legal cases.

```python
from envs import EntailmentEnv

env = EntailmentEnv(data_path="data_tasks/entail/train.jsonl")
```

**Action Format**: Relationship label (AFFIRMS, REVERSES, DISTINGUISHES, etc.)
**Reward**: 1.0 for correct classification, 0.0 for incorrect

## Testing

Run the comprehensive test suite:

```bash
python test_gym_envs.py
```

This will test all environments and display results. The test can run with or without actual data files.

## Integration with RLHF

These environments are designed to work with reinforcement learning from human feedback (RLHF) systems:

```python
from envs import HoldingSelectionEnv

# Create environment for RL training
env = HoldingSelectionEnv(
    data_path="data_tasks/holding/train.jsonl",
    subset_size=None  # Use full dataset
)

# Your RL training loop
for episode in range(num_episodes):
    obs = env.reset()
    
    # Generate model response using your policy
    model_response = policy.generate(obs['inputs'])
    
    # Get reward
    obs, reward, done, info = env.step(model_response)
    
    # Update policy based on reward
    policy.update(reward, info)
```

## Advanced Configuration

### Subset Training

For faster development and testing, use dataset subsets:

```python
env = HoldingSelectionEnv(subset_size=1000)  # Use only 1000 samples
```

### Custom Reward Functions

The environments use the unified reward system from `scripts/rewards.py`. To customize rewards, modify the appropriate reward function in the `scripts/` directory.

### FAISS Integration

The retrieval environment requires a FAISS index for similarity search:

```bash
cd scripts
python build_faiss.py --in ../data_tasks/retrieval/train.jsonl --out ../data_tasks/retrieval/embeddings.faiss
```

## Troubleshooting

1. **Missing Data Files**: Environments will work with empty datasets for testing. Run data preparation scripts to generate actual training data.

2. **FAISS Import Error**: Install FAISS dependencies:
   ```bash
   pip install faiss-cpu sentence-transformers
   ```

3. **Memory Issues**: Use subset_size parameter to limit dataset size during development.

4. **Gym Version**: This code is tested with gym>=0.26.2. Some older versions may have compatibility issues.

## File Structure

```
envs/
├── __init__.py              # Package initialization
├── base_env.py             # Base environment class
├── holding_env.py          # Holding selection environment
├── bluebook_env.py         # Bluebook citation environment
├── summarise_env.py        # IRAC summary environment
├── retrieval_env.py        # Case retrieval environment
├── entail_env.py           # Entailment environment
└── README.md               # This file
```

Each environment file contains its specific implementation plus a test function that can be run independently.

## Contributing

When adding new features or modifying environments:

1. Update the base class (`BaseCapRLVREnv`) for shared functionality
2. Implement task-specific logic in individual environment files
3. Update reward functions in `scripts/reward_*.py`
4. Add test cases to `test_gym_envs.py`
5. Update this README with new functionality