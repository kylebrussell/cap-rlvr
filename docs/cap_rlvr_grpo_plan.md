# CAP → Qwen Reinforcement Learning with Verifiable Rewards (RLVR)

Comprehensive engineering guide for turning the **Caselaw Access Project (CAP)** corpus + **Qwen-3-14B** into a continuously-improving legal LLM with fully deterministic rewards using **GRPO (Group Relative Policy Optimization)**.

---

## 0. Environment snapshot

| Stage               | Recommended host                                                | Why                                                   |
| ------------------- | --------------------------------------------------------------- | ----------------------------------------------------- |
| **Download + Prep** | **Vast.ai** CPU-only instance ≥ 16 vCPU / 64 GB RAM / 2 TB NVMe | Cheapest I/O box, point-and-click UI, hourly billing |
| **Training**        | Lambda Labs 4× A100-80 GB (14B path)                           | Fastest spot cost for QLoRA + GRPO                   |
| **Serving**         | CPU AWQ (32 GB RAM) *or* GPU (RTX 4090)                        | AWQ 4-bit runs at <1 sec/100 tok                     |

### SSH cheat-sheet (macOS)

```bash
# Vast.ai (keys shown in dashboard)
ssh -i ~/.ssh/vast_cap.pem root@<VAST_IP>
# Lambda Labs
ssh -i ~/.ssh/lambda_key ubuntu@<LAMBDA_IP>
```

---

## 1. Toolchain

| Component       | Version / Spec                                                                                                                                      |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python          | ≥ 3.10                                                                                                                                              |
| CUDA (training) | 12.x                                                                                                                                                |
| Key libs        | transformers >= 4.41 · datasets 2.19.1 · trl 0.7.8 · peft 0.10 · bitsandbytes 0.43 · sentence-transformers 2.7 · faiss-cpu · tiktoken · uvicorn |

```bash
conda create -n cap_rlvr python=3.10 -y && conda activate cap_rlvr
pip install transformers==4.41.0 datasets==2.19.1 trl==0.7.8 peft==0.10.0 bitsandbytes==0.43.0 sentence-transformers==2.7 faiss-cpu tiktoken uvicorn pytest
```

---

## 2. Folder layout

```text
workspace/
├── data_raw/            # CAP HF dump
├── data_tasks/          # JSONL shards per task
├── scripts/             # prep_* + reward_* files
├── envs/                # Gym wrappers
├── models/{base,sft,grpo}
└── deploy/              # merged + quantised
```

```bash
export WS=~/cap_rlvr && mkdir -p $WS/{data_raw,data_tasks,scripts,envs,models/{base,sft,grpo},deploy}
```

---

## 3. Dataset acquisition (on Vast.ai)

### Robust approach for large datasets (78GB+)

```bash
pip install huggingface_hub[cli]
# Use CLI with built-in resume capability for large datasets
huggingface-cli download common-pile/caselaw_access_project \
  --repo-type=dataset \
  --local-dir=data_raw/cap_raw \
  --resume-download
```

**Why CLI over Python API:**
- Built-in retry/resume for network failures
- Better memory efficiency (no RAM loading)
- Robust handling of connection timeouts
- Automatic chunked downloads

**Alternative streaming approach (if CLI unavailable):**
```python
# For datasets too large for memory
from datasets import load_dataset
dataset = load_dataset("common-pile/caselaw_access_project", split="train", streaming=True)
# Process in chunks to avoid memory issues
```

78 GB written to NVMe in ~25-45 min (depending on network stability).

---

## 4. Micro-Task generation (explicit scripts)

All scripts live under `scripts/`. Each emits train/eval/test JSONL.

### 4.0 Shared helpers `scripts/prep_utils.py`

```python
from typing import Iterator, Dict, List
import json, pathlib, random
from datasets import load_from_disk

CAP_PATH = pathlib.Path('../data_raw/cap')

_SPLIT = {
    'train': lambda i: i % 10 not in (8, 9),
    'eval':  lambda i: i % 10 == 8,
    'test':  lambda i: i % 10 == 9,
}

def stream_cap() -> Iterator[Dict]:
    ds = load_from_disk(CAP_PATH)
    for rec in ds:
        yield rec

def dump(task: str, rec: Dict, idx: int, out_root='../data_tasks'):
    for split, cond in _SPLIT.items():
        if cond(idx):
            p = pathlib.Path(out_root)/task
            p.mkdir(parents=True, exist_ok=True)
            with open(p/f'{split}.jsonl','a') as f:
                f.write(json.dumps(rec, ensure_ascii=False)+"\n")
```

### 4.1 Holding Selection `scripts/prep_holding_task.py`

```python
import re, random, spacy
from prep_utils import stream_cap, dump

HOLD_RE = re.compile(r'^(Held|Holding)[:\s]', re.I|re.M)
nlp = spacy.blank('en')

holds = []
for i, rec in enumerate(stream_cap()):
    m = HOLD_RE.search(rec['casebody'])
    if not m:
        continue
    sent = next(nlp(rec['casebody'][m.end():]).sents, None)
    if not sent:
        continue
    holds.append({'year': rec['decision_date'][:4], 'id': rec['case_id'], 'txt': sent.text.strip()})

for idx, pos in enumerate(holds):
    same_year = [h for h in holds if h['year']==pos['year'] and h['id']!=pos['id']]
    if len(same_year) < 4:
        continue
    distract = random.sample(same_year, 4)
    rec = {
        'case_id': pos['id'],
        'inputs': 'Choose the correct holding:',
        'choices': [pos['txt']] + [d['txt'] for d in distract],
        'answer_idx': 0
    }
    dump('holding', rec, idx)
```

### 4.2 Bluebook Fill-in `scripts/prep_bluebook_task.py`

```python
import re
from prep_utils import stream_cap, dump

U_RE = re.compile(r'(\d+) U\.S\. (\d+) \((\d{4})\)')
for idx, rec in enumerate(stream_cap()):
    cite = rec.get('citation')
    if not cite or cite.get('reporter')!='U.S.':
        continue
    rec_out={
        'case_id': rec['case_id'],
        'inputs': 'Fill in the citation: ___ U.S. ___ (___)',
        'ground_truth': f"{cite['volume']} U.S. {cite['page']} ({cite['year']})",
        'metadata': {'volume': cite['volume'], 'page': cite['page'], 'year': cite['year']}
    }
    dump('bluebook', rec_out, idx)
```

### 4.3 IRAC Summary `scripts/prep_summarise_task.py`

```python
import re, textwrap
from prep_utils import stream_cap, dump

SYL = re.compile(r'Syllabus(.*?)(?:Opinion|Held)', re.S|re.I)
MAX=2048
for idx, rec in enumerate(stream_cap()):
    body = rec['casebody'][:MAX]
    m = SYL.search(body)
    if not m: continue
    syllabus = textwrap.shorten(m.group(1), 2000)
    parties = [w for w in rec['name'].replace('v.',' ').split() if w[0].isupper()][:2]
    dump('summarise', {
        'case_id': rec['case_id'],
        'inputs': 'Summarise in <=200 words (IRAC):\n'+body,
        'ground_truth': {'syllabus': syllabus, 'key_parties': parties}
    }, idx)
```

### 4.4 Retrieval `scripts/prep_retrieval_task.py`

```python
from collections import defaultdict
from prep_utils import stream_cap, dump

tag_ix = defaultdict(list)
recs = list(stream_cap())
for r in recs:
    for t in r.get('topic_tags', []):
        tag_ix[t].append(r['case_id'])
for idx, r in enumerate(recs):
    tags=r.get('topic_tags', [])
    if not tags: continue
    pos=set()
    for t in tags:
        pos.update(tag_ix[t])
    pos.discard(r['case_id'])
    if not pos: continue
    facts = r['casebody'][:1500]
    dump('retrieval', {
        'case_id': r['case_id'],
        'inputs': 'List 5 analogous cases:\n'+facts,
        'positives': list(pos)
    }, idx)
```

### 4.5 Entail / Conflict `scripts/prep_entail_task.py`

```python
import re
from prep_utils import stream_cap, dump

KW = {'OVERRULES':'overrule|abrogat', 'DISTINGUISHES':'distinguish', 'AFFIRMS':'affirm'}
re_kw={k:re.compile(v,re.I) for k,v in KW.items()}
recs=list(stream_cap())
cite_map={r['case_id']:r for r in recs}
for idx, r in enumerate(recs):
    for cid in r.get('citations_out', []):
        tgt=cite_map.get(cid)
        if not tgt: continue
        window=r['casebody'][max(0,r['casebody'].find(tgt['name'])-150):][:300]
        lbl='NONE'
        for k,pat in re_kw.items():
            if pat.search(window): lbl=k; break
        dump('entail', {
            'pair_id': f"{r['case_id']}__{cid}",
            'inputs': 'Relation between cases:\n'+window,
            'label': lbl
        }, idx)
```

### 4.6 Run all tasks

```bash
python scripts/prep_holding_task.py
python scripts/prep_bluebook_task.py
python scripts/prep_summarise_task.py
python scripts/prep_retrieval_task.py
python scripts/prep_entail_task.py
```

Total prep ~40 min.

---

## 5. Frozen embeddings (retrieval)

```bash
python scripts/build_faiss.py --in data_tasks/retrieval/train.jsonl --out data_tasks/retrieval/embeddings.faiss
```

---

## 6. Reward Functions (Complete Implementation)

All reward functions have been implemented with comprehensive scoring mechanisms:

### 6.1 Available Reward Functions

| Task | File | Description | Reward Components |
|------|------|-------------|-------------------|
| Holding Selection | `reward_holding.py` | Multiple choice evaluation | Binary: 1.0 for correct choice, 0.0 for incorrect |
| Bluebook Citation | `reward_bluebook.py` | Citation format accuracy | Component-wise (80%) + format validation (20%) |
| IRAC Summary | `reward_irac.py` | Legal case summarization | Structure (40%) + content (30%) + length (15%) + legal language (15%) |
| Case Retrieval | `reward_retrieval.py` | Similar case finding | FAISS similarity + quantity bonus |
| Relationship | `reward_entail.py` | Case relationship classification | Exact match (60%) + context consistency (25%) + quality (15%) |

### 6.2 Unified Interface

```python
from rewards import UnifiedRewardFunction

# Initialize with optional FAISS index for retrieval
reward_fn = UnifiedRewardFunction(faiss_index_path='data_tasks/retrieval/embeddings.faiss')

# Auto-detect task type and compute reward
reward = reward_fn.reward(sample, model_output)

# Or specify task explicitly
reward = reward_fn.reward(sample, model_output, task_type='bluebook')
```

### 6.3 Individual Reward Function Usage

```python
# Holding Selection (Multiple Choice)
from reward_holding import HoldingRewardFunction
holding_reward = HoldingRewardFunction()
# Returns 1.0 for correct choice, 0.0 for incorrect

# Bluebook Citation (Fill-in-the-blank)
from reward_bluebook import BluebookRewardFunction
bluebook_reward = BluebookRewardFunction()
# Component scoring: volume, page, year, court, format validation

# IRAC Summarization (Structured text generation)
from reward_irac import IRACRewardFunction
irac_reward = IRACRewardFunction()
# Multi-component: IRAC structure detection + content quality + length + legal language

# Case Relationship Classification
from reward_entail import EntailmentRewardFunction
entail_reward = EntailmentRewardFunction()
# Classification accuracy + context consistency + reasoning quality
```

### 6.4 Reward Function Features

**Robust Response Parsing:**
- Multiple response formats (letter choices, numbers, text similarity)
- Fuzzy matching for classification tasks
- Citation component extraction with validation

**Quality Assessment:**
- Legal language usage evaluation
- Context consistency checking
- Response length and structure validation
- Partial credit for reasonable errors

**Process Supervision Ready:**
- All rewards return 0.0-1.0 float values
- Supports intermediate step evaluation
- Compatible with GRPO group ranking requirements

---

## 7. Gym Environments (Complete Implementation)

**✅ COMPLETE**: All five gym environments have been implemented with full OpenAI Gym interface compatibility.

### 7.1 Environment Structure

```text
envs/
├── __init__.py              # Package initialization with all environments
├── base_env.py             # Base environment class (BaseCapRLVREnv)
├── holding_env.py          # Holding selection environment
├── bluebook_env.py         # Bluebook citation environment  
├── summarise_env.py        # IRAC summary environment
├── retrieval_env.py        # Case retrieval environment
├── entail_env.py           # Entailment environment
└── README.md               # Complete documentation
```

### 7.2 Environment Features

**Standard Gym Interface:**
- `env.reset()` → observation dict
- `env.step(action)` → (observation, reward, done, info)
- `env.render()` → human-readable display
- `env.close()` → cleanup resources

**Unified Architecture:**
- All environments inherit from `BaseCapRLVREnv`
- Automatic reward function integration via `UnifiedRewardFunction`
- Text-based action space for natural language responses
- Flexible data loading with subset support
- Task-specific observation formatting

### 7.3 Quick Usage Examples

```python
# Individual Environment Usage
from envs import HoldingSelectionEnv, BluebookCitationEnv, CaseRetrievalEnv

# Create holding selection environment
holding_env = HoldingSelectionEnv(
    data_path="data_tasks/holding/train.jsonl",
    subset_size=1000  # Use subset for faster development
)

# Standard gym loop
obs = holding_env.reset()
model_response = "A"  # Model's choice selection
obs, reward, done, info = holding_env.step(model_response)
print(f"Reward: {reward}")

# Citation completion environment
citation_env = BluebookCitationEnv(data_path="data_tasks/bluebook/train.jsonl")
obs = citation_env.reset()
citation_response = "123 U.S. 456 (1990)"
obs, reward, done, info = citation_env.step(citation_response)

# Retrieval environment with FAISS integration
retrieval_env = CaseRetrievalEnv(
    data_path="data_tasks/retrieval/train.jsonl",
    faiss_index_path="data_tasks/retrieval/embeddings.faiss"
)
```

### 7.4 RLHF/GRPO Integration Ready

```python
# Multi-task environment wrapper for GRPO training
import random
from envs import HoldingSelectionEnv, BluebookCitationEnv, IRACsSummaryEnv

class MultiTaskLegalEnv:
    def __init__(self):
        self.envs = {
            'holding': HoldingSelectionEnv(),
            'bluebook': BluebookCitationEnv(), 
            'summarise': IRACsSummaryEnv()
        }
        self.current_env = None
        
    def reset(self):
        # Sample random task for curriculum learning
        task = random.choice(list(self.envs.keys()))
        self.current_env = self.envs[task]
        obs = self.current_env.reset()
        obs['task_name'] = task  # Add task identifier
        return obs
        
    def step(self, action):
        return self.current_env.step(action)

# Use in GRPO training
multi_env = MultiTaskLegalEnv()
for episode in range(num_episodes):
    obs = multi_env.reset()
    model_response = policy.generate(obs['inputs'])
    obs, reward, done, info = multi_env.step(model_response)
    # Process supervision: reward based on reasoning quality
    policy.update_with_reward(reward, info)
```

### 7.5 Environment Testing

All environments have been tested and validated:

```bash
# Test all environments
python test_gym_envs.py

# Results: ✅ 5/5 environments passed testing
# - Holding Selection Environment: ✓ PASSED
# - Bluebook Citation Environment: ✓ PASSED  
# - IRAC Summary Environment: ✓ PASSED
# - Case Retrieval Environment: ✓ PASSED
# - Entailment Environment: ✓ PASSED
```

### 7.6 Advanced Features

**Task-Specific Enhancements:**
- **HoldingSelectionEnv**: Formatted multiple choice display, choice extraction
- **BluebookCitationEnv**: Citation component validation, metadata handling
- **IRACsSummaryEnv**: Legal summary quality assessment
- **CaseRetrievalEnv**: FAISS similarity search, case ID extraction
- **EntailmentEnv**: Relationship classification, context analysis

**Development Support:**
- Subset training for faster iteration
- Empty dataset fallback for testing
- Comprehensive error handling and logging
- Detailed info dictionaries for analysis

---

## 8. Data Migration (Vast.ai → Lambda Labs)

### Filesystem Transfer Only

After data preparation completes on Vast.ai, transfer the prepared datasets to Lambda Labs filesystem:

```bash
# Step 1: Format data for SFT training (on Vast.ai)
cd ~/cap_rlvr/scripts
source ../cap_env/bin/activate

# Generate TRL-compatible prompt-completion pairs
python format_for_sft.py --format separate    # Individual task files
python format_for_sft.py --format unified     # Multi-task training file
python format_for_sft.py --format chat        # Chat message format

# Check formatting statistics
python format_for_sft.py --stats-only

# Step 2: Transfer data to Lambda Labs filesystem
python migrate_to_lambda.py --lambda-host your-lambda-host

# Or check data readiness first
python migrate_to_lambda.py --check-only
```

### SFT Data Formats

The `format_for_sft.py` script creates TRL-compatible datasets:

**Individual Task Format** (`--format separate`):
```json
{"prompt": "Complete this legal citation: Smith v. Jones, 123", "completion": "Smith v. Jones, 123 F.3d 456 (1st Cir. 1999)", "task": "bluebook"}
```

**Unified Multi-Task Format** (`--format unified`):
- All 5 tasks combined and shuffled
- Balanced sampling across legal reasoning types
- Ready for curriculum learning approaches

**Chat Message Format** (`--format chat`):
```json
{"messages": [{"role": "user", "content": "Complete this citation..."}, {"role": "assistant", "content": "Smith v. Jones..."}]}
```

### Migration Pipeline Features

**Data Transfer Only:**
- **Scope**: Filesystem transfer only - no training orchestration
- **Validation**: Verifies all 5 data prep tasks completed before transfer
- **Integrity**: MD5 checksums ensure accurate data transfer

**Efficient Transfer:**
- Creates compressed archive (~5-8GB from ~16GB raw)  
- Includes both raw task data and SFT-formatted datasets
- Automatic cleanup of temporary transfer files

**Lambda Labs Setup:**
- Transfers to Lambda Labs filesystem with verification
- Sets up directory structure for subsequent training steps
- Clean handoff - training steps run independently on Lambda Labs

---

## 9. Training pipeline (SFT → GRPO)

### Why GRPO over PPO for Legal Reasoning?

Group Relative Policy Optimization (GRPO) is superior to PPO for our legal reasoning tasks because:

1. **Verifiable Rewards**: GRPO's group-based comparisons align perfectly with our deterministic reward functions (citation accuracy, holding selection, etc.)
2. **Process Supervision**: GRPO naturally supports process-based rewards by comparing multiple reasoning paths for the same legal query
3. **Sample Efficiency**: GRPO requires fewer samples to converge on structured legal outputs compared to PPO's continuous advantage estimation
4. **Stability**: The relative ranking approach prevents reward hacking when dealing with formal legal language

### Warm-start SFT

Use the pre-formatted SFT datasets for efficient training:

```bash
# Multi-task SFT (recommended)
python -m trl.sft_trainer --model_name Qwen/Qwen3-14B-Instruct \
  --dataset_path data_tasks/sft_formatted/unified/train_sft_unified.jsonl \
  --dataset_text_field text --use_lora True --q_lora True \
  --batch_size 2 --accum_steps 16 --bf16 --epochs 2 --output_dir models/sft

# Single-task SFT (for task-specific models)
python -m trl.sft_trainer --model_name Qwen/Qwen3-14B-Instruct \
  --dataset_path data_tasks/sft_formatted/bluebook/train_sft.jsonl \
  --dataset_text_field text --use_lora True --q_lora True \
  --batch_size 4 --accum_steps 8 --bf16 --epochs 3 --output_dir models/sft_bluebook
```

**Dataset Format**: The formatted datasets use standard prompt-completion structure compatible with TRL's SFTTrainer, with task-specific instruction templates and proper legal language formatting.

### Data Preparation for GRPO

**✅ COMPLETE**: GRPO data preparation script has been fully implemented.

GRPO requires generating multiple candidate responses per query and ranking them. This creates the process supervision dataset:

#### Complete Implementation: `scripts/prep_grpo_dataset.py`

The script provides a comprehensive solution for GRPO dataset generation:

**Key Features:**
- **Multi-response generation**: Creates 4 candidate responses per query using different sampling parameters (temperature 0.6-0.9, top_p 0.8-0.95)
- **Unified reward scoring**: Integrates with existing UnifiedRewardFunction for consistent scoring
- **Task auto-detection**: Automatically detects task type from file path
- **Development support**: Mock mode for testing, subset processing, comprehensive logging
- **Robust error handling**: Fallback modes and detailed error reporting

**Usage Examples:**

```bash
# Process single task with SFT model
python scripts/prep_grpo_dataset.py --task bluebook --model_path models/sft --num_candidates 4

# Process all tasks with subset for development
python scripts/prep_grpo_dataset.py --task all --model_path models/sft --subset 1000

# Mock mode for testing without model loading
python scripts/prep_grpo_dataset.py --task bluebook --model_path models/sft --mock_mode

# Specify custom FAISS index for retrieval task
python scripts/prep_grpo_dataset.py --task retrieval --model_path models/sft \
  --faiss_index data_tasks/retrieval/embeddings.faiss
```

**Output Format:**
Creates structured JSON files ready for GRPO training:

```json
{
  "metadata": {
    "total_samples": 1000,
    "num_candidates_per_query": 4,
    "avg_max_score": 0.85,
    "avg_score_range": 0.3,
    "generation_model": "models/sft"
  },
  "samples": [
    {
      "query": "Complete this legal citation: Smith v. Jones, 123",
      "responses": [
        "Smith v. Jones, 123 F.3d 456 (1st Cir. 1999)",
        "Smith v. Jones, 123 U.S. 789 (1999)", 
        "Smith v. Jones, 123 F.2d 321 (1st Cir. 1998)",
        "Smith v. Jones, 123 Fed. Appx. 654 (1st Cir. 1999)"
      ],
      "scores": [0.9, 0.7, 0.8, 0.6],
      "metadata": {"volume": 123, "case_type": "circuit"},
      "sample_id": "case_12345"
    }
  ]
}
```

**Integration with Existing Pipeline:**
- Automatic FAISS index detection for retrieval tasks
- Uses existing reward functions without modification  
- Outputs compatible with GRPO training loop implementation
- Supports all 5 legal reasoning tasks (bluebook, holding, summarise, retrieval, entail)

**Generate GRPO datasets for all tasks:**
```bash
# Full pipeline after SFT training completes
cd scripts
python prep_grpo_dataset.py --task all --model_path ../models/sft
```

This creates the process supervision dataset required for GRPO's group ranking approach, with multiple scored responses per legal query.

### Multi-Stage GRPO Training Strategy

GRPO training follows a **multi-iteration curriculum** rather than single-pass training. This progressive approach ensures robust legal reasoning across all task types.

#### Stage-Based Training Pipeline

**Stage 0: Individual Task Mastery (Sequential)**
- Train each legal task separately until proficiency threshold
- Target: ≥80% reward score per task type
- Duration: ~2-3 training runs per task (5-15 epochs total)
- Purpose: Establish baseline competency in each legal reasoning domain

```bash
# Stage 0: Individual task training iterations
python scripts/train_grpo.py --task bluebook --model_path models/sft \
  --data_path data_grpo/bluebook/train_grpo.json --num_epochs 5

python scripts/train_grpo.py --task holding --model_path models/grpo/bluebook_grpo \
  --data_path data_grpo/holding/train_grpo.json --num_epochs 5

python scripts/train_grpo.py --task summarise --model_path models/grpo/holding_grpo \
  --data_path data_grpo/summarise/train_grpo.json --num_epochs 3

# Continue for retrieval and entail tasks...
```

**Stage 1: Multi-Task Integration (Iterative)**
- Combine high-performing task adapters for joint training
- Target: Maintain >75% reward across all tasks simultaneously
- Duration: 3-5 iterations with different task weightings
- Purpose: Balance competency across legal reasoning types

```bash
# Stage 1: Multi-task integration iterations
python scripts/train_grpo.py --task all --multi_task \
  --model_path models/grpo/stage0_complete \
  --data_path data_grpo/unified/train_grpo.json \
  --num_epochs 3 --learning_rate 5e-6

# Iteration 2: Adjust for weak tasks
python scripts/train_grpo.py --task all --multi_task \
  --model_path models/grpo/stage1_iter1 \
  --data_path data_grpo/unified/train_grpo.json \
  --resume_from_checkpoint models/grpo/stage1_iter1/checkpoint-1500
```

**Stage 2: Curriculum Refinement (Adaptive)**
- Progressive difficulty increase within tasks
- Error analysis and targeted improvement
- Target: >85% reward with consistent performance
- Duration: 2-4 adaptive iterations based on eval metrics

```bash
# Stage 2: Curriculum refinement with difficulty progression
python scripts/train_grpo.py --task all --multi_task \
  --model_path models/grpo/stage1_complete \
  --data_path data_grpo/curriculum/hard_cases.json \
  --beta 0.15 --learning_rate 3e-6
```

**Stage 3: Production Optimization (Final)**
- Edge case handling and robustness testing
- Hyperparameter fine-tuning for deployment
- Target: >90% reward with low variance
- Duration: 1-2 final polishing iterations

#### Iteration Decision Framework

**Continue Training If:**
- Any task reward <80% (Stage 0)
- Multi-task reward drops >5% from single-task performance (Stage 1)
- High reward variance (std >0.2) across evaluation batches
- Evaluation metrics show catastrophic forgetting

**Progress to Next Stage If:**
- All tasks meet reward thresholds for current stage
- Evaluation metrics stable for 2+ checkpoints
- No significant performance degradation on held-out test set

#### Automated Training Orchestration

**✅ COMPLETE**: Full automation support for multi-stage GRPO training has been implemented through comprehensive script enhancements.

**Enhanced Scripts for Iterative Training:**

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `train_grpo.py` | Core training + evaluation | `--eval_only` mode for stage validation |
| `validate_stage_progression.py` | Stage progression validation | Reward threshold checking, comprehensive reporting |
| `prep_grpo_dataset.py` | Dataset preparation | `--unified_output` for multi-task datasets |
| `orchestrate_grpo_training.py` | Full pipeline automation | 4-stage automated training with retry logic |

**Manual Training Iteration Monitoring:**
```bash
# Check current performance across all tasks
python scripts/train_grpo.py --task all --eval_only \
  --model_path models/grpo/current \
  --eval_data_path data_grpo/unified/eval_grpo.json

# Resume training from best checkpoint if needed
python scripts/train_grpo.py --task holding --model_path models/grpo/current \
  --resume_from_checkpoint models/grpo/holding_grpo/checkpoint-2000 \
  --num_epochs 2

# Validate stage progression manually
python scripts/validate_stage_progression.py --stage 0 --check_all_tasks \
  --model_path models/grpo/ --output_report stage0_validation.txt
```

**Automated Pipeline Execution:**
```bash
# Full automated pipeline from SFT to production
python scripts/orchestrate_grpo_training.py \
  --sft_model_path models/sft \
  --start_stage 0 \
  --output_dir models/grpo

# Resume from specific stage with existing model
python scripts/orchestrate_grpo_training.py \
  --base_model_path models/grpo/stage1_complete \
  --start_stage 2 \
  --output_dir models/grpo

# Dry run to preview pipeline execution
python scripts/orchestrate_grpo_training.py \
  --sft_model_path models/sft \
  --dry_run
```

**Multi-Task Dataset Generation:**
```bash
# Generate unified datasets for Stage 1+ training
python scripts/prep_grpo_dataset.py --task all --unified_output \
  --model_path models/sft --num_candidates 4

# Creates:
# - Individual task datasets: data_grpo/{task}/train_grpo.json
# - Unified multi-task dataset: data_grpo/unified/train_grpo.json
# - Evaluation dataset: data_grpo/unified/eval_grpo.json
```

### GRPO Training Implementation

**✅ COMPLETE**: A production-ready GRPO training script has been implemented.

The complete implementation is available as `scripts/train_grpo.py` with the following features:

#### Complete Implementation: `scripts/train_grpo.py`

**Key Features:**
- **Modern TRL Integration**: Uses current TRL library API with proper GRPOTrainer and GRPOConfig
- **Unified Reward Integration**: Seamless integration with existing UnifiedRewardFunction
- **Production-Ready**: Comprehensive error handling, logging, checkpointing, and evaluation
- **Multi-Task Support**: Can train on single tasks or combined multi-task datasets
- **Hardware Optimization**: Automatic device detection, mixed precision, gradient accumulation

**Usage Examples:**

```bash
# Single task GRPO training
python scripts/train_grpo.py --task bluebook --model_path models/sft \
  --data_path data_grpo/bluebook/train_grpo.json

# Multi-task GRPO training  
python scripts/train_grpo.py --task all --multi_task --model_path models/sft \
  --data_path data_grpo/unified/train_grpo.json

# With custom configuration
python scripts/train_grpo.py --task holding --model_path models/sft \
  --data_path data_grpo/holding/train_grpo.json \
  --batch_size 4 --learning_rate 5e-6 --num_epochs 5 --beta 0.15

# Resume from checkpoint
python scripts/train_grpo.py --task bluebook --model_path models/sft \
  --data_path data_grpo/bluebook/train_grpo.json \
  --resume_from_checkpoint models/grpo/bluebook_grpo/checkpoint-1000
```

**Configuration Options:**
- `--task`: Task type (bluebook, holding, summarise, retrieval, entail, all)
- `--model_path`: Path to SFT fine-tuned model
- `--data_path`: Path to GRPO dataset JSON file (from prep_grpo_dataset.py)
- `--eval_data_path`: Optional evaluation dataset path
- `--batch_size`: Training batch size per device (default: 2)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--num_epochs`: Number of training epochs (default: 3)
- `--beta`: GRPO KL penalty coefficient (default: 0.1)
- `--output_dir`: Model output directory (default: models/grpo)

**Training Pipeline Integration:**
```bash
# After SFT training completes and GRPO data is generated
python scripts/train_grpo.py --task bluebook --model_path models/sft \
  --data_path data_grpo/bluebook/train_grpo.json \
  --eval_data_path data_grpo/bluebook/eval_grpo.json
```

**Key Implementation Improvements:**
- ✅ **Correct TRL API Usage**: Uses proper GRPOConfig and GRPOTrainer APIs
- ✅ **Proper Model Loading**: Standard transformers model loading (not deprecated APIs)
- ✅ **Dataset Preparation**: Converts GRPO JSON format to HuggingFace Dataset format
- ✅ **Reward Function Integration**: Proper reward_funcs parameter usage
- ✅ **Training Loop**: Uses trainer.train() method (not manual loops)
- ✅ **Error Handling**: Comprehensive try/catch and validation
- ✅ **Checkpointing**: Automatic saving and resumption capabilities
- ✅ **Evaluation**: Optional evaluation dataset integration
- ✅ **Logging**: Detailed logging with legal-specific metrics
- ✅ **Memory Optimization**: Conservative batch sizes and gradient accumulation

---

## 9. Metrics Tracking

### Key Metrics to Monitor During Training

| Category | Metric | Why It Matters | Target |
| --- | --- | --- | --- |
| **Task-Specific Rewards** | | | |
| Average reward per task | Track bluebook, holding, etc. separately | Ensures balanced learning across legal tasks | >0.8 per task |
| Reward distribution | Min/max/std deviation | Detect reward gaming/exploitation | Std <0.2 |
| Task completion rate | % valid/parseable outputs | Model reliability indicator | >95% |
| **GRPO Training** | | | |
| Group ranking accuracy | Correct ranking of response quality | GRPO learning effectiveness | >75% |
| Reward gap | Best - worst response in group | Quality differentiation | >0.3 |
| Policy divergence | KL from SFT baseline | Prevents catastrophic forgetting | <5.0 |
| **Legal Quality** | | | |
| Citation format validity | % correct Bluebook format | Legal accuracy | >90% |
| Reasoning length | Avg tokens per response | Verbosity control | 100-500 |
| Hallucination rate | Non-existent case citations | Factual grounding | <5% |
| **Training Dynamics** | | | |
| Learning rate | Current LR value | Training stability | - |
| Gradient norm | L2 norm of gradients | Detect instabilities | <1.0 |
| Loss curves | SFT + GRPO policy loss | Convergence tracking | Decreasing |
| **Efficiency** | | | |
| Tokens/second | Training throughput | Hardware utilization | >1000 |
| GPU memory % | VRAM usage | Batch size optimization | 80-95% |
| Checkpoint size | LoRA adapter weights | Model growth | <2GB |

### Implementation

```python
# scripts/metrics_logger.py
from datetime import datetime
import json, wandb

class MetricsLogger:
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="cap-rlvr")
        
    def log(self, metrics, step):
        # Console output
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] Step {step}: {json.dumps(metrics, indent=2)}")
        
        # File logging
        with open('logs/metrics.jsonl', 'a') as f:
            f.write(json.dumps({'step': step, 'time': timestamp, **metrics}) + '\n')
        
        # W&B if enabled
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_task_rewards(self, task_name, rewards):
        """Log distribution of rewards for a specific task"""
        import numpy as np
        self.log({
            f'{task_name}/mean': np.mean(rewards),
            f'{task_name}/std': np.std(rewards),
            f'{task_name}/min': np.min(rewards),
            f'{task_name}/max': np.max(rewards),
            f'{task_name}/completion_rate': sum(r > 0 for r in rewards) / len(rewards)
        }, step=self.global_step)
```

### Critical Thresholds

- **Stop training if**: Citation validity <70% or hallucination rate >10%
- **Reduce LR if**: Gradient norm >2.0 for 100+ steps
- **Checkpoint if**: Any task reward improves by >5%

---

## 10. Merge, quantise, serve

### Merge LoRA Weights

```python
# Merge LoRA adapters with base model
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-14B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-14B-Instruct')

# Merge for single task
merged = PeftModel.from_pretrained(base, 'models/grpo/bluebook_final').merge_and_unload()

# Save merged model
merged.save_pretrained('deploy/qwen_cap_rlvr')
tokenizer.save_pretrained('deploy/qwen_cap_rlvr')
```

### Export to Multiple Formats

**GGUF for llama.cpp:**
```bash
# Install conversion tools
pip install gguf

# Convert to GGUF format with quantization
python -m gguf.gguf_writer \
  --model deploy/qwen_cap_rlvr \
  --output deploy/qwen_cap_rlvr.gguf \
  --quantize q4_0

# Alternative: use llama.cpp conversion script
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python convert.py ../deploy/qwen_cap_rlvr --outfile ../deploy/qwen_cap_rlvr.gguf --outtype q4_0
```

**MLX for Apple Silicon:**
```bash
# Install MLX conversion tools
pip install mlx-lm

# Convert to MLX format
python -m mlx_lm.convert \
  --hf-path deploy/qwen_cap_rlvr \
  --mlx-path deploy/qwen_cap_rlvr_mlx \
  --quantize

# Test MLX inference
python -m mlx_lm.generate \
  --model deploy/qwen_cap_rlvr_mlx \
  --prompt "Complete this citation: Smith v. Jones, 123"
```

**ONNX for Cross-Platform:**
```bash
# Install ONNX conversion tools
pip install optimum[onnxruntime]

# Convert to ONNX with quantization
optimum-cli export onnx \
  --model deploy/qwen_cap_rlvr \
  --output deploy/qwen_cap_rlvr_onnx \
  --quantize

# Test ONNX inference
python -c "
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
model = ORTModelForCausalLM.from_pretrained('deploy/qwen_cap_rlvr_onnx')
tokenizer = AutoTokenizer.from_pretrained('deploy/qwen_cap_rlvr')
"
```

**Additional Quantization Options:**
```bash
# AWQ 4-bit quantization for vLLM/TGI
pip install autoawq
python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
model = AutoAWQForCausalLM.from_pretrained('deploy/qwen_cap_rlvr')
tokenizer = AutoTokenizer.from_pretrained('deploy/qwen_cap_rlvr')
model.quantize(tokenizer, quant_config={'zero_point': True, 'q_group_size': 128})
model.save_quantized('deploy/qwen_cap_rlvr_awq')
"

# GPTQ quantization
pip install auto-gptq
python -c "
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_pretrained('deploy/qwen_cap_rlvr')
model.quantize(['your_calibration_data'])
model.save_quantized('deploy/qwen_cap_rlvr_gptq')
"
```

---

## 10. Compliance & release

1. Bundle licences.
2. Publish reward scripts for verifiability.
3. Push weights + model card to HF.

---

## 12. Multi-Stage Training Timeline

### Initial Setup (Days 1-4)
| Day | Deliverable                                 | Status |
| --- | ------------------------------------------- | ------ |
| 1   | CAP downloaded, scripts cloned              | ✅ |
| 2   | Micro-tasks JSONL ready                     | ✅ |
| 3   | **Reward functions + tests pass** | ✅ COMPLETE |
| 3.5 | **Gym environments + integration** | ✅ COMPLETE |
| 3.8 | **SFT formatting + migration pipeline** | ✅ COMPLETE |
| 3.9 | **GRPO data prep + training scripts** | ✅ COMPLETE |
| 4   | Warm-start SFT complete                     | 🔄 In Progress |

### Multi-Stage GRPO Training (Days 5-10)

**Stage 0: Individual Task Mastery (Days 5-7)**
| Task | Target | Iterations | Duration | Status |
|------|--------|------------|----------|--------|
| Bluebook Citations | ≥80% reward | 2-3 runs | 0.5 days | Pending |
| Holding Selection | ≥80% reward | 2-3 runs | 0.5 days | Pending |
| IRAC Summaries | ≥80% reward | 3-4 runs | 1.0 days | Pending |
| Case Retrieval | ≥80% reward | 2-3 runs | 0.5 days | Pending |
| Entailment Classification | ≥80% reward | 2-3 runs | 0.5 days | Pending |

**Stage 1: Multi-Task Integration (Days 8-9)**
| Iteration | Focus | Target | Duration | Status |
|-----------|-------|--------|----------|--------|
| Integration 1 | Combine all tasks | >75% all tasks | 0.5 days | Pending |
| Integration 2 | Balance weak tasks | >78% all tasks | 0.5 days | Pending |
| Integration 3 | Stability check | Consistent performance | 0.5 days | Pending |

**Stage 2: Curriculum Refinement (Day 10)**
| Phase | Focus | Target | Duration | Status |
|-------|-------|--------|----------|--------|
| Hard Cases | Edge cases, complexity | >85% reward | 0.3 days | Pending |
| Robustness | Variance reduction | Std <0.2 | 0.2 days | Pending |

**Stage 3: Production Polish (Day 11)**
| Phase | Focus | Target | Duration | Status |
|-------|-------|--------|----------|--------|
| Optimization | Hyperparameter tuning | >90% reward | 0.3 days | Pending |
| Final Evaluation | Test set validation | All metrics pass | 0.2 days | Pending |

### Deployment Pipeline (Days 12-14)
| Day | Deliverable | Status |
|-----|-------------|--------|
| 12  | Model merging + format exports | Pending |
| 13  | GGUF, MLX, ONNX conversions | Pending |
| 14  | HF release + documentation | Pending |

### Iteration Decision Points

**Stage Progression Gates:**
- **Stage 0 → Stage 1**: All individual tasks ≥80% reward
- **Stage 1 → Stage 2**: Multi-task performance >75% all tasks  
- **Stage 2 → Stage 3**: Stable performance (std <0.2) for 2+ checkpoints
- **Stage 3 → Deploy**: Production metrics >90% reward

**Fallback Protocol:**
- If any stage fails criteria → return to previous stage with adjusted hyperparameters
- Maximum 2 fallback iterations per stage before strategy revision
- Continuous eval monitoring prevents catastrophic forgetting

---

**End of full plan.**