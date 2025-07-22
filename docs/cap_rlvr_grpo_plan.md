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

## 8. Training pipeline (SFT → GRPO)

### Why GRPO over PPO for Legal Reasoning?

Group Relative Policy Optimization (GRPO) is superior to PPO for our legal reasoning tasks because:

1. **Verifiable Rewards**: GRPO's group-based comparisons align perfectly with our deterministic reward functions (citation accuracy, holding selection, etc.)
2. **Process Supervision**: GRPO naturally supports process-based rewards by comparing multiple reasoning paths for the same legal query
3. **Sample Efficiency**: GRPO requires fewer samples to converge on structured legal outputs compared to PPO's continuous advantage estimation
4. **Stability**: The relative ranking approach prevents reward hacking when dealing with formal legal language

### Warm-start SFT

```bash
python -m trl.sft_trainer --model_name Qwen/Qwen3-14B-Instruct \
  --dataset_path data_tasks/summarise/train.jsonl --use_lora True --q_lora True \
  --batch_size 2 --accum_steps 16 --bf16 --epochs 2 --output_dir models/sft
```

### Data Preparation for GRPO

GRPO requires generating multiple candidate responses per query and ranking them. This creates the process supervision dataset:

```python
# scripts/prep_grpo_dataset.py
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import reward_bluebook

def generate_grpo_dataset(task_file, model_path, num_candidates=4):
    """Generate multiple responses per query for GRPO training"""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    grpo_data = []
    with open(task_file) as f:
        for line in f:
            sample = json.loads(line)
            query = sample['inputs']
            
            # Generate multiple candidate responses
            candidates = []
            for _ in range(num_candidates):
                inputs = tokenizer.encode(query, return_tensors='pt')
                response = model.generate(
                    inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7
                )
                candidates.append(tokenizer.decode(response[0], skip_special_tokens=True))
            
            # Score each candidate using our unified reward function
            from rewards import UnifiedRewardFunction
            reward_fn = UnifiedRewardFunction()
            scores = [reward_fn.reward(sample, cand) for cand in candidates]
            
            grpo_data.append({
                'query': query,
                'responses': candidates,
                'scores': scores,
                'metadata': sample.get('metadata', {})
            })
    
    with open(task_file.replace('.jsonl', '_grpo.json'), 'w') as f:
        json.dump(grpo_data, f)
    
    return grpo_data

# Generate GRPO datasets for all tasks
for task in ['bluebook', 'holding', 'summarise', 'retrieval', 'entail']:
    generate_grpo_dataset(f'data_tasks/{task}/train.jsonl', 'models/sft')
```

### GRPO Training Loop

```python
from trl import AutoModelForCausalLMWithValueHead, GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer
import json, numpy as np
from rewards import UnifiedRewardFunction

tok = AutoTokenizer.from_pretrained('models/sft', padding_side='left')
mdl = AutoModelForCausalLMWithValueHead.from_pretrained('models/sft', load_in_4bit=True,
        peft_config={'r':64,'target_modules':['q_proj','v_proj']})

# Load pre-generated GRPO dataset with multiple responses per query
grpo_dataset = json.load(open('data_tasks/bluebook/train_grpo.json'))

# Configure GRPO with legal-specific parameters
grpo_config = GRPOConfig(
    model_name='models/sft',
    learning_rate=1e-5,
    batch_size=4,
    num_train_epochs=3,
    gradient_accumulation_steps=8,
    # GRPO-specific: compare groups of 4 responses
    num_responses_per_query=4,
    # Use our deterministic rewards for ranking
    reward_model=None,  # We use explicit reward functions
    # Process supervision: reward intermediate steps
    use_process_rewards=True
)

grpo_trainer = GRPOTrainer(
    model=mdl,
    tokenizer=tok,
    config=grpo_config
)

# Training loop with process supervision
for epoch in range(grpo_config.num_train_epochs):
    for i in range(0, len(grpo_dataset), grpo_config.batch_size):
        batch = grpo_dataset[i:i+grpo_config.batch_size]
        
        queries = [item['query'] for item in batch]
        response_groups = [item['responses'] for item in batch]
        
        # Score each response using our unified reward system
        from rewards import UnifiedRewardFunction
        reward_fn = UnifiedRewardFunction()
        rewards = []
        for item, responses in zip(batch, response_groups):
            query_rewards = [reward_fn.reward(item, resp) for resp in responses]
            rewards.append(query_rewards)
        
        # GRPO step: learn from relative rankings within each group
        grpo_trainer.step(queries, response_groups, rewards)
        
        # Log process metrics
        if grpo_trainer.global_step % 100 == 0:
            avg_reward = np.mean([np.max(r) for r in rewards])
            print(f"Step {grpo_trainer.global_step}: Max avg reward = {avg_reward:.3f}")

grpo_trainer.save_pretrained('models/grpo/bluebook_final')
```

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

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-14B-Instruct')
merged = PeftModel.from_pretrained(base,'models/grpo/bluebook_final').merge_and_unload()
merged.save_pretrained('deploy/qwen_cap_rlvr')
```

Quantise AWQ 4-bit then run via vLLM or TGI.

---

## 10. Compliance & release

1. Bundle licences.
2. Publish reward scripts for verifiability.
3. Push weights + model card to HF.

---

## 11. 7-day timeline

| Day | Deliverable                                 |
| --- | ------------------------------------------- |
| 1   | CAP downloaded, scripts cloned              |
| 2   | Micro-tasks JSONL ready                     |
| 3   | **✅ COMPLETE: Reward functions + tests pass** |
| 3.5 | **✅ COMPLETE: Gym environments + integration** |
| 4   | Warm-start SFT complete                     |
| 5   | GRPO stage0 done (All tasks ≥80% reward)   |
| 6   | Curriculum complete, eval gate passes       |
| 7   | Merge, quantise, HF release + vLLM endpoint |

---

**End of full plan.**