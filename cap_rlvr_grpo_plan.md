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

```bash
pip install datasets huggingface_hub
python - <<'PY'
from datasets import load_dataset
load_dataset("common-pile/caselaw_access_project", split="train").save_to_disk("data_raw/cap")
PY
```

78 GB written to NVMe in ~25 min.

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

## 6. Reward scripts (example: Bluebook)

```python
import re
PAT=r"(\d+)\s+U\.S\.\s+(\d+)\s+\((\d{4})\)"

def reward(sample,out):
    m=re.search(PAT,out)
    if not m: return 0.0
    v,p,y=m.groups(); md=sample['metadata']
    return 0.25*(v==md['volume'])+0.25*(p==md['page'])+0.25*("U.S." in out)+0.25*(y==md['year'])
```

Add one file per task.

---

## 7. Gym environments snippets

```python
import gym, json, random, reward_bluebook
class BluebookEnv(gym.Env):
    def __init__(self, split='train'):
        self.data=[json.loads(l) for l in open(f'../data_tasks/bluebook/{split}.jsonl')]
        self.action_space=self.observation_space=gym.spaces.Text()
    def reset(self):
        self.sample=random.choice(self.data)
        return self.sample['inputs']
    def step(self,action):
        r=reward_bluebook.reward(self.sample,action)
        return '',r,True,{}
```

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
            
            # Score each candidate using our verifiable reward function
            scores = [reward_bluebook.reward(sample, cand) for cand in candidates]
            
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
import reward_bluebook

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
        
        # Score each response using our verifiable rewards
        rewards = []
        for item, responses in zip(batch, response_groups):
            query_rewards = [reward_bluebook.reward(item, resp) for resp in responses]
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
| 3   | Frozen embeddings + reward tests pass       |
| 4   | Warm-start SFT complete                     |
| 5   | GRPO stage0 done (Bluebook ≥90%)            |
| 6   | Curriculum complete, eval gate passes       |
| 7   | Merge, quantise, HF release + vLLM endpoint |

---

**End of full plan.**