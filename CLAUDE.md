# CLAUDE.md - Project Memory

This file contains important context and instructions for working with this CAP RLVR project.

## Data Preparation Scripts - Remote Execution Instructions

### Critical Setup
When running data prep scripts on remote CPU instance (e.g., Vast.ai), follow these exact steps:

```bash
# Always run from the scripts directory with proper environment
cd ~/cap_rlvr/scripts
source ../cap_env/bin/activate

# Run individual scripts in background with logging
python prep_holding_task.py > ../logs/holding.log 2>&1 &
python prep_bluebook_task.py > ../logs/bluebook.log 2>&1 &
python prep_summarise_task.py > ../logs/summary.log 2>&1 &
python prep_retrieval_task.py > ../logs/retrieval.log 2>&1 &
python prep_entail_task.py > ../logs/entail.log 2>&1 &
```

### Why This Matters
- **Working Directory**: Scripts expect `../data_raw/cap_raw/` path to exist
- **Environment**: Requires activated Python virtual environment with dependencies
- **CPU Utilization**: Run ALL 5 scripts simultaneously to utilize multi-core systems (15+ cores)
- **Logging**: Background processes need proper output redirection

### System Requirements
- **CPU**: 15+ cores (all prep scripts can run simultaneously)
- **Memory**: 98GB+ recommended (holding task uses ~3GB, retrieval uses ~1GB)
- **Storage**: 80GB+ for CAP dataset (23GB compressed) + output files
- **Time**: ~2-3 hours to process full CAP dataset

### Monitoring Progress
```bash
# Check running processes
ps aux | grep "python prep_" | grep -v grep

# Monitor progress logs
tail -f logs/holding.log
tail -f logs/bluebook.log

# Check system resources
top -bn1 | head -10
free -h
```

### Expected Output
Each script generates train/eval/test splits in `~/cap_rlvr/data_tasks/`:
- `holding/` - Multiple choice holding selection (expect ~100K+ questions)
- `bluebook/` - Citation format completion (expect ~50K+ questions)  
- `summarise/` - IRAC case summarization (expect ~30K+ questions)
- `retrieval/` - Analogous case finding (expect ~20K+ questions)
- `entail/` - Case relationship classification (expect ~40K+ questions)

### Common Issues Avoided
- ❌ Running from wrong directory (`~/cap_rlvr` instead of `~/cap_rlvr/scripts`)
- ❌ Not activating virtual environment
- ❌ Running scripts sequentially instead of parallel
- ❌ Log file creation failures with improper redirection

## Dataset Information
- **Source**: Caselaw Access Project (CAP) via HuggingFace
- **Size**: 78GB uncompressed, 23GB compressed (173 files)
- **Records**: ~7 million legal case documents
- **Download Method**: Use `downloads/cli_download.py` for robust acquisition with resume capability

## Next Steps After Data Prep
1. Verify all 5 task types generated successfully
2. Run reward function tests on sample outputs
3. Begin SFT warm-start training on Qwen-3-14B
4. Implement GRPO training pipeline with process supervision