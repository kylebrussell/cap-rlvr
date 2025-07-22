#!/bin/bash
# Vast.ai setup script for CAP data download and prep

echo "=== Setting up CAP RLVR environment ==="

# Update system
apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    tmux \
    htop \
    iotop \
    ncdu

# Create workspace
export WS=~/cap_rlvr
mkdir -p $WS/{data_raw,data_tasks,scripts,envs,models/{base,sft,grpo},deploy,logs}
cd $WS

# Setup Python environment
python3 -m venv cap_env
source cap_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install \
    transformers==4.41.0 \
    datasets==2.19.1 \
    trl==0.7.8 \
    peft==0.10.0 \
    sentence-transformers==2.7 \
    faiss-cpu \
    tiktoken \
    spacy \
    "huggingface_hub[cli]"

# Download spacy model for holding extraction
python -m spacy download en_core_web_sm

echo "=== Environment ready! ==="
echo ""
echo "To download CAP dataset (78GB) robustly:"
echo "cd ~/cap_rlvr"
echo "source cap_env/bin/activate"
echo ""
echo "# Robust CLI download with auto-resume:"
echo "huggingface-cli download common-pile/caselaw_access_project \\"
echo "  --repo-type=dataset \\"
echo "  --local-dir=data_raw/cap_raw \\"
echo "  --resume-download"
echo ""
echo "# Alternative if CLI fails - streaming approach:"
echo "python3 -c \""
echo "from datasets import load_dataset"
echo "ds = load_dataset('common-pile/caselaw_access_project', split='train', streaming=True)"
echo "# Process in chunks to avoid memory issues"
echo "\""