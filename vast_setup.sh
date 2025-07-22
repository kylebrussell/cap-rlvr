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
    huggingface_hub

# Download spacy model for holding extraction
python -m spacy download en_core_web_sm

echo "=== Environment ready! ==="
echo "Next steps:"
echo "1. cd ~/cap_rlvr"
echo "2. source cap_env/bin/activate"
echo "3. Start data download script"