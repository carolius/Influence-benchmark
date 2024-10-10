#!/bin/bash

# Check if Targeted-Manipulation-and-Deception-in-LLMs folder exists
if [ ! -d "Targeted-Manipulation-and-Deception-in-LLMs" ]; then
    echo "Targeted-Manipulation-and-Deception-in-LLMs folder not found. Cloning from GitHub..."
    git clone git@github.com:carolius/Targeted-Manipulation-and-Deception-in-LLMs.git
else
    echo "Targeted-Manipulation-and-Deception-in-LLMs folder already exists."
fi

# Change directory to Targeted-Manipulation-and-Deception-in-LLMs
cd Targeted-Manipulation-and-Deception-in-LLMs

conda install -c nvidia cuda-compiler -y
sudo apt update
sudo apt install nano

# Install the package in editable mode
python3 -m pip install --upgrade pip
pip install -e .
pip install flash-attn --no-build-isolation
pip install nvitop

git config --global user.name "Micah Carroll"
git config --global user.email "mdc@berkeley.edu"

source targeted_llm_manipulation/.env && huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINGPuGqEcx5Y3DnOsn7JzIgidePgwZ9lR1r+YPj4vPAV marcu@Marcus-laptop" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

pytest

echo "Install vscode extensions"
