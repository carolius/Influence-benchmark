#!/bin/bash

###############################################################
# PARAMETERS

CONFIG_NAMES="KTO_mixed_therapist1t_env_20p" # Space-separated list of config names

# SLURM job parameters
SLURM_CPUS_PER_TASK=8
SLURM_MEM="64gb" # May require up to 100gb per GPU for bigger models
SLURM_GPUS="4" # IF USING MORE GPUS, NEED TO INCREASE MEMORY PROPORTIONALLY
GPU_TYPE="either" # A100 (faster generation) or A6000 (often more available), "either" (for either A100 or A6000), 
#or "all" (for all available GPUs, will break most jobs). PCI for A100-PCI and SXM4 for A100-SXM4. SXM4 should be used for training LLama-3.1-70B
SLURM_TIME="20:00:00"
SLURM_QOS="default" # can set to high if this is blocking your progress and you only need one/two jobs to run

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Loop through each config name and run autocopy_and_sbatch.sh for each one
for CONFIG_NAME in $CONFIG_NAMES; do
    bash $SCRIPT_DIR/autocopy_and_sbatch.sh --config-name "$CONFIG_NAME" --cpus "$SLURM_CPUS_PER_TASK" --mem "$SLURM_MEM" --gpus "$SLURM_GPUS" --gpu-type "$GPU_TYPE" --time "$SLURM_TIME" --qos "$SLURM_QOS"
done