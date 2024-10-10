#!/bin/bash

###############################################################
# PARAMETERS

CONFIG_NAMES="dummy_test"

# SLURM job parameters
SLURM_CPUS_PER_TASK=1
SLURM_MEM="1gb"
SLURM_GPUS="0"
GPU_TYPE="all" # sinfo -o "%n %G" to see available nodes
SLURM_TIME="00:00:30"
SLURM_QOS="high"

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Loop through each config name and run autocopy_and_sbatch.sh for each one
for CONFIG_NAME in $CONFIG_NAMES; do
    bash $SCRIPT_DIR/../autocopy_and_sbatch.sh --config-name "$CONFIG_NAME" --cpus "$SLURM_CPUS_PER_TASK" --mem "$SLURM_MEM" --gpus "$SLURM_GPUS" --gpu-type "$GPU_TYPE" --time "$SLURM_TIME" --qos "$SLURM_QOS"
done