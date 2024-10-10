#!/bin/bash

###############################################################
# PARAMETERS

CONFIG_NAMES="EI_test"

# SLURM job parameters
SLURM_CPUS_PER_TASK=10
SLURM_MEM="100gb"
SLURM_GPUS="1"
GPU_TYPE="either"
SLURM_TIME="00:05:00"
SLURM_QOS="high"

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Loop through each config name and run autocopy_and_sbatch.sh for each one
for CONFIG_NAME in $CONFIG_NAMES; do
    bash $SCRIPT_DIR/../autocopy_and_sbatch.sh --config-name "$CONFIG_NAME" --cpus "$SLURM_CPUS_PER_TASK" --mem "$SLURM_MEM" --gpus "$SLURM_GPUS" --gpu-type "$GPU_TYPE" --time "$SLURM_TIME" --qos "$SLURM_QOS"
done