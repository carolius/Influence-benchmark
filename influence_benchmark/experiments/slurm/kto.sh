#!/bin/bash

###############################################################
# PARAMETERS

# Python file to run (should be in `experiments` directory)
FILE_TO_RUN="KTO_experiment.py"

# By default, have the slurm job name be the same as the Python file
JOB_NAME=$FILE_TO_RUN

# SLURM job parameters
SLURM_OUTPUT="slurm/%j.out"
SLURM_CPUS_PER_TASK=64
SLURM_MEM="250gb"
SLURM_GPUS="A6000:4"
SLURM_TIME="0:30:00"
SLURM_NODES=1
SLURM_NTASKS_PER_NODE=1

###############################################################

export NCCL_P2P_LEVEL=NVL

# Get the current username
CURRENT_USER=$(whoami)
PROJ_DIR="/nas/ucb/$CURRENT_USER/Influence-benchmark"

# Check if we're already in the correct Conda environment
if [[ "$CONDA_DEFAULT_ENV" != "influence" ]]; then
    echo "Error: Not in the 'influence' Conda environment. Please activate it before running this script."
    exit 1
fi

echo "Using Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Define the original project directory
ORIGINAL_DIR="$PROJ_DIR/influence_benchmark"

# Create a unique temporary directory
TEMP_DIR="$PROJ_DIR/tmp/tmp_$(date +%m_%d_%H%M%S)"
mkdir -p $TEMP_DIR

# Copy the project directory to the temporary location
cp -r $ORIGINAL_DIR $TEMP_DIR

# Change to the temporary directory
cd $TEMP_DIR/influence_benchmark

# Run the import modification script
python utils/prep_for_slurm.py . $FILE_TO_RUN

# Create JOB_NAME.sh file on the fly
cat << EOF > $JOB_NAME
#!/bin/bash
#SBATCH --output=$SLURM_OUTPUT
#SBATCH --cpus-per-task=$SLURM_CPUS_PER_TASK
#SBATCH --mem=$SLURM_MEM
#SBATCH --gpus=$SLURM_GPUS
#SBATCH --time=$SLURM_TIME
#SBATCH --nodes=$SLURM_NODES
#SBATCH --ntasks-per-node=$SLURM_NTASKS_PER_NODE

# Get the current username
CURRENT_USER=$(whoami)
PROJ_DIR="/nas/ucb/\$CURRENT_USER/Influence-benchmark"

# module load anaconda3
export NCCL_P2P_LEVEL=NVL
conda activate influence
echo "Conda environment: \$CONDA_DEFAULT_ENV"

# Get the file to run and the temporary directory from command-line arguments
FILE_TO_RUN=\$1
TEMP_DIR=\$2/influence_benchmark

# Change to the temporary directory
cd \$TEMP_DIR

# Run the Python script
srun python experiments/\$FILE_TO_RUN

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf \$TEMP_DIR
EOF

# Make the slurm_job.sh file executable
# chmod +x $JOB_NAME

# Run the SLURM job
sbatch $JOB_NAME $FILE_TO_RUN $TEMP_DIR

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf $TEMP_DIR