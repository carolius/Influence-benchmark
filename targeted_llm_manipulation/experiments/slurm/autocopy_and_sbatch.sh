#!/bin/bash

# This script is pretty wild.
# It:
# - Copies the targeted_llm_manipulation directory to a temporary location 
#   (so that the code won't be modified between submitting and running the script).
# - Modifies the import statements in the Python files, so that imports will all be 
#   from the version of the code in the temporary directory.
# - Makes sure that data writing is done in the actual project directory 
#   (so you don't have to go looking for it).
# - Creates a script with all the necessary SLURM parameters in the temporary directory.
# - Submits the SLURM job.
# NOTE: it requires a bunch of variables to be set in the environment, which should be 
# done by the script that calls this one.

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --cpus)
            SLURM_CPUS_PER_TASK="$2"
            shift 2
            ;;
        --mem)
            SLURM_MEM="$2"
            shift 2
            ;;
        --gpus)
            SLURM_GPUS="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --qos)
            SLURM_QOS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if all required parameters are provided
required_params="CONFIG_NAME SLURM_CPUS_PER_TASK SLURM_MEM SLURM_GPUS GPU_TYPE SLURM_TIME SLURM_QOS"
for param in $required_params; do
    if [ -z "${!param}" ]; then
        echo "Error: --${param,,} is required"
        exit 1
    fi
done

# Python file to run (should be in `experiments` directory)
if [ "$CONFIG_NAME" = "dummy_test" ]; then
    FILE_TO_RUN="test.py"
else
    FILE_TO_RUN="run_experiment.py"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assumes this script is in targeted_llm_manipulation/experiments/slurm (three levels up from the project root)
PROJ_DIR="$( dirname "$( dirname "$( dirname "$SCRIPT_DIR" )" )" )"

# Check if /nas/ directory exists to determine if we're on the CHAI cluster
if [ -d "/nas" ]; then
    if [ "$GPU_TYPE" == "A100" ]; then
        NODE_LIST="cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu"
    elif [ "$GPU_TYPE" == "A6000" ]; then
        NODE_LIST="ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu"
    elif [ "$GPU_TYPE" == "SXM4" ]; then
        NODE_LIST="sac.ist.berkeley.edu,airl.ist.berkeley.edu"
    elif [ "$GPU_TYPE" == "PCI" ]; then
        NODE_LIST="cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu"    
    elif [ "$GPU_TYPE" == "either" ]; then
        NODE_LIST="cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu,ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu"
    elif [ "$GPU_TYPE" == "all" ]; then
        NODE_LIST="ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu"
    else
        echo "Invalid GPU type: $GPU_TYPE"
        exit 1
    fi

    NODE_PARAM="--nodelist=$NODE_LIST"
    MEM_PARAM="#SBATCH --mem=$SLURM_MEM"
    QOS="#SBATCH --qos=$SLURM_QOS"
    # If $SLURM_QOS is "scavenger", we need to specify the partition
    if [ "$SLURM_QOS" == "scavenger" ]; then
        QOS="$QOS --partition scavenger --requeue" # NOTE: for now hardcoded to always requeue
    fi
else
    # If we're on CAIS, specifying memory doesn't work, and the nodes are different so they can be ignored.
    # Also, we need to use the "single" partition or things error.
    NODE_PARAM="--partition=single"
    MEM_PARAM=""
    QOS=""
fi

# Generate timestamp
TIMESTAMP=$(date +"%m_%d_%H%M%S")
JOB_NAME="${CONFIG_NAME}_${TIMESTAMP}"
TEMP_DIR="$PROJ_DIR/tmp/tmp_$TIMESTAMP"

# Fixed SLURM params
export SLURM_NODES=1
export SLURM_NTASKS_PER_NODE=1
export SLURM_OUTPUT="$PROJ_DIR/slurm_logging/$JOB_NAME-%j.out"

# Check if we're already in the correct Conda environment
if [[ "$CONDA_DEFAULT_ENV" != "influence" ]]; then
    echo "Error: Not in the 'influence' Conda environment. Please activate it before running this script."
    exit 1
fi

echo "Using Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Define the original project directory
ORIGINAL_DIR="$PROJ_DIR/targeted_llm_manipulation"

# Create a unique temporary directory and copy the project to it
echo "Creating temporary directory: $TEMP_DIR"
mkdir -p $TEMP_DIR
cp -r $ORIGINAL_DIR $TEMP_DIR

# Modify the import statements in the tmp copy
cd $TEMP_DIR/targeted_llm_manipulation
python utils/prep_for_slurm.py . $FILE_TO_RUN

# Create JOB_NAME.sh file on the fly
cat << EOF > $JOB_NAME
#!/bin/bash
#SBATCH --output=$SLURM_OUTPUT
#SBATCH --cpus-per-task=$SLURM_CPUS_PER_TASK
#SBATCH --gpus=$SLURM_GPUS
#SBATCH --time=$SLURM_TIME
#SBATCH --nodes=$SLURM_NODES
#SBATCH --ntasks-per-node=$SLURM_NTASKS_PER_NODE
#SBATCH $NODE_PARAM
$MEM_PARAM
$QOS

# module load anaconda3
export NCCL_P2P_LEVEL=NVL
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Change to the temporary directory
cd $TEMP_DIR/targeted_llm_manipulation

# Run the Python script
python experiments/$FILE_TO_RUN --config $CONFIG_NAME.yaml --all-gpus --timestamp $TIMESTAMP

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf $TEMP_DIR
EOF

# Run the SLURM job
echo Command to run: "python experiments/$FILE_TO_RUN --config $CONFIG_NAME.yaml --all-gpus --timestamp $TIMESTAMP"
echo "About to run sbatch $TEMP_DIR/targeted_llm_manipulation/$JOB_NAME"
echo "====================CONFIG INFO===================="
python experiments/$FILE_TO_RUN --config $CONFIG_NAME.yaml --all-gpus --only-load-config
echo "====================END CONFIG INFO===================="
sbatch $TEMP_DIR/targeted_llm_manipulation/$JOB_NAME

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf $TEMP_DIR