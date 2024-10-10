#!/bin/bash

# Check if a run name was provided
if [ $# -eq 0 ]; then
    echo "Error: No run name provided."
    echo "Usage: $0 <run_name>"
    exit 1
fi

# Assign the run name to a variable
RUN_NAME="$1"

# Define the source directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"  # Parent of the script's directory
TRAJS_DIR="${SRC_DIR}/data/trajectories/${RUN_NAME}"
MODELS_DIR="${SRC_DIR}/data/models/${RUN_NAME}"

# Define the destination on the remote server
REMOTE_USER="adhyyan"
REMOTE_HOST="rnn.ist.berkeley.edu"
DEST_DIR_TRAJS="/nas/ucb/$REMOTE_USER/Targeted-Manipulation-and-Deception-in-LLMs/data/trajectories/${RUN_NAME}"
DEST_DIR_MODELS="/nas/ucb/$REMOTE_USER/Targeted-Manipulation-and-Deception-in-LLMs/data/models/${RUN_NAME}"

# Function to copy a directory using rsync
copy_directory() {
    local src="$1"
    local dest="$2"
    
    if [ ! -d "$src" ]; then
        echo "Warning: Source directory $src does not exist. Skipping."
        return
    fi
    
    echo "Copying $src to $REMOTE_HOST:$dest"
    rsync -avz --progress -e ssh "$src/" "${REMOTE_USER}@${REMOTE_HOST}:${dest}"
    
    if [ $? -eq 0 ]; then
        echo "Successfully copied $src"
    else
        echo "Error: Failed to copy $src"
    fi
}

# Function to create remote directory
create_remote_dir() {
    local dir="$1"
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p $dir"
}

# Create remote directories
create_remote_dir "$DEST_DIR_TRAJS"
create_remote_dir "$DEST_DIR_MODELS"

# Copy directories in parallel
copy_directory "${TRAJS_DIR}" "${DEST_DIR_TRAJS}" &
copy_directory "${MODELS_DIR}" "${DEST_DIR_MODELS}" &

# Wait for both copy operations to complete
wait

echo "Copy operations completed."