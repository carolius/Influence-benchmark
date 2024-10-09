#!/bin/bash

# Define the source and destination
SOURCE_FILE="/root/Influence-benchmark/notebooks/data_for_figures/cross_env_gen_eval_politics-10-01_06-02_gpt.pkl"
DEST_FILE="/nas/ucb/adhyyan/Influence-benchmark/notebooks/data_for_figures/cross_env_gen_eval_politics-10-01_06-02_gpt.pkl"

# Define the remote server details
REMOTE_USER="adhyyan"
REMOTE_HOST="rnn.ist.berkeley.edu"

# Function to copy a file using rsync
copy_file() {
    local src="$1"
    local dest="$2"
    
    if [ ! -f "$src" ]; then
        echo "Error: Source file $src does not exist."
        exit 1
    fi
    
    echo "Copying $src to $REMOTE_HOST:$dest"
    rsync -avz --progress -e ssh "$src" "${REMOTE_USER}@${REMOTE_HOST}:${dest}"
    
    if [ $? -eq 0 ]; then
        echo "Successfully copied $src"
    else
        echo "Error: Failed to copy $src"
        exit 1
    fi
}

# Function to create remote directory
create_remote_dir() {
    local dir="$(dirname "$1")"
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p $dir"
}

# Create remote directory
create_remote_dir "$DEST_FILE"

# Copy file
copy_file "$SOURCE_FILE" "$DEST_FILE"

echo "File transfer completed."