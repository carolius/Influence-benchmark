#!/bin/bash

# Maximum number of parallel processes
MAX_PARALLEL=4

# Function to copy files for a single run
copy_run() {
    RUN_NAME=$1
    echo "Starting copy for $RUN_NAME..."
    
    echo "  Copying trajectories for $RUN_NAME..."
    cp -r "/nas/ucb/micah/Targeted-Manipulation-and-Deception-in-LLMs/data/trajectories/$RUN_NAME" "/nas/ucb/marcuswilliams/Targeted-Manipulation-and-Deception-in-LLMs/data/trajectories/$RUN_NAME"
    
    echo "  Copying models for $RUN_NAME..."
    cp -r "/nas/ucb/micah/Targeted-Manipulation-and-Deception-in-LLMs/data/models/$RUN_NAME" "/nas/ucb/marcuswilliams/Targeted-Manipulation-and-Deception-in-LLMs/data/models/$RUN_NAME"
    
    echo "Finished copying $RUN_NAME"
}

echo "Starting copy process with maximum $MAX_PARALLEL parallel jobs"
echo "Number of runs to process: $#"

# Counter for completed jobs
completed=0

# Loop through all arguments
for RUN_NAME in "$@"
do
    # Run the copy_run function in the background
    copy_run "$RUN_NAME" &

    # If we've reached the maximum number of parallel processes, wait for one to finish
    if [[ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL ]]; then
        wait -n
        ((completed++))
        echo "Completed $completed out of $# runs"
    fi
done

# Wait for all remaining background jobs to finish
wait

echo "All copy jobs completed successfully!"