#!/bin/bash

# Define the tasks, FT_types, and SP_types
tasks=("sentiment")
FT_types=("SFT" "LoRA")
SP_types=("Same_SP")

# Function to kill all GPU-related processes
kill_gpu_processes() {
    # Get all the process IDs that are using CUDA and kill them
    for pid in $(nvidia-smi | grep ' C ' | awk '{print $5}'); do
        echo "Killing process ID: $pid"
        kill -9 $pid
    done
}

# Iterate over all combinations of task, FT_type, and SP
for task in "${tasks[@]}"; do
    for FT_type in "${FT_types[@]}"; do
        for SP_type in "${SP_types[@]}"; do
            # Define the output and logging directories based on task, FT_type, and SP_type
            OUTPUT_DIR="./llama-3.2-fine-tuned-model/single/output_${task}_${FT_type}_${SP_type}_new"
            LOGGING_DIR="./logs/${task}_${FT_type}_${SP_type}_new"
            
            # Create the directories if they don't exist
            mkdir -p $OUTPUT_DIR
            mkdir -p $LOGGING_DIR
            
            # Update the config.yaml file with the current task, FT_type, SP_type, output_dir, and logging_dir
            echo "Running experiment for task: $task, FT_type: $FT_type, SP: $SP_type, Output: $OUTPUT_DIR, Logs: $LOGGING_DIR"
            
            # Update the config.yaml file
            sed -i "s/^FT_type: .*/FT_type: $FT_type/" config.yaml
            sed -i "s/^SP: .*/SP: $SP_type/" config.yaml
            sed -i "s|^output_dir: .*|output_dir: $OUTPUT_DIR|" config.yaml
            sed -i "s|^logging_dir: .*|logging_dir: $LOGGING_DIR|" config.yaml
            sed -i "s/^task: .*/task: \"$task\"/" config.yaml  # Update task in config.yaml
            
            # Run the Python script for fine-tuning
            python Finetune_single.py --config config.yaml

            # Kill all GPU-related processes to free up resources
            kill_gpu_processes

        done
    done
done
