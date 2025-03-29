#!/bin/bash

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Get the directory path from argument
dir_path="$1"

# Remove trailing slash if present
dir_path=${dir_path%/}

# Extract just the directory name from the path
dir_name=$(basename "$dir_path")

# Find all .txt files recursively in the specified directory and concatenate them
find "$dir_path" -type f -name "*.txt" -exec cat {} + > "${dir_name}.txt"

echo "All text files from $dir_path have been concatenated into ${dir_name}.txt"