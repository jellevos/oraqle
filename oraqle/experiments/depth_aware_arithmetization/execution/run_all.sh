#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script's directory
cd "$SCRIPT_DIR"

# Loop through all Python files in the script's directory
for file in *.py
do
  # Check if there are any Python files
  if [ -e "$file" ]; then
    echo "Running $file"
    python3 "$file"
  else
    echo "No Python files found in the script's directory."
    break
  fi
done
