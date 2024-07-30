#!/bin/bash

# Check if a class argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <class_name>"
  exit 1
fi

# Assign the class argument to a variable
CLASS=$1

# Define the directory containing the log files
LOG_DIR="."

# Find the latest log file
CLASS_FILE=$(ls -t $LOG_DIR/log_*.txt | head -n 1)
echo $CLASS_FILE

# Check if the class file exists
if [ ! -f "$CLASS_FILE" ]; then
  echo "No log files found in $LOG_DIR."
  exit 1
fi

# Define the base directory for the images
BASE_DIR="../data/IORS/png"

# Read the class file and extract images for the specified class
IMAGES=()
FOUND_CLASS=0
while IFS= read -r line; do
  if [ "$line" == "$CLASS" ]; then
    FOUND_CLASS=1
    continue
  fi

  if [ $FOUND_CLASS -eq 1 ]; then
    if [[ "$line" =~ ^class_ ]] || [[ "$line" =~ ^outliers ]]; then
      break
    fi
    # Check if line is not empty to avoid adding base directory to images
    if [ -n "$line" ]; then
      IMAGES+=("$BASE_DIR/$line")
    fi
  fi
done < "$CLASS_FILE"

# Check if any images were found for the specified class
if [ ${#IMAGES[@]} -eq 0 ]; then
  echo "No images found for class $CLASS."
  exit 1
fi

# Open all images in the specified class using eog
echo "${IMAGES[@]}" | xargs eog 
