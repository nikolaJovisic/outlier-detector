#!/bin/bash

# Source and target directories
SOURCE_DIR="/mnt/lun2/vanja/iors-2020"
TARGET_DIR="/home/jovisic/projects/outlier-detector/data"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Function to create symlinks for files in a directory
create_symlinks() {
  SRC_DIR=$1
  DEST_DIR=$2

  for FILE in "$SRC_DIR"/*; do
    if [ -f "$FILE" ]; then
      FILENAME=$(basename "$FILE")
      ln -s "$FILE" "$DEST_DIR/$FILENAME"
    fi
  done
}

# Loop through all subdirectories in the source directory
for SUBDIR in "$SOURCE_DIR"/*/; do
  create_symlinks "$SUBDIR" "$TARGET_DIR"
done

echo "Symlinks created successfully."

