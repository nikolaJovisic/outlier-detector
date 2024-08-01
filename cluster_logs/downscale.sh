#!/bin/bash

# Usage: ./downscale_images.sh /path/to/src /path/to/dst 800

# Check for the correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 /path/to/src /path/to/dst width"
    exit 1
fi

SRC_DIR=$1
DST_DIR=$2
WIDTH=$3

# Check if source directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Source directory $SRC_DIR does not exist."
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$DST_DIR" ]; then
    mkdir -p "$DST_DIR"
fi

# Process each image in the source directory
for img in "$SRC_DIR"/*.{jpg,jpeg,png}; do
    if [ -f "$img" ]; then
        filename=$(basename "$img")
        convert "$img" -resize "${WIDTH}x" "$DST_DIR/$filename"
        echo "Processed $img -> $DST_DIR/$filename"
    fi
done

echo "Downscaling completed."
