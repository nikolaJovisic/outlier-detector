#!/bin/bash

# Output directory
output_dir="pngs"

# Create output directory if it does not exist
mkdir -p "$output_dir"

# Loop through each file path passed as arguments
for dicom_file in "$@"; do
    echo $dicom_file
    # Extract the base filename (without path and extension)
    base_name=$(basename "$dicom_file" )
    
    # Define the output PNG file path
    output_file="$output_dir/$base_name.png"
    
    # Convert DICOM to PNG using dcm2pnm
    dcm2pnm "$dicom_file" "$output_file"
done

