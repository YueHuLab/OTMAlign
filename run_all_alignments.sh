#!/bin/bash

# This script runs the protein alignment for all pairs specified in id_pair.txt

# Ensure output directories exist
mkdir -p results/pdb_aligned
mkdir -p results/logs

# Check if the input file exists
if [ ! -f "id_pair.txt" ]; then
    echo "Error: id_pair.txt not found!"
    exit 1
fi

# Read the file line by line
while read -r mobile_id ref_id; do
    # Handle potential empty lines or lines with incorrect formatting
    if [ -z "$mobile_id" ] || [ -z "$ref_id" ]; then
        continue
    fi

    echo "Processing pair: $mobile_id vs $ref_id"

    # Define file paths
    mobile_pdb="RPIC_all/${mobile_id}.pdb"
    ref_pdb="RPIC_all/${ref_id}.pdb"
    output_pdb="results/pdb_aligned/${mobile_id}_aligned_to_${ref_id}.pdb"
    log_file="results/logs/${mobile_id}_vs_${ref_id}.log"

    # Check if input PDBs exist
    if [ ! -f "$mobile_pdb" ]; then
        echo "  Warning: Mobile PDB not found: $mobile_pdb. Skipping."
        echo "Warning: Mobile PDB not found: $mobile_pdb" > "$log_file"
        continue
    fi
    if [ ! -f "$ref_pdb" ]; then
        echo "  Warning: Reference PDB not found: $ref_pdb. Skipping."
        echo "Warning: Reference PDB not found: $ref_pdb" > "$log_file"
        continue
    fi

    # Run the alignment, redirecting all output (stdout and stderr) to the log file
    python3 OTMalign.py \
        --mobile "$mobile_pdb" \
        --reference "$ref_pdb" \
        --output "$output_pdb" > "$log_file" 2>&1
    
    echo "  Done. Log saved to $log_file"

done < id_pair.txt

echo -e "\nAll pairs processed. Results are in the 'results' directory."
