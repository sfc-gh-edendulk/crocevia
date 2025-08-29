#!/usr/bin/env python3
"""
Process Decathlon stores text file and convert to CSV format.
Organizes data into Store, Address columns.
"""

import csv
import re
from pathlib import Path

def process_decathlon_stores(input_file: str, output_file: str):
    """
    Process the Decathlon stores text file and convert to CSV.
    
    Args:
        input_file: Path to the input text file
        output_file: Path to the output CSV file
    """
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines, keep empty lines for parsing structure
    lines = [line.strip() for line in content.split('\n')]
    
    print(f"Total lines in file: {len(lines)}")
    print(f"First 10 lines: {lines[:10]}")
    
    stores = []
    i = 0
    
    while i < len(lines):
        # Skip empty lines
        if not lines[i]:
            i += 1
            continue
            
        # Store name (first non-empty line)
        store_name = lines[i]
        i += 1
        
        # Skip empty lines
        while i < len(lines) and not lines[i]:
            i += 1
            
        # Street address (next non-empty line)
        if i < len(lines) and lines[i]:
            street_address = lines[i]
            i += 1
            
            # Skip empty lines
            while i < len(lines) and not lines[i]:
                i += 1
                
            # City with postal code (next non-empty line)
            if i < len(lines) and lines[i]:
                city_postal = lines[i]
                i += 1
                
                # Combine street address and city for full address
                full_address = f"{street_address}, {city_postal}"
                
                stores.append({
                    'Store': store_name,
                    'Address': full_address
                })
            else:
                # Missing city, skip this store
                continue
        else:
            # Missing street address, skip this store
            continue
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Store', 'Address']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for store in stores:
            writer.writerow(store)
    
    print(f"Processed {len(stores)} Decathlon stores")
    print(f"Output written to: {output_file}")
    
    return len(stores)

if __name__ == "__main__":
    input_file = "data/decathlonstores.txt"
    output_file = "data/decathlon_stores_france.csv"
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    store_count = process_decathlon_stores(input_file, output_file)
    
    print(f"\n✅ Successfully processed {store_count} Decathlon stores!")
    print(f"📁 CSV file saved as: {output_file}")
