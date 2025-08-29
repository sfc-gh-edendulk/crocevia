#!/usr/bin/env python3
"""
Process Carrefour stores text file and convert to CSV format.
Removes "Voir les catalogues" and organizes data into Store, Address, Open_Hours columns.
"""

import csv
import re
from pathlib import Path

def process_carrefour_stores(input_file: str, output_file: str):
    """
    Process the Carrefour stores text file and convert to CSV.
    
    Args:
        input_file: Path to the input text file
        output_file: Path to the output CSV file
    """
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove "Voir les catalogues" lines
    content = re.sub(r'^Voir les catalogues\s*$', '', content, flags=re.MULTILINE)
    
    # Split into lines and clean up
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    stores = []
    i = 0
    
    while i < len(lines):
        # Look for store name (starts with "Carrefour")
        if lines[i].startswith('Carrefour'):
            store_name = lines[i]
            
            # Next line should be address
            if i + 1 < len(lines) and not lines[i + 1].startswith('Carrefour'):
                address = lines[i + 1]
                
                # Next line should be opening hours (starts with "Ouvert")
                if i + 2 < len(lines) and lines[i + 2].startswith('Ouvert'):
                    open_hours = lines[i + 2]
                    
                    stores.append({
                        'Store': store_name,
                        'Address': address,
                        'Open_Hours': open_hours
                    })
                    
                    i += 3  # Move past this store entry
                else:
                    # Missing opening hours, skip this entry
                    i += 2
            else:
                # Missing address, skip this entry
                i += 1
        else:
            i += 1
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Store', 'Address', 'Open_Hours']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for store in stores:
            writer.writerow(store)
    
    print(f"Processed {len(stores)} Carrefour stores")
    print(f"Output written to: {output_file}")
    
    return len(stores)

if __name__ == "__main__":
    input_file = "data/carrefourstores2.txt"
    output_file = "data/carrefour_stores_france.csv"
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    store_count = process_carrefour_stores(input_file, output_file)
    
    print(f"\n✅ Successfully processed {store_count} Carrefour stores!")
    print(f"📁 CSV file saved as: {output_file}")
