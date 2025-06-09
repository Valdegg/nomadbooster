#!/usr/bin/env python3
"""
Script to fix country fields in airbnb_accommodation_costs.json
"""

import json

# City to country mapping
CITY_TO_COUNTRY = {
    "Berlin": "Germany",
    "Munich": "Germany", 
    "Amsterdam": "Netherlands",
    "Barcelona": "Spain",
    "Madrid": "Spain",
    "Prague": "Czech Republic",
    "Lisbon": "Portugal",
    "Vienna": "Austria",
    "Rome": "Italy",
    "Paris": "France",
    "Copenhagen": "Denmark", 
    "Stockholm": "Sweden",
    "Brussels": "Belgium",
    "Zurich": "Switzerland",
    "Dublin": "Ireland",
    "Budapest": "Hungary",
    "Warsaw": "Poland",
    "Athens": "Greece",
    "Helsinki": "Finland",
    "Oslo": "Norway"
}

def fix_countries():
    """Fix all country fields in the JSON file"""
    file_path = "airbnb_accommodation_costs.json"
    
    # Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Update all accommodation cost entries
    updated_count = 0
    for entry in data.get('accommodation_costs', []):
        city = entry.get('city')
        if city in CITY_TO_COUNTRY:
            if entry.get('country') != CITY_TO_COUNTRY[city]:
                entry['country'] = CITY_TO_COUNTRY[city]
                updated_count += 1
                print(f"Updated {city} -> {CITY_TO_COUNTRY[city]}")
    
    # Update sample data if it exists
    for entry in data.get('sample_data', []):
        city = entry.get('city')
        if city in CITY_TO_COUNTRY:
            if entry.get('country') != CITY_TO_COUNTRY[city]:
                entry['country'] = CITY_TO_COUNTRY[city]
                updated_count += 1
                print(f"Updated sample {city} -> {CITY_TO_COUNTRY[city]}")
    
    # Save the updated JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Fixed {updated_count} entries!")

if __name__ == "__main__":
    fix_countries() 