#!/usr/bin/env python3
"""
Numbeo Data to CSV Converter

Reads the detailed Numbeo cost and safety JSON data and creates a clean CSV
with key travel-related metrics for easy analysis and integration.

Output: cities_static_properties_real.csv
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

def load_numbeo_data() -> tuple[Dict, Dict]:
    """
    Load both Numbeo JSON data files
    
    Returns:
        tuple: (cost_data, safety_data) dictionaries
    """
    cost_file = "../data/sources/numbeo_detailed_costs.json"
    safety_file = "../data/sources/numbeo_safety_score.json"
    
    # Load cost data
    try:
        with open(cost_file, 'r') as f:
            cost_data = json.load(f)
        print(f"✅ Loaded cost data: {cost_data['total_cities']} cities")
    except FileNotFoundError:
        print(f"❌ Cost data file not found: {cost_file}")
        cost_data = {"cities": []}
    
    # Load safety data
    try:
        with open(safety_file, 'r') as f:
            safety_data = json.load(f)
        print(f"✅ Loaded safety data: {len(safety_data['cities'])} cities")
    except FileNotFoundError:
        print(f"❌ Safety data file not found: {safety_file}")
        safety_data = {"cities": []}
    
    return cost_data, safety_data

def clean_currency(currency_str: str) -> str:
    """
    Clean and standardize currency codes
    
    Args:
        currency_str: Raw currency string (e.g., "€", "$", "£")
        
    Returns:
        str: Clean currency code (e.g., "EUR", "USD", "GBP")
    """
    currency_map = {
        '€': 'EUR',
        '$': 'USD', 
        '£': 'GBP',
        '¥': 'JPY',
        'kr': 'SEK',  # Swedish Krona
        'CHF': 'CHF'  # Swiss Franc
    }
    
    if currency_str in currency_map:
        return currency_map[currency_str]
    
    # Return as-is if not in map
    return currency_str if currency_str else 'EUR'  # Default to EUR

def extract_key_cost_metrics(city_costs: Dict) -> Dict:
    """
    Extract key travel-related cost metrics from detailed cost data
    
    Args:
        city_costs: City cost data from Numbeo
        
    Returns:
        Dict: Key cost metrics (prices as numbers, currencies as clean codes)
    """
    detailed_costs = city_costs.get('detailed_costs', {})
    
    # Key travel-related cost items
    key_items = {
        'meal_inexpensive': 'Meal, Inexpensive Restaurant',
        'meal_mid_range_2p': 'Meal for 2 People, Mid-range Restaurant, Three-course', 
        'domestic_beer': 'Domestic Beer (1 pint draught)',
        'cappuccino': 'Cappuccino (regular)',
        'transport_ticket': 'One-way Ticket (Local Transport)',
        'transport_monthly': 'Monthly Pass (Regular Price)',
        'taxi_start': 'Taxi Start (Normal Tariff)',
        'taxi_1mile': 'Taxi 1 mile (Normal Tariff)',
        'gasoline_gallon': 'Gasoline (1 gallon)',
        'utilities_basic': 'Basic (Electricity, Heating, Cooling, Water, Garbage) for 915 sq ft Apartment',
        'internet': 'Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)',
        'fitness_club': 'Fitness Club, Monthly Fee for 1 Adult',
        'cinema_ticket': 'Cinema, International Release, 1 Seat',
        'milk_gallon': 'Milk (regular), (1 gallon)',
        'bread_1lb': 'Loaf of Fresh White Bread (1 lb)',
        'eggs_12': 'Eggs (regular) (12)',
        'chicken_1lb': 'Chicken Fillets (1 lb)',
        'apartment_1br_center': 'Apartment (1 bedroom) in City Center',
        'apartment_1br_outside': 'Apartment (1 bedroom) Outside of Centre',
        'apartment_3br_center': 'Apartment (3 bedrooms) in City Center',
        'apartment_3br_outside': 'Apartment (3 bedrooms) Outside of Centre'
    }
    
    metrics = {}
    currencies_found = set()
    
    # Extract prices for key items (clean numerical values)
    for key, item_name in key_items.items():
        if item_name in detailed_costs:
            price = detailed_costs[item_name]['price']
            currency = clean_currency(detailed_costs[item_name]['currency'])
            
            # Store clean numerical price
            metrics[key] = price
            currencies_found.add(currency)
        else:
            metrics[key] = None
    
    # Add overall metrics
    metrics['cost_index'] = city_costs.get('cost_index')
    metrics['total_cost_items'] = city_costs.get('total_items', 0)
    
    # Add primary currency (most common currency for this city)
    if currencies_found:
        metrics['primary_currency'] = list(currencies_found)[0]  # Take first currency found
    else:
        metrics['primary_currency'] = 'EUR'  # Default
    
    return metrics

def combine_city_data(cost_data: Dict, safety_data: Dict) -> List[Dict]:
    """
    Combine cost and safety data for all cities
    
    Args:
        cost_data: Cost data from Numbeo
        safety_data: Safety data from Numbeo
        
    Returns:
        List[Dict]: Combined city data
    """
    # Create lookup for safety data
    safety_lookup = {}
    for city in safety_data.get('cities', []):
        safety_lookup[city['city']] = {
            'safety_score': city['safety_score'],
            'crime_index': city['crime_index'],
            'safety_index': city.get('safety_index')
        }
    
    combined_data = []
    
    # Process each city with cost data
    for city_costs in cost_data.get('cities', []):
        city_name = city_costs['city']
        
        # Start with basic city info
        city_row = {
            'city': city_name,
            'last_updated': datetime.now().isoformat()[:10]  # Just date
        }
        
        # Add cost metrics
        cost_metrics = extract_key_cost_metrics(city_costs)
        city_row.update(cost_metrics)
        
        # Add safety metrics
        if city_name in safety_lookup:
            city_row.update(safety_lookup[city_name])
        else:
            city_row.update({
                'safety_score': None,
                'crime_index': None,
                'safety_index': None
            })
        
        combined_data.append(city_row)
    
    return combined_data

def write_csv(data: List[Dict], output_file: str = "../data/cities_static_properties_real.csv"):
    """
    Write combined data to CSV file
    
    Args:
        data: Combined city data
        output_file: Output CSV file path
    """
    if not data:
        print("❌ No data to write")
        return
    
    # Define column order for better readability (all numerical, no currency symbols)
    column_order = [
        'city',
        'primary_currency',
        'cost_index',
        'safety_score',
        'crime_index',
        'safety_index',
        # Restaurant costs
        'meal_inexpensive',
        'meal_mid_range_2p', 
        'domestic_beer',
        'cappuccino',
        # Transportation costs
        'transport_ticket',
        'transport_monthly',
        'taxi_start',
        'taxi_1mile',
        'gasoline_gallon',
        # Housing costs
        'apartment_1br_center',
        'apartment_1br_outside',
        'apartment_3br_center',
        'apartment_3br_outside',
        # Utilities & Services
        'utilities_basic',
        'internet',
        'fitness_club',
        'cinema_ticket',
        # Groceries
        'milk_gallon',
        'bread_1lb',
        'eggs_12',
        'chicken_1lb',
        # Meta
        'total_cost_items',
        'last_updated'
    ]
    
    # Get all columns (in case some are missing from our predefined order)
    all_columns = set()
    for row in data:
        all_columns.update(row.keys())
    
    # Add any missing columns at the end
    final_columns = column_order + [col for col in sorted(all_columns) if col not in column_order]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_columns)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✅ CSV written: {output_file}")
    print(f"📊 {len(data)} cities, {len(final_columns)} columns")

def main():
    """
    Main execution function
    """
    print("🚀 Converting Numbeo JSON data to CSV...")
    print("=" * 50)
    
    # Load JSON data
    cost_data, safety_data = load_numbeo_data()
    
    # Combine the data
    combined_data = combine_city_data(cost_data, safety_data)
    print(f"📊 Combined data for {len(combined_data)} cities")
    
    # Write CSV
    write_csv(combined_data)
    
    # Show sample data
    if combined_data:
        print("\n📋 Sample data (first city):")
        sample_city = combined_data[0]
        for key, value in list(sample_city.items())[:10]:  # Show first 10 fields
            print(f"   {key}: {value}")
        if len(sample_city) > 10:
            print(f"   ... and {len(sample_city) - 10} more fields")
    
    print("\n🎉 CSV conversion complete!")

if __name__ == "__main__":
    main() 