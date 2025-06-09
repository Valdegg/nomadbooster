#!/usr/bin/env python3
"""
Script to integrate Airbnb accommodation costs into the main cities CSV file
"""

import json
import pandas as pd
from pathlib import Path

def load_accommodation_data():
    """Load and process accommodation cost data from JSON"""
    with open('airbnb_accommodation_costs.json', 'r') as f:
        data = json.load(f)
    
    accommodation_costs = {}
    
    # Process each accommodation entry
    for entry in data.get('accommodation_costs', []):
        city = entry.get('city')
        property_type = entry.get('property_type_filter')  # 'entire_place' or 'private_room'
        cost = entry.get('accommodation_cost_eur')
        
        if city and property_type and cost:
            if city not in accommodation_costs:
                accommodation_costs[city] = {}
            accommodation_costs[city][property_type] = cost
    
    return accommodation_costs

def integrate_accommodation_costs():
    """Integrate accommodation costs into main cities CSV"""
    
    # Load accommodation data
    print("Loading accommodation costs...")
    accommodation_costs = load_accommodation_data()
    
    # Load main cities CSV
    cities_csv_path = '../cities_static_properties_real.csv'
    print(f"Loading cities data from {cities_csv_path}...")
    df = pd.read_csv(cities_csv_path)
    
    # Add new columns for accommodation costs
    df['accommodation_entire_place_eur'] = None
    df['accommodation_private_room_eur'] = None
    df['accommodation_avg_eur'] = None
    
    # Integrate data
    matched_cities = 0
    for index, row in df.iterrows():
        city = row['city']
        
        if city in accommodation_costs:
            city_costs = accommodation_costs[city]
            
            # Add entire place cost
            if 'entire_place' in city_costs:
                df.at[index, 'accommodation_entire_place_eur'] = city_costs['entire_place']
            
            # Add private room cost  
            if 'private_room' in city_costs:
                df.at[index, 'accommodation_private_room_eur'] = city_costs['private_room']
            
            # Calculate average if both are available
            costs = []
            if 'entire_place' in city_costs:
                costs.append(city_costs['entire_place'])
            if 'private_room' in city_costs:
                costs.append(city_costs['private_room'])
            
            if costs:
                df.at[index, 'accommodation_avg_eur'] = round(sum(costs) / len(costs))
                matched_cities += 1
                print(f"✅ {city}: Entire place €{city_costs.get('entire_place', 'N/A')}, Private room €{city_costs.get('private_room', 'N/A')}")
            else:
                print(f"⚠️  {city}: No cost data found")
        else:
            print(f"❌ {city}: Not found in accommodation data")
    
    print(f"\n📊 Integration Summary:")
    print(f"   Total cities in CSV: {len(df)}")
    print(f"   Cities with accommodation data: {matched_cities}")
    print(f"   Cities missing data: {len(df) - matched_cities}")
    
    # Save updated CSV
    output_path = '../cities_static_properties_with_accommodation.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved updated CSV to: {output_path}")
    
    # Show sample of updated data
    print(f"\n📋 Sample of updated data:")
    cols_to_show = ['city', 'accommodation_entire_place_eur', 'accommodation_private_room_eur', 'accommodation_avg_eur']
    sample_df = df[cols_to_show].head(10)
    print(sample_df.to_string(index=False))
    
    return df

def show_accommodation_cost_ranges(df):
    """Show accommodation cost statistics"""
    print(f"\n📈 Accommodation Cost Statistics:")
    
    # Entire places
    entire_place_costs = df['accommodation_entire_place_eur'].dropna()
    if len(entire_place_costs) > 0:
        print(f"\n🏠 Entire Places (€/night):")
        print(f"   Min: €{entire_place_costs.min()}")
        print(f"   Max: €{entire_place_costs.max()}")
        print(f"   Median: €{entire_place_costs.median()}")
        print(f"   Cities: {len(entire_place_costs)}")
    
    # Private rooms
    private_room_costs = df['accommodation_private_room_eur'].dropna()
    if len(private_room_costs) > 0:
        print(f"\n🛏️  Private Rooms (€/night):")
        print(f"   Min: €{private_room_costs.min()}")
        print(f"   Max: €{private_room_costs.max()}")
        print(f"   Median: €{private_room_costs.median()}")
        print(f"   Cities: {len(private_room_costs)}")
    
    # Average costs
    avg_costs = df['accommodation_avg_eur'].dropna()
    if len(avg_costs) > 0:
        print(f"\n💰 Average Accommodation (€/night):")
        print(f"   Min: €{avg_costs.min()}")
        print(f"   Max: €{avg_costs.max()}")
        print(f"   Median: €{avg_costs.median()}")
        print(f"   Cities: {len(avg_costs)}")

if __name__ == "__main__":
    df = integrate_accommodation_costs()
    show_accommodation_cost_ranges(df) 