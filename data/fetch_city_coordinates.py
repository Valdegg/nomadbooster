#!/usr/bin/env python3
"""
Script to fetch coordinates for cities in european_iatas_df.csv
and create a new CSV with coordinates added.
"""

import pandas as pd
import sys
import os
import time
from typing import Optional, Tuple

# Add the parent directory to the Python path to import from datasource_integrations
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasource_integrations.meteostat_openmeteo_climate import get_city_coordinates
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coordinate_fetching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_coordinates_for_cities():
    """
    Read the European IATA cities CSV, fetch coordinates for each unique city,
    and create a new CSV with coordinates added.
    """
    
    # Read the existing CSV
    input_file = 'european_iatas_df.csv'
    output_file = 'european_cities_with_coordinates.csv'
    
    logger.info(f"Reading cities from {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Clean up column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        # Get unique cities to avoid duplicate API calls
        unique_cities = df['city'].unique()
        logger.info(f"Found {len(unique_cities)} unique cities")
        
        # Dictionary to store coordinates for each city
        city_coordinates = {}
        
        # Fetch coordinates for each unique city
        for i, city in enumerate(unique_cities, 1):
            logger.info(f"Processing {i}/{len(unique_cities)}: {city}")
            
            # Add small delay to be respectful to the API
            if i > 1:
                time.sleep(0.5)  # 500ms delay between requests
            
            coordinates = get_city_coordinates(city)
            
            if coordinates:
                lat, lon = coordinates
                city_coordinates[city] = {
                    'latitude': lat,
                    'longitude': lon
                }
                logger.info(f"✅ {city}: {lat}, {lon}")
            else:
                city_coordinates[city] = {
                    'latitude': None,
                    'longitude': None
                }
                logger.warning(f"❌ Could not find coordinates for {city}")
        
        # Add coordinates to the dataframe
        df['latitude'] = df['city'].map(lambda city: city_coordinates[city]['latitude'])
        df['longitude'] = df['city'].map(lambda city: city_coordinates[city]['longitude'])
        
        # Save the updated CSV
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Saved updated CSV to {output_file}")
        
        # Statistics
        total_cities = len(unique_cities)
        cities_with_coords = sum(1 for coords in city_coordinates.values() if coords['latitude'] is not None)
        cities_without_coords = total_cities - cities_with_coords
        
        logger.info(f"📊 Summary:")
        logger.info(f"   Total unique cities: {total_cities}")
        logger.info(f"   Cities with coordinates: {cities_with_coords}")
        logger.info(f"   Cities without coordinates: {cities_without_coords}")
        logger.info(f"   Success rate: {cities_with_coords/total_cities*100:.1f}%")
        
        # List cities without coordinates for manual review
        if cities_without_coords > 0:
            logger.info(f"❌ Cities without coordinates:")
            for city, coords in city_coordinates.items():
                if coords['latitude'] is None:
                    logger.info(f"   - {city}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error processing cities: {e}")
        return None

def create_city_coordinates_mapping():
    """
    Create a simplified CSV with just city names and coordinates
    for easy loading in the application.
    """
    
    # Read the file with coordinates
    input_file = 'european_cities_with_coordinates.csv'
    mapping_file = 'city_coordinates_mapping.csv'
    
    try:
        df = pd.read_csv(input_file)
        
        # Create a simplified mapping with unique cities only
        mapping_df = df[['city', 'latitude', 'longitude']].drop_duplicates(subset=['city'])
        
        # Remove cities without coordinates
        mapping_df = mapping_df.dropna(subset=['latitude', 'longitude'])
        
        # Save the mapping
        mapping_df.to_csv(mapping_file, index=False)
        logger.info(f"✅ Created city coordinates mapping: {mapping_file}")
        logger.info(f"   Contains {len(mapping_df)} cities with coordinates")
        
        return mapping_file
        
    except Exception as e:
        logger.error(f"Error creating city coordinates mapping: {e}")
        return None

if __name__ == "__main__":
    logger.info("🌍 Starting coordinate fetching for European cities")
    
    # Fetch coordinates for all cities
    coordinates_file = fetch_coordinates_for_cities()
    
    if coordinates_file:
        # Create simplified mapping for the application
        mapping_file = create_city_coordinates_mapping()
        
        if mapping_file:
            logger.info("🎉 Coordinate fetching completed successfully!")
            logger.info(f"📁 Files created:")
            logger.info(f"   - {coordinates_file} (full data with coordinates)")
            logger.info(f"   - {mapping_file} (simplified mapping for app)")
        else:
            logger.error("❌ Failed to create city coordinates mapping")
    else:
        logger.error("❌ Failed to fetch coordinates") 