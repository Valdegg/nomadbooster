"""
UN DESA City Population Data Integration

Data Source: UN Department of Economic and Social Affairs World Urbanization Prospects
URL: https://population.un.org/wup/Download/Files/WUP2018-F12-Cities_Over_300K.xls
Access Method: CSV/Excel download from UN DESA database
Update Frequency: Static (updated every 2-4 years)
Data Type: Static city properties

Metric: city_size (enum: small, medium, large, mega)
Description: City population size classification for urban lifestyle filtering.
Uses official UN population data to classify cities into standardized size categories
for filtering based on urban environment preferences.

Integration Status: ✅ READY - Straightforward population data classification
Implementation: 
1. Download UN DESA World Urbanization Prospects dataset
2. Extract population data for target cities
3. Apply classification thresholds to determine city size categories
4. Handle metropolitan area vs city proper population differences
5. Output standardized city size classifications

City Size Classification:
- Small: <500,000 people (intimate, walkable, local feel)
- Medium: 500,000 - 1,500,000 (balanced urban amenities + manageability)  
- Large: 1,500,000 - 5,000,000 (major city amenities, complexity)
- Mega: >5,000,000 people (megacity, maximum urban intensity)

Population Data Considerations:
- City proper vs metropolitan area definitions
- Administrative boundaries vs urban agglomeration
- Most recent available data (2020-2025 estimates)
- Consistency across cities in measurement approach

Output: ../data/sources/undesa_city_population.json
Schema: {"city": str, "country": str, "city_size": str, "population": int, "population_year": int, "metropolitan_area_population": int, "data_source": str, "last_updated": str}
"""

import requests
import pandas as pd
import json
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# City size classification thresholds
CITY_SIZE_THRESHOLDS = {
    "mega": 5000000,      # >5M people
    "large": 1500000,     # 1.5M - 5M people  
    "medium": 500000,     # 500K - 1.5M people
    "small": 0            # <500K people
}

def download_undesa_population_data() -> pd.DataFrame:
    """
    Download UN DESA World Urbanization Prospects dataset
    
    Returns:
        pd.DataFrame: UN DESA population data
    """
    # TODO: Implement UN DESA data download
    pass

def extract_city_population_data(undesa_data: pd.DataFrame, target_cities: List[str]) -> pd.DataFrame:
    """
    Extract population data for target cities from UN DESA dataset
    
    Args:
        undesa_data: Full UN DESA population dataset
        target_cities: List of cities to extract data for
        
    Returns:
        pd.DataFrame: Population data for target cities
    """
    # TODO: Implement city data extraction
    pass

def classify_city_size(population: int) -> str:
    """
    Classify city size based on population thresholds
    
    Args:
        population: City population
        
    Returns:
        str: City size classification (small/medium/large/mega)
    """
    # TODO: Implement city size classification logic
    pass

def handle_metropolitan_vs_city_proper(city_data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle differences between city proper and metropolitan area populations
    
    Args:
        city_data: Raw city population data
        
    Returns:
        pd.DataFrame: Standardized population data
    """
    # TODO: Implement population standardization logic
    pass

def validate_population_data(population_data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate population data for consistency and accuracy
    
    Args:
        population_data: Population data to validate
        
    Returns:
        pd.DataFrame: Validated population data
    """
    # TODO: Implement population data validation
    pass

def get_most_recent_population_estimates(population_data: pd.DataFrame) -> pd.DataFrame:
    """
    Get most recent population estimates for each city
    
    Args:
        population_data: Historical population data
        
    Returns:
        pd.DataFrame: Most recent population estimates
    """
    # TODO: Implement recent estimate extraction
    pass

def save_city_population_data(data: List[Dict], output_path: str = "../data/sources/undesa_city_population.json"):
    """
    Save city population data to JSON file
    
    Args:
        data: Processed population data
        output_path: Output file path
    """
    # TODO: Implement save logic
    pass

# Target cities for population classification
TARGET_CITIES = [
    "Berlin", "Amsterdam", "Barcelona", "Prague", "Lisbon",
    "Vienna", "Rome", "Paris", "Copenhagen", "Stockholm", 
    "Brussels", "Madrid", "Munich", "Zurich", "Dublin",
    "Budapest", "Warsaw", "Athens", "Helsinki", "Oslo"
]

if __name__ == "__main__":
    # TODO: Implement main execution logic
    pass 