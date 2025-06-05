"""
IQAir World Air Quality → Pollution Index Data Integration

Data Source: IQAir World Air Quality Ranking / WHO PM₂.₅ data
URLs: 
- IQAir: https://www.iqair.com/world-air-quality-ranking
- WHO: https://www.who.int/data/gho/data/themes/air-pollution
Access Method: JSON API / CSV download
Update Frequency: Static (annual averages)
Data Type: Static city properties

Metric: pollution_index (int, 0-100 scale, lower = cleaner air)
Description: Air quality index derived from PM2.5 measurements and IQAir rankings.
Lower values indicate cleaner air, higher values indicate more polluted cities.
Based on annual average air quality measurements.

Integration Status: ✅ READY - Static metric, reliable air quality data
Implementation: Fetch IQAir city rankings JSON + WHO PM2.5 data, normalize to 0-100 scale,
merge data sources for comprehensive coverage, and output structured JSON.

Normalization Logic:
- PM2.5 <10 µg/m³ (WHO Good) → Index 0-20
- PM2.5 10-15 µg/m³ (Moderate) → Index 20-40  
- PM2.5 15-25 µg/m³ (Unhealthy for sensitive) → Index 40-60
- PM2.5 25-35 µg/m³ (Unhealthy) → Index 60-80
- PM2.5 >35 µg/m³ (Very Unhealthy) → Index 80-100

Output: ../data/sources/iqair_pollution_index.json
Schema: {"city": str, "country": str, "pollution_index": int, "pm25_ugm3": float, "iqair_rank": int, "last_updated": str}
"""

import requests
import json
from typing import List, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def fetch_iqair_ranking_data() -> List[Dict]:
    """
    Fetch IQAir world air quality ranking data
    
    Returns:
        List[Dict]: City air quality rankings from IQAir
    """
    # TODO: Implement IQAir API/JSON fetching logic
    pass

def fetch_who_pm25_data() -> pd.DataFrame:
    """
    Fetch WHO PM2.5 air pollution data
    
    Returns:
        pd.DataFrame: WHO PM2.5 measurements by city/country
    """
    # TODO: Implement WHO PM2.5 data fetching logic
    pass

def normalize_pm25_to_pollution_index(pm25_value: float) -> int:
    """
    Convert PM2.5 measurement to 0-100 pollution index
    
    Args:
        pm25_value: PM2.5 concentration in µg/m³
        
    Returns:
        int: Pollution index (0-100, lower = cleaner)
    """
    # TODO: Implement PM2.5 to index conversion logic
    pass

def merge_air_quality_sources(iqair_data: List[Dict], who_data: pd.DataFrame) -> List[Dict]:
    """
    Merge IQAir and WHO data sources for comprehensive coverage
    
    Args:
        iqair_data: IQAir city rankings
        who_data: WHO PM2.5 measurements
        
    Returns:
        List[Dict]: Merged air quality data
    """
    # TODO: Implement data merging logic
    pass

def save_pollution_index_data(data: List[Dict], output_path: str = "../data/sources/iqair_pollution_index.json"):
    """
    Save pollution index data to JSON file
    
    Args:
        data: Processed pollution index data
        output_path: Output file path
    """
    # TODO: Implement save logic
    pass

if __name__ == "__main__":
    # TODO: Implement main execution logic
    pass 