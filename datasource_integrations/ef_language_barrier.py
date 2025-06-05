"""
EF English Proficiency → Language Barrier Data Integration

Data Source: EF English Proficiency Index (EPI)
URL: https://www.ef.com/wwen/epi/
Access Method: CSV download from EF EPI report
Update Frequency: Static (annual updates)
Data Type: Static city properties

Metric: language_barrier (LanguageBarrier enum, 1-5 scale)
Description: Language barrier level derived from EF English Proficiency Index rankings.
Converts EPI scores/ranks to standardized 1-5 barrier scale where:
1 = English native/widely spoken, 5 = Local language required

Integration Status: ✅ READY - Static metric, rank to enum conversion
Implementation: Download EF EPI CSV data, extract country rankings, convert ranks to 
barrier levels using threshold mapping, map to cities, and output structured JSON.

Mapping Logic:
- EPI Score 600+ (Very High) → Barrier 1 (English native)
- EPI Score 550-599 (High) → Barrier 2 (Minimal barrier)  
- EPI Score 500-549 (Moderate) → Barrier 3 (Moderate barrier)
- EPI Score 450-499 (Low) → Barrier 4 (Significant barrier)
- EPI Score <450 (Very Low) → Barrier 5 (High barrier)

Output: ../data/sources/ef_language_barrier.json
Schema: {"city": str, "country": str, "language_barrier": int, "epi_score": float, "epi_rank": int, "last_updated": str}
"""

import requests
import pandas as pd
import json
from typing import List, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LanguageBarrier(int, Enum):
    ENGLISH_NATIVE = 1        # English is widely spoken
    MINIMAL_BARRIER = 2       # Minimal language barrier
    MODERATE_BARRIER = 3      # Some English, basic communication possible
    SIGNIFICANT_BARRIER = 4   # Limited English, local language helpful
    HIGH_BARRIER = 5          # Local language required

def fetch_ef_epi_data() -> pd.DataFrame:
    """
    Fetch EF English Proficiency Index data
    
    Returns:
        pd.DataFrame: EF EPI scores and rankings by country
    """
    # TODO: Implement EF EPI data fetching logic
    pass

def convert_epi_to_barrier_level(epi_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert EPI scores to language barrier levels
    
    Args:
        epi_data: EF EPI scores by country
        
    Returns:
        pd.DataFrame: Language barrier levels by country
    """
    # TODO: Implement EPI score to barrier level conversion logic
    pass

def map_barriers_to_cities(barrier_data: pd.DataFrame) -> List[Dict]:
    """
    Map country-level language barriers to cities
    
    Args:
        barrier_data: Language barrier levels by country
        
    Returns:
        List[Dict]: Language barriers mapped to cities
    """
    # TODO: Implement city mapping logic
    pass

def save_language_barrier_data(data: List[Dict], output_path: str = "../data/sources/ef_language_barrier.json"):
    """
    Save language barrier data to JSON file
    
    Args:
        data: Processed language barrier data
        output_path: Output file path
    """
    # TODO: Implement save logic
    pass

if __name__ == "__main__":
    # TODO: Implement main execution logic
    pass 