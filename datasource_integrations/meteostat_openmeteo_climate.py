"""
Open-Meteo Climate Data Integration

Data Source: Open-Meteo Weather Forecast API
URL: https://api.open-meteo.com/v1/forecast
Access Method: Free REST API (no API key required)
Update Frequency: Dynamic (hourly updates)
Data Type: Time-dependent city properties

Metrics: 
- Hard filters: max/min temperature, precipitation probability, UV index
- Soft scores: sunshine hours (0-100), comfort index (feels-like 18-26°C)
- Categorical: sunshine (bleak/mixed/bright), rain (arid/showery/wet), wind (calm/breezy/windy)

Integration Status: ✅ READY - Free API with comprehensive weather data
Implementation: Single API call per city returns all needed metrics for 7-14 day periods

Output: ../data/sources/openmeteo_climate.json
Schema: {"city": str, "travel_month": int, "avg_temp_c": int, "rainfall_mm": int, 
         "sunshine_category": str, "rain_category": str, "wind_category": str,
         "sunshine_score": int, "comfort_score": int, "uv_index_max": int, 
         "precipitation_probability_max": int, "last_updated": str}
"""

import requests
import pandas as pd
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)

def get_city_coordinates(city_name: str) -> Optional[tuple]:
    """
    Use Open-Meteo geocoding API to get precise coordinates for a European city
    
    Args:
        city_name: Name of the city
        
    Returns:
        Tuple of (latitude, longitude) or None if not found
    """
    try:
        # Use Open-Meteo geocoding API
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": city_name,
            "count": 10,  # Get multiple results to filter for Europe
            "language": "en",
            "format": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "results" in data and len(data["results"]) > 0:
            # Filter for European and adjacent regions (lat: 30-75, lon: -25-90)
            european_results = []
            for result in data["results"]:
                lat = result["latitude"]
                lon = result["longitude"]
                # European and adjacent regions bounds (including Iceland, Russia, Georgia, etc.)
                if 30 <= lat <= 75 and -25 <= lon <= 90:
                    # Add country info if available for logging
                    country = result.get("country", "Unknown")
                    admin1 = result.get("admin1", "")
                    european_results.append((lat, lon, country, admin1))
            
            if european_results:
                # Take the first European result
                lat, lon, country, admin1 = european_results[0]
                location_info = f"{country}" + (f", {admin1}" if admin1 else "")
                logger.info(f"✅ Found coordinates for {city_name}: {lat}, {lon} ({location_info})")
                return (lat, lon)
            else:
                # If no European results, log all results for debugging
                logger.warning(f"❌ No European/adjacent region coordinates found for {city_name}")
                for result in data["results"][:3]:  # Show first 3 results
                    lat = result["latitude"]
                    lon = result["longitude"]
                    country = result.get("country", "Unknown")
                    logger.warning(f"   Alternative: {lat}, {lon} ({country})")
                return None
        else:
            logger.warning(f"❌ No coordinates found for {city_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting coordinates for {city_name}: {e}")
        return None

def get_climate_normals_for_month(lat: float, lon: float, month: int) -> Dict:
    """
    Fetch climate normals from Open-Meteo Climate API for specific location and month
    
    Args:
        lat: Latitude
        lon: Longitude
        month: Month (1-12)
        
    Returns:
        Dict: Monthly climate averages
    """
    # Use 30-year historical period for climate normals (1991-2020)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "1991-01-01",
        "end_date": "2020-12-31",
        "models": "EC_Earth3P_HR",  # Use one of the available climate models
        "daily": ",".join([  # "daily", not "fields" for climate API
            "temperature_2m_mean",
            "temperature_2m_max", 
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "shortwave_radiation_sum"  # Add solar radiation to estimate sunshine
        ]),
        "temperature_unit": "celsius"
    }
    
    try:
        response = requests.get("https://climate-api.open-meteo.com/v1/climate", params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        if "daily" not in data:
            logger.error(f"No daily data returned for {lat}, {lon}")
            return None
            
        daily_data = data["daily"]
        
        # Extract data arrays
        times = daily_data["time"]
        temps = daily_data["temperature_2m_mean"]
        temp_maxs = daily_data["temperature_2m_max"]
        temp_mins = daily_data["temperature_2m_min"] 
        precipitations = daily_data["precipitation_sum"]
        wind_speeds = daily_data["wind_speed_10m_max"]
        solar_radiation = daily_data["shortwave_radiation_sum"]  # MJ/m² per day
        
        # Filter data for the specific month across all years
        month_indices = []
        for i, time_str in enumerate(times):
            date = datetime.fromisoformat(time_str)
            if date.month == month:
                month_indices.append(i)
        
        if not month_indices:
            logger.error(f"No data found for month {month}")
            return None
        
        # Calculate averages for the month across all years
        monthly_temp = sum(temps[i] for i in month_indices) / len(month_indices)
        monthly_temp_max = sum(temp_maxs[i] for i in month_indices) / len(month_indices)
        monthly_temp_min = sum(temp_mins[i] for i in month_indices) / len(month_indices)
        monthly_rainfall = sum(precipitations[i] for i in month_indices) / len(month_indices)
        monthly_wind = sum(wind_speeds[i] for i in month_indices) / len(month_indices)
        monthly_solar = sum(solar_radiation[i] for i in month_indices) / len(month_indices)
        
        # For weekly estimates (7 days), scale daily values appropriately
        weekly_rainfall = monthly_rainfall * 7  # 7 days worth
        
        # Convert solar radiation to sunshine hours with latitude and seasonal adjustments
        # Base conversion: ~0.3-0.5 hours per MJ/m², adjusted for location and season
        base_sunshine_per_mj = 0.4
        
        # Adjust for latitude (higher latitudes have longer summer days, shorter winter days)
        latitude_factor = 1.0 + (abs(lat) - 45) * 0.01  # Slight adjustment based on latitude
        
        # Apply seasonal adjustment using existing function
        seasonal_factor = get_seasonal_sunshine_adjustment(month)
        
        # Calculate weekly sunshine hours with adjustments
        daily_sunshine = monthly_solar * base_sunshine_per_mj * latitude_factor * seasonal_factor
        weekly_sunshine = daily_sunshine * 7
        
        # Ensure realistic bounds (0-84 hours max for 7 days)
        weekly_sunshine = max(0, min(84, weekly_sunshine))
        
        return {
            "avg_temp_c": round(monthly_temp),
            "temp_max_c": round(monthly_temp_max),
            "temp_min_c": round(monthly_temp_min),
            "rainfall_mm": round(weekly_rainfall),  # 7-day equivalent for filtering
            "sunshine_hours": round(weekly_sunshine, 1),
            "wind_speed_max_kmh": round(monthly_wind),
            # Estimate other metrics based on climate data
            "uv_index_max": estimate_uv_index(lat, month),
            "precipitation_probability_max": estimate_precip_probability(weekly_rainfall)
        }
        
    except Exception as e:
        logger.error(f"Error fetching climate data for {lat}, {lon}: {e}")
        return None

def get_seasonal_temp_adjustment(month: int) -> float:
    """Get temperature adjustment for month relative to annual average"""
    # European seasonal temperature pattern (approximate)
    adjustments = {
        1: -8,   # January: coldest
        2: -6,   # February
        3: -3,   # March
        4: 2,    # April
        5: 6,    # May
        6: 10,   # June
        7: 12,   # July: warmest
        8: 11,   # August
        9: 7,    # September
        10: 2,   # October
        11: -3,  # November
        12: -6   # December
    }
    return adjustments.get(month, 0)

def get_seasonal_rain_adjustment(month: int) -> float:
    """Get rainfall multiplier for month relative to annual average"""
    # European seasonal rainfall pattern (1.0 = average month)
    adjustments = {
        1: 0.9,   # January
        2: 0.8,   # February: drier
        3: 0.9,   # March
        4: 0.8,   # April
        5: 1.0,   # May
        6: 1.1,   # June
        7: 1.2,   # July: wetter
        8: 1.1,   # August
        9: 1.0,   # September
        10: 1.1,  # October
        11: 1.2,  # November: wetter
        12: 1.0   # December
    }
    return adjustments.get(month, 1.0)

def get_seasonal_sunshine_adjustment(month: int) -> float:
    """Get sunshine multiplier for month relative to annual average"""
    # European seasonal sunshine pattern (1.0 = average month)
    adjustments = {
        1: 0.4,   # January: least sunshine
        2: 0.6,   # February
        3: 0.8,   # March
        4: 1.2,   # April
        5: 1.4,   # May
        6: 1.6,   # June: most sunshine
        7: 1.6,   # July: most sunshine
        8: 1.4,   # August
        9: 1.2,   # September
        10: 0.8,  # October
        11: 0.5,  # November
        12: 0.4   # December: least sunshine
    }
    return adjustments.get(month, 1.0)

def estimate_uv_index(lat: float, month: int) -> int:
    """Estimate UV index based on latitude and month"""
    # Simple estimation based on latitude and season
    # Peak UV is around June/July, minimum in December/January
    seasonal_factor = max(0.3, abs(6.5 - month) / 6.5)  # 0.3 to 1.0
    latitude_factor = max(0.2, (90 - abs(lat)) / 90)  # Higher near equator
    
    base_uv = 10  # Maximum possible UV
    estimated_uv = base_uv * seasonal_factor * latitude_factor
    
    return max(1, min(11, round(estimated_uv)))

def estimate_precip_probability(weekly_rainfall: float) -> int:
    """Estimate precipitation probability based on rainfall amount"""
    # More rainfall = higher probability of precipitation
    if weekly_rainfall < 2:
        return 10  # Very low
    elif weekly_rainfall < 5:
        return 25  # Low
    elif weekly_rainfall < 15:
        return 50  # Moderate
    elif weekly_rainfall < 30:
        return 75  # High
    else:
        return 90  # Very high

def categorize_sunshine(sunshine_hours: float) -> str:
    """Categorize sunshine hours into bleak/mixed/bright"""
    if sunshine_hours > 42:  # >6h per day for 7 days
        return "bright"
    elif sunshine_hours > 21:  # >3h per day for 7 days
        return "mixed"
    else:
        return "bleak"

def categorize_rain(rainfall_mm: float) -> str:
    """Categorize rainfall into arid/showery/wet"""
    if rainfall_mm < 7.5:  # <30mm per month equivalent
        return "arid"
    elif rainfall_mm > 22.5:  # >90mm per month equivalent  
        return "wet"
    else:
        return "showery"

def categorize_wind(wind_speed_kmh: float) -> str:
    """Categorize wind speed into calm/breezy/windy"""
    if wind_speed_kmh < 15:
        return "calm"
    elif wind_speed_kmh > 35:
        return "windy"
    else:
        return "breezy"

def calculate_sunshine_score(sunshine_hours: float) -> int:
    """Convert sunshine hours to 0-100 score (based on max ~12h/day * 7 days = 84h)"""
    max_possible = 84  # 12 hours per day for 7 days
    return min(100, round((sunshine_hours / max_possible) * 100))

def calculate_comfort_score(temp_max: float, temp_min: float) -> int:
    """Calculate comfort score based on how well temps fit 18-26°C range"""
    # Check how much of the temp range falls within comfort zone
    comfort_min, comfort_max = 18, 26
    
    # Calculate overlap between actual range and comfort range
    actual_min, actual_max = temp_min, temp_max
    overlap_min = max(actual_min, comfort_min)
    overlap_max = min(actual_max, comfort_max)
    
    if overlap_max <= overlap_min:
        # No overlap with comfort zone
        # Calculate distance from comfort zone
        if actual_max < comfort_min:
            distance = comfort_min - actual_max
        else:
            distance = actual_min - comfort_max
        # Score decreases with distance from comfort zone
        return max(0, round(100 - (distance * 10)))
    else:
        # Some overlap - score based on how much of range is comfortable
        overlap_size = overlap_max - overlap_min
        total_range = actual_max - actual_min
        if total_range == 0:
            return 100 if comfort_min <= actual_min <= comfort_max else 0
        overlap_ratio = overlap_size / total_range
        return round(overlap_ratio * 100)

def fetch_monthly_climate_data(city_coords: Dict[str, tuple], months: List[int] = None) -> List[Dict]:
    """
    Fetch climate normals for multiple cities across different months
    
    Args:
        city_coords: Dictionary mapping city names to (lat, lon) coordinates
        months: List of months to fetch (1-12), defaults to all months
        
    Returns:
        List[Dict]: Climate data suitable for filtering
    """
    if months is None:
        months = list(range(1, 13))  # All months
    
    climate_data = []
    
    for city, (lat, lon) in city_coords.items():
        logger.info(f"Fetching climate normals for {city}")
        
        for month in months:
            climate_normals = get_climate_normals_for_month(lat, lon, month)
            
            if climate_normals:
                # Create the processed record
                record = {
                    "city": city,
                    "travel_month": month,
                    "avg_temp_c": climate_normals["avg_temp_c"],
                    "temp_max_c": climate_normals["temp_max_c"],
                    "temp_min_c": climate_normals["temp_min_c"],
                    "rainfall_mm": climate_normals["rainfall_mm"],
                    "sunshine_hours": climate_normals["sunshine_hours"],
                    "uv_index_max": climate_normals["uv_index_max"],
                    "wind_speed_max_kmh": climate_normals["wind_speed_max_kmh"],
                    "precipitation_probability_max": climate_normals["precipitation_probability_max"],
                    # Categorical classifications
                    "sunshine_category": categorize_sunshine(climate_normals["sunshine_hours"]),
                    "rain_category": categorize_rain(climate_normals["rainfall_mm"]),
                    "wind_category": categorize_wind(climate_normals["wind_speed_max_kmh"]),
                    # Soft scores
                    "sunshine_score": calculate_sunshine_score(climate_normals["sunshine_hours"]),
                    "comfort_score": calculate_comfort_score(climate_normals["temp_max_c"], climate_normals["temp_min_c"]),
                    "last_updated": datetime.now().isoformat()
                }
                
                climate_data.append(record)
                logger.info(f"✅ {city} month {month}: {record['avg_temp_c']}°C, {record['rainfall_mm']}mm, {record['sunshine_category']}")
            else:
                logger.warning(f"❌ Failed to fetch climate normals for {city} month {month}")
            
            # Rate limiting - respect API limits (climate API is less restrictive)
            time.sleep(0.2)  # 200ms between requests
    
    return climate_data

def save_climate_data(data: List[Dict], output_path: str = "../data/sources/openmeteo_climate.json"):
    """
    Save climate data to JSON file
    
    Args:
        data: Processed climate data
        output_path: Output file path
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Saved {len(data)} climate records to {output_path}")
        
        # Print summary statistics
        if data:
            df = pd.DataFrame(data)
            logger.info(f"📊 Climate data summary:")
            logger.info(f"   Cities: {df['city'].nunique()}")
            logger.info(f"   Months: {sorted(df['travel_month'].unique())}")
            logger.info(f"   Temp range: {df['avg_temp_c'].min()}°C to {df['avg_temp_c'].max()}°C") 
            logger.info(f"   Rainfall range: {df['rainfall_mm'].min()}mm to {df['rainfall_mm'].max()}mm")
            logger.info(f"   Sunshine categories: {df['sunshine_category'].value_counts().to_dict()}")
            logger.info(f"   Rain categories: {df['rain_category'].value_counts().to_dict()}")
            
    except Exception as e:
        logger.error(f"Error saving climate data: {e}")

# European cities for climate data collection  
EUROPEAN_CITIES = [
    "Berlin", "Amsterdam", "Barcelona", "Prague", "Lisbon", "Vienna", 
    "Rome", "Paris", "Copenhagen", "Stockholm", "Brussels", "Madrid", 
    "Munich", "Zurich", "Dublin", "Budapest", "Warsaw", "Athens", 
    "Helsinki", "Oslo"
]

import pandas as pd 

iata_data = pd.read_csv('../data/european_iatas_df.csv')
EUROPEAN_CITIES = set(iata_data['city'].tolist())


def get_all_city_coordinates(city_names: List[str]) -> Dict[str, tuple]:
    """
    Get coordinates for all cities using geocoding API
    
    Args:
        city_names: List of city names
        
    Returns:
        Dictionary mapping city names to (lat, lon) coordinates
    """
    coordinates = {}
    
    for city in city_names:
        coords = get_city_coordinates(city)
        if coords:
            coordinates[city] = coords
            # Rate limiting for geocoding API
            time.sleep(0.1)  # 100ms between requests
        else:
            logger.error(f"❌ Could not get coordinates for {city}")
    
    return coordinates

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("🌤️  Starting Open-Meteo climate normals collection...")
    
    # First, get precise coordinates for all cities using geocoding
    logger.info("🗺️  Getting precise coordinates for all cities...")
    city_coordinates = get_all_city_coordinates(EUROPEAN_CITIES)
    
    if not city_coordinates:
        logger.error("❌ No city coordinates could be obtained!")
        exit(1)
    
    logger.info(f"✅ Successfully got coordinates for {len(city_coordinates)} cities")
    
    # Fetch climate normals for all months (since we're using historical averages)
    all_months = list(range(1, 13))  # January through December
    
    climate_data = fetch_monthly_climate_data(city_coordinates, all_months)
    
    if climate_data:
        save_climate_data(climate_data)
        logger.info("🎯 Climate normals collection completed successfully!")
    else:
        logger.error("❌ No climate data collected!") 