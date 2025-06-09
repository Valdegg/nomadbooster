"""
Numbeo Crime Index → Safety Score Data Integration

Data Source: Numbeo Crime Index
URL: https://www.numbeo.com/crime/rankings.jsp
Access Method: BrightData Browser API (HTML scraping via browser automation)
Update Frequency: Static (periodic updates)
Data Type: Static city properties

Metric: safety_score (int, 0-100 scale, higher = safer)
Description: Safety score derived from Numbeo's Crime Index. Crime Index is inverted and 
normalized to create a safety score where 100 = very safe, 0 = very dangerous.

Integration Status: ✅ READY - BrightData browser automation implementation
Implementation: Use BrightData Browser API to load Numbeo crime page with JS execution,
scrape rankings table as markdown, parse crime indices, convert to safety scores.

BrightData Value: Numbeo loads data via JS and has rate limits. BrightData's unlocker
proxies + Browser API execute JS and dodge IP blocks.

Output: ../data/sources/numbeo_safety_score.json
Schema: {"city": str, "country": str, "safety_score": int, "crime_index": float, "numbeo_rank": int, "last_updated": str}
"""

import json
import re
import os
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import logging
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

# BrightData configuration
AUTH = os.getenv("BRIGHTDATA_AUTH")  # Format: "brd-customer-123456-zone-my_scraper_zone:password"
BR_ENDPOINT = os.getenv("BRIGHTDATA_ENDPOINT")

# Try to extract auth from endpoint if not provided directly
if not AUTH and BR_ENDPOINT and "@brd.superproxy.io" in BR_ENDPOINT:
    try:
        auth_part = BR_ENDPOINT.split("@")[0].replace("wss://", "")
        if auth_part.startswith("brd-customer-"):
            AUTH = auth_part
            logger.info(f"Extracted auth from endpoint: {AUTH[:20]}...")
    except Exception as e:
        logger.warning(f"Could not extract auth from endpoint: {e}")

# If no endpoint provided, generate from auth
if not BR_ENDPOINT and AUTH:
    BR_ENDPOINT = f"wss://{AUTH}@brd.superproxy.io:9222"

# Target cities for our travel recommendation system
TARGET_CITIES = [
    "Berlin", "Amsterdam", "Barcelona", "Prague", "Lisbon",
    "Vienna", "Rome", "Paris", "Copenhagen", "Stockholm", 
    "Brussels", "Madrid", "Munich", "Zurich", "Dublin",
    "Budapest", "Warsaw", "Athens", "Helsinki", "Oslo"
]

import pandas as pd 

iata_data = pd.read_csv('../data/european_iatas_df.csv')
TARGET_CITIES = set(iata_data['city'].tolist())

async def fetch_safety_data_for_city(city: str) -> Dict:
    """
    Fetch safety/crime data for a specific city using BrightData Browser API
    
    Args:
        city: City name to fetch data for
        
    Returns:
        Dict: City safety data {"city": str, "crime_index": float, "safety_index": float}
    """
    if not BR_ENDPOINT:
        raise ValueError("BrightData configuration missing. Set BRIGHTDATA_AUTH or BRIGHTDATA_ENDPOINT")
    
    if "brd.superproxy.io" not in BR_ENDPOINT:
        raise ValueError("Invalid BrightData endpoint format")
    
    target_url = f"https://www.numbeo.com/crime/in/{city}"
    
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(BR_ENDPOINT)
            try:
                page = await browser.new_page()
                await page.goto(target_url, timeout=120_000)
                
                logger.info(f"Page loaded for {city}, waiting for crime data table...")
                
                # Save page screenshot for debugging (optional)
                if os.getenv("DEBUG_NUMBEO"):
                    await page.screenshot(path=f"debug_safety_{city.lower()}.png")
                    logger.info(f"Saved debug screenshot: debug_safety_{city.lower()}.png")
                
                # Wait for the crime data table
                table_found = False
                selectors_to_try = [
                    "table.data_wide_table",
                    "table.tablesorter",
                    "table:has(td:contains('Crime Index'))",
                    "table:has(td:contains('Safety Index'))"
                ]
                
                for selector in selectors_to_try:
                    try:
                        await page.wait_for_selector(selector, timeout=10_000)
                        logger.info(f"Found table with selector: {selector}")
                        table_found = True
                        break
                    except Exception:
                        logger.warning(f"Selector '{selector}' not found, trying next...")
                        continue
                
                if not table_found:
                    logger.error("Could not find any crime data table on the page")
                    # Log page content for debugging
                    content = await page.content()
                    logger.info(f"Page content preview: {content[:1000]}...")
                    raise ValueError(f"No crime data table found on Numbeo page for {city}")
                
                # Extract crime and safety indices
                crime_safety_data = await page.evaluate("""
                    () => {
                        const tables = document.querySelectorAll('table.data_wide_table, table.tablesorter, table');
                        let crimeIndex = null;
                        let safetyIndex = null;
                        
                        for (const table of tables) {
                            const rows = [...table.querySelectorAll('tr')];
                            
                            for (const row of rows) {
                                const cells = [...row.children];
                                if (cells.length < 2) continue;
                                
                                const labelText = cells[0].textContent.trim();
                                const valueText = cells[1].textContent.trim();
                                
                                // Look for Crime Index
                                if (labelText.includes('Crime Index') && !labelText.includes('in ')) {
                                    const match = valueText.match(/([0-9.]+)/);
                                    if (match) {
                                        crimeIndex = parseFloat(match[1]);
                                    }
                                }
                                
                                // Look for Safety Index
                                if (labelText.includes('Safety Index') && !labelText.includes('in ')) {
                                    const match = valueText.match(/([0-9.]+)/);
                                    if (match) {
                                        safetyIndex = parseFloat(match[1]);
                                    }
                                }
                            }
                        }
                        
                        return {
                            crimeIndex: crimeIndex,
                            safetyIndex: safetyIndex,
                            found: crimeIndex !== null || safetyIndex !== null
                        };
                    }
                """)
                
                logger.info(f"Crime/Safety extraction result: {crime_safety_data}")
                
                if not crime_safety_data or not crime_safety_data.get('found'):
                    logger.error(f"Could not find crime or safety index for {city}")
                    raise ValueError(f"Could not find crime/safety data for {city}")
                
                crime_index = crime_safety_data.get('crimeIndex')
                safety_index = crime_safety_data.get('safetyIndex')
                
                # If we have crime index but no safety index, calculate it
                if crime_index is not None and safety_index is None:
                    safety_index = 100 - crime_index
                
                # If we have safety index but no crime index, calculate it
                if safety_index is not None and crime_index is None:
                    crime_index = 100 - safety_index
                
                logger.info(f"Fetched safety data for {city}: Crime={crime_index}, Safety={safety_index}")
                
                return {
                    "city": city,
                    "crime_index": crime_index,
                    "safety_index": safety_index,
                    "last_updated": datetime.now().isoformat()
                }
                
            finally:
                await browser.close()
                
    except Exception as e:
        logger.error(f"Error fetching safety data for {city}: {e}")
        raise

def load_existing_safety_data(output_path: str = "../data/sources/numbeo_safety_score.json") -> Dict:
    """
    Load existing safety data file if it exists
    
    Returns:
        Dict: Existing data structure or empty structure
    """
    try:
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load existing safety data: {e}")
    
    # Return empty structure if file doesn't exist or can't be loaded
    return {
        "data_source": "Numbeo Crime Index → Safety Score via BrightData",
        "url": "https://www.numbeo.com/crime/rankings.jsp",
        "last_updated": datetime.now().isoformat(),
        "description": "Safety score derived from crime index, 100 = very safe, 0 = dangerous",
        "calculation": "safety_score = 100 - crime_index (or use Numbeo safety index if available)",
        "cities": []
    }

def is_safety_data_recent(city_data: Dict, max_age_hours: int = 24) -> bool:
    """
    Check if safety data is recent enough to skip re-fetching
    
    Args:
        city_data: City data dict with 'last_updated' field
        max_age_hours: Maximum age in hours before data is considered stale
        
    Returns:
        bool: True if data is recent enough
    """
    try:
        last_updated = datetime.fromisoformat(city_data['last_updated'].replace('Z', '+00:00'))
        age_hours = (datetime.now() - last_updated).total_seconds() / 3600
        return age_hours < max_age_hours
    except Exception:
        return False

def save_single_safety_data(city_data: Dict, output_path: str = "../data/sources/numbeo_safety_score.json"):
    """
    Save or update safety data for a single city incrementally
    
    Args:
        city_data: Safety data for single city
        output_path: Output file path
    """
    try:
        # Load existing data
        existing_data = load_existing_safety_data(output_path)
        
        # Find and update/add the city data
        city_name = city_data['city']
        city_found = False
        
        for i, existing_city in enumerate(existing_data['cities']):
            if existing_city['city'] == city_name:
                existing_data['cities'][i] = city_data
                city_found = True
                break
        
        if not city_found:
            existing_data['cities'].append(city_data)
        
        # Update metadata
        existing_data['last_updated'] = datetime.now().isoformat()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save updated data
        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        logger.info(f"Saved/updated safety data for {city_name} (total: {len(existing_data['cities'])} cities)")
        
    except Exception as e:
        logger.error(f"Error saving safety data for {city_data.get('city', '?')}: {e}")
        raise

async def fetch_all_city_safety_data(force_refresh: bool = False) -> List[Dict]:
    """
    Fetch safety data for all target cities with incremental saving
    
    Args:
        force_refresh: If True, ignore existing data age and refresh all cities
    
    Returns:
        List[Dict]: Safety data for all cities (loaded from saved file)
    """
    output_path = "../data/sources/numbeo_safety_score.json"
    
    # Load existing data to check what we already have
    existing_data = load_existing_safety_data(output_path)
    existing_cities = {city['city']: city for city in existing_data['cities']}
    
    cities_to_fetch = []
    cities_skipped = []
    
    # Determine which cities need to be fetched
    for city in TARGET_CITIES:
        if not force_refresh and city in existing_cities and is_safety_data_recent(existing_cities[city]):
            cities_skipped.append(city)
            logger.info(f"Skipping {city} - safety data is recent")
        else:
            cities_to_fetch.append(city)
    
    if force_refresh and cities_skipped:
        logger.info("Force refresh mode - will fetch all cities regardless of age")
        cities_to_fetch = TARGET_CITIES.copy()
        cities_skipped = []
    
    logger.info(f"Will fetch {len(cities_to_fetch)} cities, skipping {len(cities_skipped)} with recent data")
    
    # Fetch missing/stale cities one by one with incremental saving
    for city in cities_to_fetch:
        try:
            logger.info(f"Fetching safety data for {city}...")
            city_data = await fetch_safety_data_for_city(city)
            
            # Convert to safety score immediately
            safety_city = convert_crime_to_safety_score([city_data])[0] if city_data else None
            
            if safety_city:
                # Save immediately
                save_single_safety_data(safety_city, output_path)
            
            # Add small delay to be respectful
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Failed to fetch safety data for {city}: {e}")
            continue
    
    # Return all data (existing + newly fetched)
    final_data = load_existing_safety_data(output_path)
    return final_data['cities']

def parse_crime_data_from_markdown(markdown_content: str) -> List[Dict]:
    """
    Parse crime index data from markdown table content
    
    Args:
        markdown_content: Markdown content from BrightData scraping
        
    Returns:
        List[Dict]: Parsed city crime data
    """
    crime_data = []
    
    try:
        # Extract table rows from markdown
        lines = markdown_content.strip().split('\n')
        
        for line in lines:
            # Look for table rows with city data
            if '|' in line and any(city.lower() in line.lower() for city in TARGET_CITIES):
                parts = [part.strip() for part in line.split('|')]
                
                if len(parts) >= 6:  # Rank, City, Country, Crime Index, Safety Index, plus empty cells
                    try:
                        rank = int(parts[1]) if parts[1].isdigit() else None
                        city = parts[2]
                        country = parts[3]
                        crime_index_str = parts[4]
                        safety_index_str = parts[5] if len(parts) > 5 else None
                        
                        # Extract numeric crime index
                        crime_index_match = re.search(r'(\d+\.?\d*)', crime_index_str)
                        if crime_index_match:
                            crime_index = float(crime_index_match.group(1))
                            
                            # Extract safety index if available, otherwise calculate
                            safety_index = None
                            if safety_index_str:
                                safety_match = re.search(r'(\d+\.?\d*)', safety_index_str)
                                if safety_match:
                                    safety_index = float(safety_match.group(1))
                            
                            # Only include our target cities
                            if any(target_city.lower() in city.lower() for target_city in TARGET_CITIES):
                                crime_data.append({
                                    'city': city,
                                    'country': country,
                                    'crime_index': crime_index,
                                    'safety_index': safety_index,
                                    'numbeo_rank': rank,
                                    'last_updated': datetime.now().isoformat()
                                })
                                
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse line: {line} - {e}")
                        continue
                        
    except Exception as e:
        logger.error(f"Error parsing markdown content: {e}")
        raise
        
    return crime_data

def convert_crime_to_safety_score(raw_data: List[Dict]) -> List[Dict]:
    """
    Convert crime index to safety score (invert and normalize to 0-100 scale)
    
    Args:
        raw_data: Raw crime/safety data from Numbeo
        
    Returns:
        List[Dict]: Safety score data
    """
    safety_data = []
    
    for city_data in raw_data:
        try:
            crime_index = city_data.get('crime_index')
            safety_index = city_data.get('safety_index')
            
            # Use safety index if available, otherwise calculate from crime index
            if safety_index is not None:
                safety_score = safety_index
            elif crime_index is not None:
                safety_score = max(0, min(100, 100 - crime_index))
            else:
                logger.warning(f"No crime or safety data for {city_data['city']}")
                continue
            
            # Validate range
            if safety_score < 0 or safety_score > 100:
                logger.warning(f"Unusual safety score for {city_data['city']}: {safety_score}")
                safety_score = max(0, min(100, safety_score))
            
            safety_data.append({
                'city': city_data['city'],
                'safety_score': int(round(safety_score)),
                'crime_index': crime_index,
                'safety_index': safety_index,
                'last_updated': city_data['last_updated']
            })
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error converting crime to safety for {city_data}: {e}")
            continue
            
    return safety_data

def save_safety_score_data(data: List[Dict], output_path: str = "../data/sources/numbeo_safety_score.json"):
    """
    Save safety score data to JSON file
    
    Args:
        data: Processed safety score data
        output_path: Output file path
    """
    try:
        # Load existing JSON structure
        try:
            with open(output_path, 'r') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            # Create basic structure if file doesn't exist
            json_data = {
                "data_source": "Numbeo Crime Index → Safety Score via BrightData",
                "url": "https://www.numbeo.com/crime/rankings.jsp",
                "last_updated": datetime.now().isoformat(),
                "description": "Safety score derived from crime index, 100 = very safe, 0 = dangerous",
                "calculation": "safety_score = 100 - crime_index (or use Numbeo safety index if available)",
                "cities": []
            }
        
        # Update cities data and timestamp
        json_data["cities"] = data
        json_data["last_updated"] = datetime.now().isoformat()
        
        # Save updated data
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        logger.info(f"Saved {len(data)} cities to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        raise

# MCP Tool wrapper for single city fetching
async def tool_fetch_safety_data(args: dict) -> dict:
    """
    MCP tool signature for fetching safety data for a specific city
    
    Args:
        args: {"city": "Lisbon"}
        
    Returns:
        dict: {"city": "Lisbon", "crime_index": 26.14, "safety_index": 73.86}
    """
    city = args.get("city")
    if not city:
        raise ValueError("City parameter is required")
    
    result = await fetch_safety_data_for_city(city)
    return result

def main():
    """
    Main execution function for Numbeo safety score data fetching with incremental saving
    """
    try:
        # Check for force refresh flag
        import sys
        force_refresh = "--force" in sys.argv or "--refresh" in sys.argv
        if force_refresh:
            logger.info("Force refresh mode enabled - will fetch all cities")
        
        logger.info("Starting Numbeo safety score data fetch via BrightData...")
        
        # Fetch data using BrightData Browser API with incremental saving
        # This function now handles loading existing data, skipping recent cities,
        # fetching missing/stale cities, converting to safety scores, and saving incrementally
        all_safety_data = asyncio.run(fetch_all_city_safety_data(force_refresh=force_refresh))
        logger.info(f"Process completed - total {len(all_safety_data)} cities available")
        
        # Load final data from file (already converted and saved incrementally)
        final_data = load_existing_safety_data("../data/sources/numbeo_safety_score.json")
        
        logger.info("Numbeo safety score data fetch completed successfully")
        logger.info(f"Data saved to: ../data/sources/numbeo_safety_score.json")
        logger.info(f"Total cities: {len(final_data['cities'])}")
        
        return final_data['cities']
        
    except Exception as e:
        logger.error(f"Failed to fetch Numbeo safety score data: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 