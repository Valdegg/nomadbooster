"""
Numbeo Cost-of-Living Detailed Data Integration

Data Source: Numbeo Cost-of-Living Database
URL: https://www.numbeo.com/cost-of-living/in/{city}
Access Method: BrightData Browser API (HTML scraping via browser automation)
Update Frequency: Monthly (Numbeo updates regularly)
Data Type: Detailed cost breakdown by category

Metrics: Complete cost breakdown including:
- Restaurants (meals, drinks, coffee)
- Markets (groceries, produce, household items)
- Transportation (public transport, taxi, fuel)
- Utilities (electricity, heating, internet)
- Sports & Leisure (gym, cinema, sports)
- Childcare (preschool, primary school)
- Clothing & Shoes (jeans, dress, sneakers)
- Rent (apartment prices by location and size)

Plus calculated cost_index based on key indicators.

Integration Status: ✅ READY - BrightData browser automation implementation
Implementation: Use BrightData Browser API to load Numbeo page with JS execution,
scrape complete cost tables, parse and categorize all pricing data.

BrightData Value: Numbeo loads data via JS and has rate limits. BrightData's unlocker
proxies + Browser API execute JS and dodge IP blocks.

Output: ../data/sources/numbeo_detailed_costs.json
Schema: {
  "city": str, 
  "cost_index": float, 
  "total_items": int,
  "costs_by_category": {"category": {"item": {"price": float, "currency": str}}},
  "detailed_costs": {"item": {"price": float, "currency": str, "category": str}},
  "last_updated": str
}
"""

import json
import re
import os
import asyncio
import random
import sys
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



async def fetch_cost_index_for_city(city: str, max_retries: int = 1) -> Dict:
    """
    Fetch cost index for a specific city using BrightData Browser API
    
    Args:
        city: City name to fetch data for
        
    Returns:
        Dict: City cost data {"city": str, "cost_index": float}
    """
    if not BR_ENDPOINT:
        raise ValueError("BrightData configuration missing. Set BRIGHTDATA_AUTH or BRIGHTDATA_ENDPOINT")
    
    if "brd.superproxy.io" not in BR_ENDPOINT:
        raise ValueError("Invalid BrightData endpoint format")
    
    target_url = f"https://www.numbeo.com/cost-of-living/in/{city}"
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for {city}")
            
            # Add jitter delay between attempts
            if attempt > 0:
                wait_time = (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.info(f"Waiting {wait_time:.2f}s before retry...")
                await asyncio.sleep(wait_time)
            
            async with async_playwright() as pw:
                browser = await pw.chromium.connect_over_cdp(BR_ENDPOINT)
                try:
                    page = await browser.new_page()
                    await page.goto(target_url, timeout=120_000)
                
                    logger.info(f"Page loaded for {city}, waiting for table...")
                    
                    # Save page screenshot for debugging (optional)
                    if os.getenv("DEBUG_NUMBEO"):
                        await page.screenshot(path=f"debug_{city.lower()}.png")
                        logger.info(f"Saved debug screenshot: debug_{city.lower()}.png")
                    
                    # Wait until Numbeo's JS table is rendered
                    table_found = False
                    selectors_to_try = [
                        "table.data_wide_table",
                        "table[data-test='cost-of-living-table']",
                        "table.tablesorter",
                        ".cost-of-living-table",
                        "table:has(td:contains('Cost of Living Index'))"
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
                        logger.error("Could not find any table on the page")
                        # Log page content for debugging
                        content = await page.content()
                        logger.info(f"Page content preview: {content[:1000]}...")
                        
                        # Save full page content for debugging
                        if os.getenv("DEBUG_NUMBEO"):
                            with open(f"debug_{city.lower()}_content.html", "w") as f:
                                f.write(content)
                            logger.info(f"Saved debug HTML: debug_{city.lower()}_content.html")
                        
                        raise ValueError(f"No data table found on Numbeo page for {city}")
                    
                    # Log available tables for debugging
                    tables_info = await page.evaluate("""
                        () => {
                            const tables = [...document.querySelectorAll('table')];
                            return tables.map(table => ({
                                className: table.className,
                                id: table.id,
                                rowCount: table.rows ? table.rows.length : 0
                            }));
                        }
                    """)
                    logger.info(f"Available tables: {tables_info}")
                    
                    # Log all rows in the main table for debugging
                    rows_info = await page.evaluate("""
                        () => {
                            const rows = [...document.querySelectorAll('table.data_wide_table tr')];
                            return rows.map((row, index) => ({
                                index: index,
                                text: row.textContent.trim(),
                                cellCount: row.children.length
                            }));
                        }
                    """)
                    logger.info(f"Table rows found: {len(rows_info) if rows_info else 0}")
                    for i, row in enumerate(rows_info[:10] if rows_info else []):  # Log first 10 rows
                        logger.info(f"Row {i}: {row}")
                    
                    # Extract all cost of living data from the main table
                    cost_data = await page.evaluate("""
                        () => {
                            const table = document.querySelector('table.data_wide_table');
                            if (!table) return { found: false, error: 'No main table found' };
                            
                            const rows = [...table.querySelectorAll('tr')];
                            const costs = {};
                            let currentCategory = null;
                            
                            for (const row of rows) {
                                const cells = [...row.children];
                                if (cells.length < 2) continue;
                                
                                const itemText = cells[0].textContent.trim();
                                const priceText = cells[1].textContent.trim();
                                
                                // Skip header rows and categories
                                if (itemText.includes('Edit') || !priceText || priceText.includes('Range')) {
                                    if (itemText.includes('Edit')) {
                                        currentCategory = itemText.replace('Edit', '').trim();
                                    }
                                    continue;
                                }
                                
                                // Extract price value (remove currency symbols and extra text)
                                const priceMatch = priceText.match(/([0-9,]+\\.?[0-9]*)/);
                                if (priceMatch) {
                                    const price = parseFloat(priceMatch[1].replace(/,/g, ''));
                                    const currency = priceText.match(/[€$£¥kr]/)?.[0] || '';
                                    
                                    costs[itemText] = {
                                        price: price,
                                        currency: currency,
                                        fullText: priceText,
                                        category: currentCategory
                                    };
                                }
                            }
                            
                            return {
                                found: true,
                                totalItems: Object.keys(costs).length,
                                costs: costs
                            };
                        }
                    """)
                    
                    logger.info(f"Cost data extraction result: Found {cost_data.get('totalItems', 0)} items")
                    
                    if not cost_data or not cost_data.get('found'):
                        logger.error(f"Could not extract cost data for {city}")
                        raise ValueError(f"Could not extract cost data for {city}")
                    
                    # Calculate a simple cost index based on a few key items
                    key_items = {
                        'Meal, Inexpensive Restaurant': 1.0,
                        'Domestic Beer (1 pint draught)': 0.5,
                        'Cappuccino (regular)': 0.3,
                        'Milk (regular), (1 liter)': 0.2
                    }
                    
                    cost_index = 0
                    items_found = 0
                    
                    for item, weight in key_items.items():
                        if item in cost_data['costs']:
                            cost_index += cost_data['costs'][item]['price'] * weight
                            items_found += 1
                    
                    if items_found > 0:
                        cost_index = round(cost_index, 2)
                    else:
                        cost_index = None
                    
                    logger.info(f"Fetched {len(cost_data['costs'])} cost items for {city}, calculated index: {cost_index}")
                    
                    result = {
                        "city": city,
                        "cost_index": cost_index,
                        "detailed_costs": cost_data['costs'],
                        "total_items": cost_data['totalItems'],
                        "last_updated": datetime.now().isoformat()
                    }
                    
                    # If we get here, the fetch was successful
                    return result
                    
                finally:
                    await browser.close()
                
        except Exception as e:
            last_error = e
            logger.error(f"Attempt {attempt + 1} failed for {city}: {e}")
            
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts failed for {city}")
                raise last_error
    
    # This shouldn't be reached, but just in case
    raise last_error or Exception(f"Failed to fetch data for {city}")

def load_existing_data(output_path: str = "../data/sources/numbeo_detailed_costs.json") -> Dict:
    """
    Load existing data file if it exists
    
    Returns:
        Dict: Existing data structure or empty structure
    """
    try:
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load existing data: {e}")
    
    # Return empty structure if file doesn't exist or can't be loaded
    return {
        "data_source": "Numbeo Detailed Cost-of-Living Data via BrightData",
        "url": "https://www.numbeo.com/cost-of-living/",
        "last_updated": datetime.now().isoformat(),
        "description": "Comprehensive cost breakdown for restaurants, markets, transportation, utilities, and more",
        "total_cities": 0,
        "cities": []
    }

def is_city_data_recent(city_data: Dict, max_age_hours: int = 24) -> bool:
    """
    Check if city data is recent enough to skip re-fetching
    
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

def save_single_city_data(city_data: Dict, output_path: str = "../data/sources/numbeo_detailed_costs.json"):
    """
    Save or update data for a single city incrementally
    
    Args:
        city_data: Normalized city data
        output_path: Output file path
    """
    try:
        # Load existing data
        existing_data = load_existing_data(output_path)
        
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
        existing_data['total_cities'] = len(existing_data['cities'])
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save updated data
        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        logger.info(f"Saved/updated data for {city_name} (total: {len(existing_data['cities'])} cities)")
        
    except Exception as e:
        logger.error(f"Error saving data for {city_data.get('city', '?')}: {e}")
        raise

async def fetch_all_city_cost_data(force_refresh: bool = False) -> List[Dict]:
    """
    Fetch cost index data for all target cities with incremental saving
    
    Args:
        force_refresh: If True, ignore existing data age and refresh all cities
    
    Returns:
        List[Dict]: Cost data for all cities (loaded from saved file)
    """
    output_path = "../data/sources/numbeo_detailed_costs.json"
    
    # Load existing data to check what we already have
    existing_data = load_existing_data(output_path)
    existing_cities = {city['city']: city for city in existing_data['cities']}
    
    cities_to_fetch = []
    cities_skipped = []
    
    # Determine which cities need to be fetched
    for city in TARGET_CITIES:
        if not force_refresh and city in existing_cities and is_city_data_recent(existing_cities[city]):
            cities_skipped.append(city)
            logger.info(f"Skipping {city} - data is recent")
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
            logger.info(f"Fetching data for {city}...")
            city_data = await fetch_cost_index_for_city(city)
            
            # Normalize single city data
            normalized_city = normalize_cost_data([city_data])[0]
            
            # Save immediately
            save_single_city_data(normalized_city, output_path)
            
            # Add small delay to be respectful
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {city}: {e}")
            continue
    
    # Return all data (existing + newly fetched)
    final_data = load_existing_data(output_path)
    return final_data['cities']

def normalize_cost_data(raw_data: List[Dict]) -> List[Dict]:
    """
    Normalize cost data and ensure consistent formatting
    
    Args:
        raw_data: Raw cost data from Numbeo
        
    Returns:
        List[Dict]: Normalized cost data with detailed breakdown
    """
    normalized_data = []
    
    for city_data in raw_data:
        try:
            cost_index = city_data.get('cost_index')
            detailed_costs = city_data.get('detailed_costs', {})
            
            # Validate cost index if present
            if cost_index is not None and (cost_index < 0 or cost_index > 1000):
                logger.warning(f"Unusual cost index for {city_data['city']}: {cost_index}")
            
            # Organize costs by category
            categorized_costs = {}
            for item, data in detailed_costs.items():
                category = data.get('category', 'Other')
                if category not in categorized_costs:
                    categorized_costs[category] = {}
                categorized_costs[category][item] = {
                    'price': data['price'],
                    'currency': data['currency'],
                    'full_text': data['fullText']
                }
            
            normalized_data.append({
                'city': city_data['city'],
                'cost_index': round(cost_index, 2) if cost_index else None,
                'total_items': city_data.get('total_items', 0),
                'costs_by_category': categorized_costs,
                'detailed_costs': detailed_costs,
                'last_updated': city_data['last_updated']
            })
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error normalizing data for {city_data}: {e}")
            continue
            
    return normalized_data

def save_cost_index_data(data: List[Dict], output_path: str = "../data/sources/numbeo_detailed_costs.json"):
    """
    Save cost index data to JSON file
    
    Args:
        data: Processed cost index data
        output_path: Output file path
    """
    try:
        # Create the data structure
        json_data = {
            "data_source": "Numbeo Detailed Cost-of-Living Data via BrightData",
            "url": "https://www.numbeo.com/cost-of-living/",
            "last_updated": datetime.now().isoformat(),
            "description": "Comprehensive cost breakdown for restaurants, markets, transportation, utilities, and more",
            "total_cities": len(data),
            "cities": data
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save data
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        logger.info(f"Saved {len(data)} cities to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        raise

# MCP Tool wrapper for single city fetching
async def tool_fetch_cost_index(args: dict) -> dict:
    """
    MCP tool signature for fetching cost index for a specific city
    
    Args:
        args: {"city": "Lisbon"}
        
    Returns:
        dict: {"city": "Lisbon", "cost_index": 49.6}
    """
    city = args.get("city")
    if not city:
        raise ValueError("City parameter is required")
    
    result = await fetch_cost_index_for_city(city)
    return result

def main():
    """
    Main execution function for Numbeo cost index data fetching with incremental saving
    """
    try:
        # Check for force refresh flag
        force_refresh = "--force" in sys.argv or "--refresh" in sys.argv
        if force_refresh:
            logger.info("Force refresh mode enabled - will fetch all cities")
        
        logger.info("Starting Numbeo cost index data fetch via BrightData...")
        
        # Fetch data using BrightData Browser API with incremental saving
        # This function now handles loading existing data, skipping recent cities,
        # fetching missing/stale cities, and saving incrementally
        all_city_data = asyncio.run(fetch_all_city_cost_data(force_refresh=force_refresh))
        logger.info(f"Process completed - total {len(all_city_data)} cities available")
        
        # Load final data from file (already normalized and saved incrementally)
        final_data = load_existing_data("../data/sources/numbeo_detailed_costs.json")
        
        logger.info("Numbeo cost index data fetch completed successfully")
        logger.info(f"Data saved to: ../data/sources/numbeo_detailed_costs.json")
        logger.info(f"Total cities: {final_data['total_cities']}")
        
        return final_data['cities']
        
    except Exception as e:
        logger.error(f"Failed to fetch Numbeo cost index data: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 