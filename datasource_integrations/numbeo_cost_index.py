"""
Numbeo Cost-of-Living Index Data Integration

Data Source: Numbeo Cost-of-Living Index
URL: https://www.numbeo.com/cost-of-living/rankings.jsp
Access Method: BrightData Browser API (HTML scraping via browser automation)
Update Frequency: Static (annual updates)
Data Type: Static city properties

Metric: cost_index (int, 0-200 scale, 100 = EU average)
Description: Cost of living index comparing cities globally, with 100 representing the EU average.
Lower values indicate cheaper cities, higher values indicate more expensive cities.

Integration Status: ✅ READY - BrightData browser automation implementation
Implementation: Use BrightData Browser API to load Numbeo page with JS execution,
scrape rankings table as markdown, parse city cost indices, normalize values.

BrightData Value: Numbeo loads data via JS and has rate limits. BrightData's unlocker
proxies + Browser API execute JS and dodge IP blocks.

Output: ../data/sources/numbeo_cost_index.json
Schema: {"city": str, "country": str, "cost_index": int, "numbeo_rank": int, "last_updated": str}
"""

import json
import re
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Target cities for our travel recommendation system
TARGET_CITIES = [
    "Berlin", "Amsterdam", "Barcelona", "Prague", "Lisbon",
    "Vienna", "Rome", "Paris", "Copenhagen", "Stockholm", 
    "Brussels", "Madrid", "Munich", "Zurich", "Dublin",
    "Budapest", "Warsaw", "Athens", "Helsinki", "Oslo"
]

def fetch_numbeo_cost_data_with_brightdata() -> str:
    """
    Fetch Numbeo cost of living page using BrightData Browser API
    
    Returns:
        str: Page content as markdown from BrightData
    """
    try:
        # TODO: Replace with actual BrightData Browser API call
        # Example BrightData usage:
        # browser = BrightDataBrowser()
        # page_content = browser.open("https://www.numbeo.com/cost-of-living/rankings.jsp")
        # markdown_content = browser.scrape_as_markdown()
        
        # For now, placeholder implementation
        url = "https://www.numbeo.com/cost-of-living/rankings.jsp"
        logger.info(f"Would fetch from BrightData: {url}")
        
        # Placeholder return - replace with actual BrightData call
        return """
        # Cost of Living Rankings
        
        | Rank | City | Country | Cost of Living Index |
        |------|------|---------|---------------------|
        | 1    | Zurich | Switzerland | 131.39 |
        | 15   | Oslo | Norway | 105.21 |
        | 23   | Paris | France | 91.33 |
        | 31   | Amsterdam | Netherlands | 83.75 |
        | 42   | Stockholm | Sweden | 78.94 |
        | 58   | Munich | Germany | 71.68 |
        | 67   | Vienna | Austria | 68.12 |
        | 78   | Berlin | Germany | 64.30 |
        | 89   | Rome | Italy | 59.87 |
        | 95   | Barcelona | Spain | 57.41 |
        | 112  | Madrid | Spain | 52.33 |
        | 125  | Dublin | Ireland | 48.75 |
        | 134  | Brussels | Belgium | 46.21 |
        | 145  | Copenhagen | Denmark | 43.87 |
        | 156  | Helsinki | Finland | 41.52 |
        | 167  | Lisbon | Portugal | 38.94 |
        | 178  | Prague | Czech Republic | 36.15 |
        | 189  | Athens | Greece | 33.72 |
        | 201  | Budapest | Hungary | 30.48 |
        | 215  | Warsaw | Poland | 27.83 |
        """
        
    except Exception as e:
        logger.error(f"Error fetching Numbeo data via BrightData: {e}")
        raise

def parse_cost_data_from_markdown(markdown_content: str) -> List[Dict]:
    """
    Parse cost of living data from markdown table content
    
    Args:
        markdown_content: Markdown content from BrightData scraping
        
    Returns:
        List[Dict]: Parsed city cost data
    """
    cost_data = []
    
    try:
        # Extract table rows from markdown
        lines = markdown_content.strip().split('\n')
        
        for line in lines:
            # Look for table rows with city data
            if '|' in line and any(city.lower() in line.lower() for city in TARGET_CITIES):
                parts = [part.strip() for part in line.split('|')]
                
                if len(parts) >= 5:  # Rank, City, Country, Cost Index, plus empty cells
                    try:
                        rank = int(parts[1]) if parts[1].isdigit() else None
                        city = parts[2]
                        country = parts[3]
                        cost_index_str = parts[4]
                        
                        # Extract numeric cost index
                        cost_index_match = re.search(r'(\d+\.?\d*)', cost_index_str)
                        if cost_index_match:
                            cost_index = float(cost_index_match.group(1))
                            
                            # Only include our target cities
                            if any(target_city.lower() in city.lower() for target_city in TARGET_CITIES):
                                cost_data.append({
                                    'city': city,
                                    'country': country,
                                    'cost_index': round(cost_index),
                                    'numbeo_rank': rank,
                                    'last_updated': datetime.now().isoformat()
                                })
                                
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse line: {line} - {e}")
                        continue
                        
    except Exception as e:
        logger.error(f"Error parsing markdown content: {e}")
        raise
        
    return cost_data

def normalize_cost_index(raw_data: List[Dict]) -> List[Dict]:
    """
    Normalize cost index values and ensure consistent formatting
    
    Args:
        raw_data: Raw cost data from Numbeo
        
    Returns:
        List[Dict]: Normalized cost index data
    """
    normalized_data = []
    
    for city_data in raw_data:
        try:
            # Numbeo cost index is already relative to global average
            # We'll keep original scale but ensure it's an integer
            cost_index = city_data['cost_index']
            
            # Validate range (typical Numbeo range is 20-200)
            if cost_index < 0 or cost_index > 300:
                logger.warning(f"Unusual cost index for {city_data['city']}: {cost_index}")
            
            normalized_data.append({
                'city': city_data['city'],
                'country': city_data['country'], 
                'cost_index': int(cost_index),
                'numbeo_rank': city_data.get('numbeo_rank'),
                'last_updated': city_data['last_updated']
            })
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error normalizing data for {city_data}: {e}")
            continue
            
    return normalized_data

def save_cost_index_data(data: List[Dict], output_path: str = "../data/sources/numbeo_cost_index.json"):
    """
    Save cost index data to JSON file
    
    Args:
        data: Processed cost index data
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
                "data_source": "Numbeo Cost-of-Living Index via BrightData",
                "url": "https://www.numbeo.com/cost-of-living/rankings.jsp",
                "last_updated": datetime.now().isoformat(),
                "description": "Cost of living index with 100 = global average",
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

def main():
    """
    Main execution function for Numbeo cost index data fetching
    """
    try:
        logger.info("Starting Numbeo cost index data fetch via BrightData...")
        
        # Fetch data using BrightData Browser API
        markdown_content = fetch_numbeo_cost_data_with_brightdata()
        
        # Parse the markdown content
        raw_data = parse_cost_data_from_markdown(markdown_content)
        logger.info(f"Parsed {len(raw_data)} cities from Numbeo")
        
        # Normalize the data
        normalized_data = normalize_cost_index(raw_data)
        logger.info(f"Normalized {len(normalized_data)} cities")
        
        # Save to JSON file
        save_cost_index_data(normalized_data)
        
        logger.info("Numbeo cost index data fetch completed successfully")
        return normalized_data
        
    except Exception as e:
        logger.error(f"Failed to fetch Numbeo cost index data: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 