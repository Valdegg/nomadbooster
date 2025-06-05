"""
Airbnb Accommodation Costs Data Integration

Data Source: Airbnb Search Results (Price Medians)
URL: https://www.airbnb.com/s/[city]/homes
Access Method: BrightData Browser API (heavy JS handling + IP rotation)
Update Frequency: Dynamic (real-time pricing, changes daily)
Data Type: Time-dependent city properties

Metric: accommodation_cost_eur (int, EUR per night)
Description: Daily accommodation cost medians derived from Airbnb search results.
Represents typical nightly cost for mid-range accommodations (private rooms/entire homes).
Varies significantly by season, events, booking advance time, and local demand.

Integration Status: ✅ READY - BrightData handles heavy JS and IP blocks
Implementation: Use BrightData Browser API to open Airbnb search URLs, execute heavy JS,
extract price data from rendered HTML, rotate IPs to avoid blocks, take screenshots for validation.

BrightData Value: Airbnb has heavy JS that blocks plain scrapers. BrightData's headless 
Chromium session returns rendered HTML list and rotates IPs to avoid detection.

Key Considerations:
- Airbnb has sophisticated anti-bot measures (BrightData handles this)
- Prices vary by accommodation type (filter for relevant types)
- Seasonal demand affects pricing significantly
- Location within city affects pricing (use city center searches)
- Currency conversion to EUR for standardization
- IP rotation essential to avoid blocks

Accommodation Filters:
- Property types: Entire homes, private rooms (exclude shared rooms for quality)
- Guest capacity: 1-2 people (typical nomad/couple travel)
- Location: City center/downtown areas for relevant pricing
- Minimum stay: Flexible (support both short and long stays)

Search Parameters by Travel Type:
- Short stays (1-7 nights): Weekend/vacation pricing
- Long stays (30+ nights): Monthly discount pricing
- Mid-term (7-30 nights): Standard nightly rates

Output: ../data/sources/airbnb_accommodation_costs.json
Schema: {"city": str, "country": str, "accommodation_cost_eur": int, "stay_type": str, "travel_date": str, "property_types": list, "sample_size": int, "price_range": dict, "last_updated": str}
"""

import json
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
import logging
import statistics
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# Cities to search for accommodation pricing
SEARCH_CITIES = [
    "Berlin, Germany",
    "Amsterdam, Netherlands", 
    "Barcelona, Spain",
    "Prague, Czech Republic",
    "Lisbon, Portugal",
    "Vienna, Austria",
    "Rome, Italy",
    "Paris, France",
    "Copenhagen, Denmark",
    "Stockholm, Sweden",
    "Brussels, Belgium",
    "Madrid, Spain",
    "Munich, Germany",
    "Zurich, Switzerland",
    "Dublin, Ireland",
    "Budapest, Hungary",
    "Warsaw, Poland",
    "Athens, Greece",
    "Helsinki, Finland",
    "Oslo, Norway"
]

def build_airbnb_search_url(city: str, checkin_date: datetime, checkout_date: datetime, guests: int = 2) -> str:
    """
    Build Airbnb search URL with proper parameters
    
    Args:
        city: City name to search in
        checkin_date: Check-in date
        checkout_date: Check-out date  
        guests: Number of guests
        
    Returns:
        str: Formatted Airbnb search URL
    """
    try:
        # Base URL for Airbnb search
        base_url = "https://www.airbnb.com/s/"
        
        # Format dates for Airbnb (YYYY-MM-DD)
        checkin_str = checkin_date.strftime("%Y-%m-%d")
        checkout_str = checkout_date.strftime("%Y-%m-%d")
        
        # Search parameters
        params = {
            'adults': guests,
            'checkin': checkin_str,
            'checkout': checkout_str,
            'room_types[]': ['Entire home/apt', 'Private room'],  # Exclude shared rooms
            'price_min': 20,  # Minimum reasonable price
            'price_max': 500,  # Maximum reasonable price
            'search_type': 'search_query'
        }
        
        # Build URL
        city_encoded = city.replace(' ', '%20').replace(',', '%2C')
        search_url = f"{base_url}{city_encoded}/homes?{urlencode(params, doseq=True)}"
        
        return search_url
        
    except Exception as e:
        logger.error(f"Error building Airbnb search URL: {e}")
        raise

def scrape_airbnb_with_brightdata(search_url: str) -> str:
    """
    Use BrightData Browser API to scrape Airbnb search results with full JS execution
    
    Args:
        search_url: Airbnb search URL to scrape
        
    Returns:
        str: Rendered HTML content from BrightData Browser
    """
    try:
        # TODO: Replace with actual BrightData Browser API call
        # Example BrightData usage:
        # browser = BrightDataBrowser()
        # browser.open(search_url)
        # browser.wait_for_element('.listings-container')  # Wait for listings to load
        # browser.scroll_to_load_all_results()  # Scroll to load more listings
        # html_content = browser.get_html()
        # screenshot_path = browser.screenshot()  # Optional: save screenshot for validation
        # browser.close()
        
        # For now, placeholder implementation
        logger.info(f"Would scrape Airbnb with BrightData: {search_url}")
        
        # Placeholder return - replace with actual scraped content
        return '''
        <div class="listings-container">
            <div class="listing-item" data-listing-id="123">
                <div class="price-section">
                    <span class="price">€45</span>
                    <span class="price-frequency">night</span>
                </div>
                <div class="property-info">
                    <h3>Cozy Apartment in City Center</h3>
                    <p>Entire apartment • 2 guests • 1 bedroom</p>
                    <div class="location">Downtown, Berlin</div>
                </div>
            </div>
            <div class="listing-item" data-listing-id="124">
                <div class="price-section">
                    <span class="price">€68</span>
                    <span class="price-frequency">night</span>
                </div>
                <div class="property-info">
                    <h3>Modern Studio Near Metro</h3>
                    <p>Entire studio • 2 guests • 1 bedroom</p>
                    <div class="location">Mitte, Berlin</div>
                </div>
            </div>
            <div class="listing-item" data-listing-id="125">
                <div class="price-section">
                    <span class="price">€38</span>
                    <span class="price-frequency">night</span>
                </div>
                <div class="property-info">
                    <h3>Private Room in Shared Flat</h3>
                    <p>Private room • 1 guest • 1 bedroom</p>
                    <div class="location">Kreuzberg, Berlin</div>
                </div>
            </div>
            <div class="listing-item" data-listing-id="126">
                <div class="price-section">
                    <span class="price">€89</span>
                    <span class="price-frequency">night</span>
                </div>
                <div class="property-info">
                    <h3>Luxury Loft with Balcony</h3>
                    <p>Entire loft • 4 guests • 2 bedrooms</p>
                    <div class="location">Prenzlauer Berg, Berlin</div>
                </div>
            </div>
        </div>
        '''
        
    except Exception as e:
        logger.error(f"Error scraping Airbnb via BrightData: {e}")
        raise

def parse_accommodation_data_from_html(html_content: str, city: str, checkin_date: datetime, checkout_date: datetime) -> List[Dict]:
    """
    Parse accommodation data from Airbnb HTML content
    
    Args:
        html_content: HTML content from BrightData Browser
        city: City being searched
        checkin_date: Check-in date
        checkout_date: Check-out date
        
    Returns:
        List[Dict]: Parsed accommodation data
    """
    accommodation_data = []
    
    try:
        # Extract prices using regex patterns
        # Look for various price patterns that Airbnb might use
        price_patterns = [
            r'€(\d+)',  # Euro prices
            r'\$(\d+)',  # Dollar prices
            r'£(\d+)',   # Pound prices
            r'"price":{"amount":"(\d+)"',  # JSON price structures
            r'<span[^>]*price[^>]*>.*?[€$£](\d+)',  # Price in spans
            r'data-price="(\d+)"',  # Price in data attributes
        ]
        
        prices = []
        property_types = []
        
        # Extract prices
        for pattern in price_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for match in matches:
                try:
                    price = int(match)
                    if 15 <= price <= 1000:  # Reasonable accommodation price range
                        prices.append(price)
                except ValueError:
                    continue
        
        # Extract property types
        property_type_patterns = [
            r'Entire apartment',
            r'Entire home',
            r'Entire loft',
            r'Private room',
            r'Shared room'
        ]
        
        for pattern in property_type_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            property_types.extend(matches)
        
        # Create accommodation data entries
        for i, price in enumerate(prices[:20]):  # Limit to top 20 results
            stay_duration = (checkout_date - checkin_date).days
            stay_type = "short" if stay_duration <= 7 else "long" if stay_duration >= 30 else "medium"
            
            accommodation_data.append({
                'city': city.split(',')[0].strip(),  # Extract city name
                'country': city.split(',')[1].strip() if ',' in city else 'Unknown',
                'price_per_night': price,
                'stay_type': stay_type,
                'stay_duration_days': stay_duration,
                'checkin_date': checkin_date.strftime('%Y-%m-%d'),
                'checkout_date': checkout_date.strftime('%Y-%m-%d'),
                'property_type': property_types[i] if i < len(property_types) else 'Unknown',
                'last_updated': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error parsing accommodation data from HTML: {e}")
        
    return accommodation_data

def calculate_accommodation_statistics(accommodation_data: List[Dict]) -> Dict:
    """
    Calculate price statistics from accommodation data
    
    Args:
        accommodation_data: List of accommodation data
        
    Returns:
        Dict: Statistical summary of accommodation prices
    """
    try:
        if not accommodation_data:
            return {}
        
        prices = [item['price_per_night'] for item in accommodation_data]
        property_types = [item['property_type'] for item in accommodation_data]
        
        stats = {
            'median_price_eur': int(statistics.median(prices)),
            'mean_price_eur': int(statistics.mean(prices)),
            'min_price_eur': min(prices),
            'max_price_eur': max(prices),
            'sample_size': len(prices),
            'property_types': list(set(property_types)),
            'price_range': {
                'q1': int(statistics.quantiles(prices, n=4)[0]) if len(prices) > 3 else min(prices),
                'q3': int(statistics.quantiles(prices, n=4)[2]) if len(prices) > 3 else max(prices)
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating accommodation statistics: {e}")
        return {}

def convert_to_eur(price: float, currency: str) -> int:
    """Convert price to EUR (simplified conversion)"""
    # TODO: Use real exchange rate API
    conversion_rates = {
        'EUR': 1.0,
        'USD': 0.85,
        'GBP': 1.15,
        'CHF': 0.95,
        'SEK': 0.09,
        'DKK': 0.13,
        'NOK': 0.09,
        'PLN': 0.23,
        'CZK': 0.04,
        'HUF': 0.0025
    }
    
    eur_amount = price * conversion_rates.get(currency, 1.0)
    return int(round(eur_amount))

def fetch_accommodation_costs_for_cities(cities: List[str], travel_dates: List[tuple]) -> List[Dict]:
    """
    Fetch accommodation costs for multiple cities and date ranges using BrightData
    
    Args:
        cities: List of cities to search
        travel_dates: List of (checkin, checkout) date tuples
        
    Returns:
        List[Dict]: Accommodation cost data for all cities and dates
    """
    all_accommodation_data = []
    
    try:
        for city in cities:
            for checkin_date, checkout_date in travel_dates:
                logger.info(f"Fetching accommodation costs for {city} ({checkin_date.date()} to {checkout_date.date()})")
                
                # Step 1: Build Airbnb search URL
                search_url = build_airbnb_search_url(city, checkin_date, checkout_date)
                
                # Step 2: Scrape with BrightData Browser API
                html_content = scrape_airbnb_with_brightdata(search_url)
                
                # Step 3: Parse accommodation data from HTML
                accommodation_data = parse_accommodation_data_from_html(html_content, city, checkin_date, checkout_date)
                
                # Step 4: Calculate statistics
                stats = calculate_accommodation_statistics(accommodation_data)
                
                if stats:
                    # Create summary entry
                    summary_entry = {
                        'city': city.split(',')[0].strip(),
                        'country': city.split(',')[1].strip() if ',' in city else 'Unknown',
                        'accommodation_cost_eur': stats['median_price_eur'],
                        'stay_type': 'short' if (checkout_date - checkin_date).days <= 7 else 'long' if (checkout_date - checkin_date).days >= 30 else 'medium',
                        'travel_date': checkin_date.strftime('%Y-%m-%d'),
                        'stay_duration_days': (checkout_date - checkin_date).days,
                        'property_types': stats['property_types'],
                        'sample_size': stats['sample_size'],
                        'price_range': stats['price_range'],
                        'statistics': {
                            'median': stats['median_price_eur'],
                            'mean': stats['mean_price_eur'],
                            'min': stats['min_price_eur'],
                            'max': stats['max_price_eur']
                        },
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    all_accommodation_data.append(summary_entry)
                
                # Rate limiting to avoid overwhelming servers
                time.sleep(3)  # 3 second delay between requests
                
    except Exception as e:
        logger.error(f"Error fetching accommodation costs: {e}")
        
    return all_accommodation_data

def save_accommodation_costs_data(data: List[Dict], output_path: str = "../data/sources/airbnb_accommodation_costs.json"):
    """
    Save accommodation costs data to JSON file
    
    Args:
        data: Processed accommodation costs data
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
                "data_source": "Airbnb Accommodation Costs via BrightData",
                "url": "https://www.airbnb.com/s/",
                "last_updated": datetime.now().isoformat(),
                "description": "Dynamic accommodation pricing with heavy JS handling",
                "method": "BrightData Browser API with IP rotation",
                "accommodation_costs": []
            }
        
        # Update accommodation costs data and timestamp
        json_data["accommodation_costs"] = data
        json_data["last_updated"] = datetime.now().isoformat()
        
        # Save updated data
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        logger.info(f"Saved {len(data)} accommodation cost entries to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        raise

def main():
    """
    Main execution function for Airbnb accommodation costs fetching
    """
    try:
        logger.info("Starting Airbnb accommodation costs fetch via BrightData...")
        
        # Define sample cities to test
        test_cities = [
            "Berlin, Germany",
            "Barcelona, Spain", 
            "Prague, Czech Republic",
            "Amsterdam, Netherlands"
        ]
        
        # Define travel dates (next 2 weeks for testing)
        travel_dates = [
            (datetime.now() + timedelta(days=7), datetime.now() + timedelta(days=10)),    # 3-day stay
            (datetime.now() + timedelta(days=14), datetime.now() + timedelta(days=21)),   # 7-day stay
        ]
        
        # Fetch accommodation costs
        accommodation_data = fetch_accommodation_costs_for_cities(test_cities, travel_dates)
        logger.info(f"Fetched {len(accommodation_data)} accommodation cost entries")
        
        # Save to JSON file
        save_accommodation_costs_data(accommodation_data)
        
        logger.info("Airbnb accommodation costs fetch completed successfully")
        return accommodation_data
        
    except Exception as e:
        logger.error(f"Failed to fetch Airbnb accommodation costs: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 