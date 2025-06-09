#!/usr/bin/env python3
"""
BrightData Dohop Flight Scraper
Fast, reliable flight price scraping using BrightData's browser API

Usage: python brightdata_dohop_scraper.py
"""

import json
import re
import os
import asyncio
import random
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

# BrightData configuration (same pattern as Airbnb/Numbeo)
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

def build_dohop_url(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1) -> str:
    """Build Dohop URL for flight search with EUR currency and English language"""
    # Use main domain with language/currency parameters
    base_url = f"https://dohop.is/flights/{origin.upper()}/{destination.upper()}/{departure_date}"
    
    if return_date:
        base_url += f"/{return_date}"
    
    base_url += f"/adults-{passengers}"
    
    # Add query parameters - try multiple language/currency approaches
    params = [
        "stops=0",           # Direct flights only
        "currency=EUR",      # Force EUR pricing
        "lang=en",           # English language
        "locale=en-US"       # US English locale
    ]
    
    base_url += "?" + "&".join(params)
    
    return base_url

async def extract_flight_prices_from_dohop_page(page) -> List[Dict]:
    """
    Extract flight prices from a loaded Dohop page using actual page structure
    
    Returns:
        List[Dict]: List of flight data with prices, times, airlines, etc.
    """
    flights = []
    
    try:
        # Wait for flight results to load
        await asyncio.sleep(8)  # Give time for JavaScript
        
        # Save HTML for debugging (silent)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = f"dohop_brightdata_content_{timestamp}.html"
        content = await page.content()
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Extract from sort buttons (shows price ranges)
        sort_buttons = await page.locator('.SortButtons__price').all()
        sort_prices = []
        for button in sort_buttons:
            price_text = await button.text_content() or ''
            # Extract EUR prices from sort buttons
            price_match = re.search(r'€(\d+)', price_text)
            if price_match:
                sort_prices.append(int(price_match.group(1)))
        
        # Extract from main itinerary price sections
        price_elements = await page.locator('.ItineraryPrice__price').all()
        main_prices = []
        for element in price_elements:
            price_text = await element.text_content() or ''
            price_match = re.search(r'€(\d+)', price_text)
            if price_match:
                main_prices.append(int(price_match.group(1)))
        
        # Extract airline information from headers
        airline_elements = await page.locator('.Itinerary__airline').all()
        airlines = []
        for element in airline_elements:
            airline_text = await element.text_content() or ''
            if airline_text.strip():
                airlines.append(airline_text.strip())
        
        # Extract departure and arrival times
        time_elements = await page.locator('.ItineraryRoute__time').all()
        times = []
        for element in time_elements:
            time_text = await element.text_content() or ''
            if time_text.strip():
                times.append(time_text.strip())
        
        # Extract detailed flight information if available
        detailed_times = []
        detailed_time_elements = await page.locator('.ItineraryDetailedRoute__date').all()
        for element in detailed_time_elements:
            time_text = await element.text_content() or ''
            if time_text.strip():
                detailed_times.append(time_text.strip())
        
        # Extract airline details from detailed view
        detailed_airline_elements = await page.locator('.ItineraryDetailedRoute__airlineText').all()
        detailed_airlines = []
        for element in detailed_airline_elements:
            airline_text = await element.text_content() or ''
            if airline_text.strip() and not airline_text.startswith('OG'):  # Skip flight numbers
                detailed_airlines.append(airline_text.strip())
        
        # Combine all price sources
        all_prices = []
        if sort_prices:
            all_prices.extend(sort_prices)
        if main_prices:
            all_prices.extend(main_prices)
        
        # Remove duplicates and sort
        unique_prices = sorted(list(set([p for p in all_prices if 50 <= p <= 1500])))
        
        # Combine airline sources
        all_airlines = []
        if detailed_airlines:
            all_airlines.extend(detailed_airlines)
        elif airlines:
            all_airlines.extend(airlines)
        
        # Use detailed times if available, otherwise use main times
        all_times = detailed_times if detailed_times else times
        
        # Create flight objects
        if unique_prices:
            for i, price in enumerate(unique_prices):
                # Assign airline (cycle through available airlines)
                airline = 'Unknown'
                if all_airlines:
                    airline = all_airlines[i % len(all_airlines)]
                
                # Assign departure time (cycle through available times)
                departure_time = 'Unknown'
                if all_times:
                    departure_time = all_times[i % len(all_times)]
                
                # Create flight entry
                flights.append({
                    'price_eur': price,
                    'airline': airline,
                    'departure_time': departure_time,
                    'arrival_time': 'Unknown',  # Could extract this too if needed
                    'duration': 'Unknown',      # Could extract from duration elements
                    'stops': 0,                 # Direct flights as per search
                    'flight_number': f'{airline} Flight {i+1}' if airline != 'Unknown' else f'Flight {i+1}',
                    'source': 'dohop_structured'
                })
        
        # If no structured data found, fall back to regex extraction
        if not flights:
            logger.warning("No structured data found, falling back to regex extraction...")
            
            price_patterns = [
                r'€(\d+)',                           # €333
                r'(\d+)\s*€',                        # 333 €
                r'EUR\s*(\d+)',                      # EUR 333
                r'(\d+)\s*EUR',                      # 333 EUR
            ]
            
            extracted_prices = []
            for pattern in price_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        price = int(match.replace(',', ''))
                        if 50 <= price <= 1500:
                            extracted_prices.append(price)
                    except ValueError:
                        continue
            
            # Remove duplicates and sort
            final_prices = sorted(list(set(extracted_prices)))[:5]  # Limit to top 5
            
            for i, price in enumerate(final_prices):
                flights.append({
                    'price_eur': price,
                    'airline': 'Unknown',
                    'departure_time': 'Unknown',
                    'arrival_time': 'Unknown',
                    'duration': 'Unknown',
                    'stops': 0,
                    'flight_number': f'Flight {i+1}',
                    'source': 'regex_fallback'
                })
        
        logger.info(f"Extracted {len(flights)} flight results")
        return flights
        
    except Exception as e:
        logger.error(f"Error extracting flight prices: {e}")
        return []

async def fetch_dohop_flights_brightdata(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1, max_retries: int = 3) -> Dict:
    """
    Fetch flight data from Dohop using BrightData browser automation
    
    Args:
        origin: Origin airport code (e.g., 'BER')
        destination: Destination airport code (e.g., 'KEF')
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date (optional)
        passengers: Number of passengers
        max_retries: Maximum retry attempts
        
    Returns:
        Dict: Flight data with prices and metadata
    """
    start_time = time.time()
    timing_data = {}
    
    if not BR_ENDPOINT:
        raise ValueError("BrightData configuration missing. Set BRIGHTDATA_AUTH or BRIGHTDATA_ENDPOINT")
    
    if "brd.superproxy.io" not in BR_ENDPOINT:
        raise ValueError("Invalid BrightData endpoint format")
    
    target_url = build_dohop_url(origin, destination, departure_date, return_date, passengers)
    logger.info(f"🔗 {origin} → {destination} | {target_url}")
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            attempt_start = time.time()
            if attempt > 0:
                logger.info(f"🔄 Retry {attempt + 1}/{max_retries}")
                wait_time = (2 ** (attempt - 1)) + random.uniform(0, 2)
                await asyncio.sleep(wait_time)
            
            async with async_playwright() as pw:
                # Browser connection timing
                browser_start = time.time()
                browser = await pw.chromium.connect_over_cdp(BR_ENDPOINT)
                timing_data['browser_connection'] = time.time() - browser_start
                
                try:
                    page = await browser.new_page()
                    page.set_default_timeout(120000)  # 2 minutes
                    
                    # Navigation timing
                    nav_start = time.time()
                    logger.info(f"⏳ Navigating to Dohop...")
                    await page.goto(target_url, timeout=120_000)
                    timing_data['navigation'] = time.time() - nav_start
                    
                    # Quick page verification
                    current_url = page.url
                    if "error" in current_url.lower() or "not-found" in current_url.lower():
                        raise ValueError(f"Dohop returned error page: {current_url}")
                    
                    # Take screenshot for debugging (no logging)
                    screenshot_file = f"dohop_brightdata_{origin}_{destination}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await page.screenshot(path=screenshot_file)
                    
                    # Handle cookie consent popup with timing
                    cookie_start = time.time()
                    try:
                        await asyncio.sleep(3)  # Wait for cookie popup
                        
                        # Streamlined cookie consent selectors
                        cookie_selectors = [
                            'button:has-text("Samþykkja")',      # "Accept" in Icelandic
                            'button:has-text("Accept")',
                            'button:has-text("Accept all")',
                            'button:has-text("OK")',
                            '[class*="cookie"] button',
                            '.cookie-consent button',
                            'button[class*="accept"]'
                        ]
                        
                        cookie_handled = False
                        for modal_sel in cookie_selectors[:3]:  # Try top 3 only
                            try:
                                elements = await page.locator(modal_sel).count()
                                if elements > 0:
                                    element = page.locator(modal_sel).first
                                    if await element.is_visible(timeout=1000):
                                        await element.click()
                                        cookie_handled = True
                                        await asyncio.sleep(1)
                                        break
                            except:
                                continue
                        
                        if not cookie_handled:
                            # JavaScript fallback (silent)
                            try:
                                await page.evaluate("""
                                    () => {
                                        const buttons = [...document.querySelectorAll('button')];
                                        const cookieButton = buttons.find(btn => 
                                            btn.textContent.includes('Samþykkja') ||
                                            btn.textContent.includes('Accept') ||
                                            btn.textContent.includes('OK')
                                        );
                                        if (cookieButton) cookieButton.click();
                                    }
                                """)
                            except:
                                pass
                        
                        timing_data['cookie_handling'] = time.time() - cookie_start
                        
                    except Exception as e:
                        timing_data['cookie_handling'] = time.time() - cookie_start
                        logger.debug(f"Cookie handling error: {e}")
                    
                    # Extract flight data with timing  
                    extract_start = time.time()
                    logger.info(f"📊 Extracting flight data...")
                    flights = await extract_flight_prices_from_dohop_page(page)
                    timing_data['data_extraction'] = time.time() - extract_start
                    timing_data['total_time'] = time.time() - start_time
                    
                    if not flights:
                        logger.warning("No flights found, analyzing page structure...")
                        
                        # Quick page analysis
                        page_analysis = await page.evaluate("""
                            () => {
                                const text = document.body.textContent || '';
                                return {
                                    hasFlightText: text.includes('flight') || text.includes('Flight'),
                                    hasPriceText: text.includes('ISK') || text.includes('€') || text.includes('kr'),
                                    hasResultText: text.includes('result') || text.includes('Result'),
                                    textLength: text.length,
                                    sampleText: text.slice(0, 300)
                                };
                            }
                        """)
                        
                        logger.info(f"Page analysis: {page_analysis}")
                        
                        if not page_analysis['hasPriceText']:
                            raise ValueError(f"No price information found on Dohop page for {origin} → {destination}")
                    
                    # Calculate statistics
                    prices = [flight['price_eur'] for flight in flights]
                    
                    if prices:
                        stats = {
                            'cheapest_flight_eur': min(prices),
                            'most_expensive_eur': max(prices),
                            'average_price_eur': int(sum(prices) / len(prices)),
                            'total_flights': len(flights),
                            'price_range': {
                                'min': min(prices),
                                'max': max(prices)
                            }
                        }
                    else:
                        stats = {
                            'cheapest_flight_eur': None,
                            'most_expensive_eur': None,
                            'average_price_eur': None,
                            'total_flights': 0,
                            'price_range': {'min': None, 'max': None}
                        }
                    
                    stay_duration = None
                    if return_date:
                        dep_date = datetime.strptime(departure_date, "%Y-%m-%d")
                        ret_date = datetime.strptime(return_date, "%Y-%m-%d")
                        stay_duration = (ret_date - dep_date).days
                    
                    result = {
                        'success': True,
                        'route': f"{origin} → {destination}",
                        'origin': origin,
                        'destination': destination,
                        'departure_date': departure_date,
                        'return_date': return_date,
                        'stay_duration_days': stay_duration,
                        'passengers': passengers,
                        'trip_type': 'round_trip' if return_date else 'one_way',
                        'dohop_url': target_url,
                        'flights': flights,
                        'statistics': stats,
                        'timing': timing_data,
                        'scrape_timestamp': datetime.now().isoformat(),
                        'data_source': 'Dohop via BrightData Browser API',
                        'screenshots': [screenshot_file]
                    }
                    
                    # Success log with timing
                    total_time = timing_data.get('total_time', 0)
                    nav_time = timing_data.get('navigation', 0) 
                    extract_time = timing_data.get('data_extraction', 0)
                    
                    logger.info(f"✅ Success! {len(flights)} flights | {total_time:.1f}s total (nav: {nav_time:.1f}s, extract: {extract_time:.1f}s)")
                    if prices:
                        logger.info(f"💰 €{min(prices)}-{max(prices)} (avg: €{stats['average_price_eur']})")
                    
                    return result
                    
                finally:
                    await browser.close()
                
        except Exception as e:
            last_error = e
            logger.error(f"Attempt {attempt + 1} failed for {origin} → {destination}: {e}")
            
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts failed for {origin} → {destination}")
                break
    
    # Return error result
    return {
        'success': False,
        'route': f"{origin} → {destination}",
        'origin': origin,
        'destination': destination,
        'departure_date': departure_date,
        'return_date': return_date,
        'passengers': passengers,
        'dohop_url': target_url,
        'error': str(last_error),
        'scrape_timestamp': datetime.now().isoformat(),
        'data_source': 'Dohop via BrightData Browser API'
    }

def main():
    """Test the BrightData Dohop scraper"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== BrightData Dohop Flight Scraper ===")
    print("Using BrightData browser automation for reliable scraping\n")
    
    # Test flight search
    test_params = {
        "origin": "BER",
        "destination": "BCN", 
        "departure_date": "2025-07-18",
        "return_date": "2025-07-20",
        "passengers": 1
    }
    
    print(f"Testing: {test_params['origin']} → {test_params['destination']}")
    print(f"Dates: {test_params['departure_date']} to {test_params['return_date']}")
    print(f"Passengers: {test_params['passengers']}\n")
    
    # Check BrightData configuration
    if not BR_ENDPOINT:
        print("❌ BrightData configuration missing!")
        print("Set BRIGHTDATA_AUTH or BRIGHTDATA_ENDPOINT environment variable")
        return
    
    print(f"🔗 BrightData endpoint: {BR_ENDPOINT[:50]}...")
    
    async def run_test():
        try:
            result = await fetch_dohop_flights_brightdata(**test_params)
            
            # Save results to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"dohop_brightdata_results_{timestamp}.json"
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n📄 Results saved to: {json_filename}")
            
            # Display results
            print(f"\n✅ Results:")
            print(f"Route: {result['route']}")
            print(f"Success: {result['success']}")
            
            if result['success']:
                stats = result['statistics']
                timing = result.get('timing', {})
                
                print(f"Flights Found: {stats['total_flights']}")
                print(f"⏱️  Total Time: {timing.get('total_time', 0):.1f}s")
                print(f"   └ Navigation: {timing.get('navigation', 0):.1f}s")
                print(f"   └ Cookie Handling: {timing.get('cookie_handling', 0):.1f}s")
                print(f"   └ Data Extraction: {timing.get('data_extraction', 0):.1f}s")
                
                if stats['cheapest_flight_eur']:
                    print(f"\n💰 Cheapest Flight: €{stats['cheapest_flight_eur']}")
                    print(f"💰 Average Price: €{stats['average_price_eur']}")
                    print(f"💰 Price Range: €{stats['price_range']['min']} - €{stats['price_range']['max']}")
                    
                    print(f"\n🛫 Sample Flights:")
                    for i, flight in enumerate(result['flights'][:3]):
                        print(f"  {i+1}. €{flight['price_eur']} - {flight['airline']} at {flight['departure_time']}")
                else:
                    print("No prices found")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise
    
    # Run the test
    result = asyncio.run(run_test())
    
    print(f"\n🎯 Summary:")
    if result['success']:
        print(f"✅ Successfully found {result['statistics']['total_flights']} flights")
        if result['statistics']['cheapest_flight_eur']:
            print(f"🏆 Best price: €{result['statistics']['cheapest_flight_eur']}")
    else:
        print(f"❌ Scraping failed: {result.get('error', 'Unknown error')}")
    
    print("\n💡 Check the saved JSON file and screenshots for detailed results!")

if __name__ == "__main__":
    main() 