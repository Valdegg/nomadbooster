#!/usr/bin/env python3
"""
Dohop Flight Lookup Tool
Fast, reliable flight price lookup using BrightData browser automation

Function: lookup_dohop_flights
- Fetches flight prices from Dohop.is
- Uses BrightData proxy network for reliable scraping
- Returns structured flight data with prices, airlines, and timing
"""

import json
import re
import os
import asyncio
import random
import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# BrightData configuration
AUTH = os.getenv("BRIGHTDATA_AUTH")
BR_ENDPOINT = os.getenv("BRIGHTDATA_ENDPOINT")

# Auto-extract auth from endpoint if needed
if not AUTH and BR_ENDPOINT and "@brd.superproxy.io" in BR_ENDPOINT:
    try:
        auth_part = BR_ENDPOINT.split("@")[0].replace("wss://", "")
        if auth_part.startswith("brd-customer-"):
            AUTH = auth_part
    except Exception:
        pass

# Generate endpoint from auth if needed
if not BR_ENDPOINT and AUTH:
    BR_ENDPOINT = f"wss://{AUTH}@brd.superproxy.io:9222"

class FlightLookupArgs(BaseModel):
    """Arguments for Dohop flight lookup"""
    
    origin: str = Field(
        ..., 
        description="Origin airport code (3 letters, e.g., 'BER', 'NYC', 'LON')",
        min_length=3,
        max_length=3
    )
    
    destination: str = Field(
        ..., 
        description="Destination airport code (3 letters, e.g., 'KEF', 'PAR', 'BCN')",
        min_length=3,
        max_length=3
    )
    
    departure_date: str = Field(
        ..., 
        description="Departure date in YYYY-MM-DD format (e.g., '2025-07-18')",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    
    return_date: Optional[str] = Field(
        None, 
        description="Return date in YYYY-MM-DD format for round trips (optional)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    
    passengers: int = Field(
        1, 
        description="Number of passengers (1-9)",
        ge=1,
        le=9
    )
    
    @field_validator('origin', 'destination')
    @classmethod
    def validate_airport_codes(cls, v):
        """Validate airport codes are uppercase and 3 letters"""
        if not v.isalpha():
            raise ValueError("Airport codes must contain only letters")
        return v.upper()
    
    @field_validator('departure_date', 'return_date')
    @classmethod
    def validate_dates(cls, v):
        """Validate date format and that it's not in the past"""
        if v is None:
            return v
        try:
            date_obj = datetime.strptime(v, "%Y-%m-%d")
            if date_obj.date() < datetime.now().date():
                raise ValueError("Date cannot be in the past")
            return v
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError("Date must be in YYYY-MM-DD format")
            raise e

def build_dohop_url(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1) -> str:
    """Build Dohop URL for flight search with EUR currency"""
    base_url = f"https://dohop.is/flights/{origin.upper()}/{destination.upper()}/{departure_date}"
    
    if return_date:
        base_url += f"/{return_date}"
    
    base_url += f"/adults-{passengers}"
    
    # Force EUR currency and English language
    params = [
        "stops=0",           # Direct flights only
        "currency=EUR",      # Force EUR pricing
        "lang=en",           # English language
        "locale=en-US"       # US English locale
    ]
    
    base_url += "?" + "&".join(params)
    return base_url

async def extract_flight_data_from_page(page) -> List[Dict]:
    """Extract flight data from loaded Dohop page"""
    flights = []
    
    try:
        # Wait for JavaScript to load flight results
        await asyncio.sleep(8)
        
        # Save HTML for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = f"dohop_content_{timestamp}.html"
        content = await page.content()
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Extract prices from sort buttons
        sort_buttons = await page.locator('.SortButtons__price').all()
        sort_prices = []
        for button in sort_buttons:
            price_text = await button.text_content() or ''
            price_match = re.search(r'€(\d+)', price_text)
            if price_match:
                sort_prices.append(int(price_match.group(1)))
        
        # Extract prices from main itinerary sections
        price_elements = await page.locator('.ItineraryPrice__price').all()
        main_prices = []
        for element in price_elements:
            price_text = await element.text_content() or ''
            price_match = re.search(r'€(\d+)', price_text)
            if price_match:
                main_prices.append(int(price_match.group(1)))
        
        # Extract airline information
        airline_elements = await page.locator('.Itinerary__airline').all()
        airlines = []
        for element in airline_elements:
            airline_text = await element.text_content() or ''
            if airline_text.strip():
                airlines.append(airline_text.strip())
        
        # Extract departure times
        time_elements = await page.locator('.ItineraryRoute__time').all()
        times = []
        for element in time_elements:
            time_text = await element.text_content() or ''
            if time_text.strip():
                times.append(time_text.strip())
        
        # Extract detailed flight info if available
        detailed_times = []
        detailed_time_elements = await page.locator('.ItineraryDetailedRoute__date').all()
        for element in detailed_time_elements:
            time_text = await element.text_content() or ''
            if time_text.strip():
                detailed_times.append(time_text.strip())
        
        detailed_airlines = []
        detailed_airline_elements = await page.locator('.ItineraryDetailedRoute__airlineText').all()
        for element in detailed_airline_elements:
            airline_text = await element.text_content() or ''
            if airline_text.strip() and not airline_text.startswith('OG'):
                detailed_airlines.append(airline_text.strip())
        
        # Combine and clean data
        all_prices = []
        if sort_prices:
            all_prices.extend(sort_prices)
        if main_prices:
            all_prices.extend(main_prices)
        
        unique_prices = sorted(list(set([p for p in all_prices if 50 <= p <= 1500])))
        
        all_airlines = detailed_airlines if detailed_airlines else airlines
        all_times = detailed_times if detailed_times else times
        
        # Create flight objects
        if unique_prices:
            for i, price in enumerate(unique_prices):
                airline = all_airlines[i % len(all_airlines)] if all_airlines else 'Unknown'
                departure_time = all_times[i % len(all_times)] if all_times else 'Unknown'
                
                flights.append({
                    'price_eur': price,
                    'airline': airline,
                    'departure_time': departure_time,
                    'arrival_time': 'Unknown',
                    'duration': 'Unknown',
                    'stops': 0,  # Direct flights only
                    'flight_number': f'{airline} Flight {i+1}' if airline != 'Unknown' else f'Flight {i+1}',
                    'source': 'dohop_structured'
                })
        
        # Regex fallback if no structured data
        if not flights:
            price_patterns = [r'€(\d+)', r'(\d+)\s*€', r'EUR\s*(\d+)', r'(\d+)\s*EUR']
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
            
            final_prices = sorted(list(set(extracted_prices)))[:5]
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
        
        return flights
        
    except Exception as e:
        logger.error(f"Error extracting flight data: {e}")
        return []

async def lookup_dohop_flights(
    origin: str,
    destination: str, 
    departure_date: str,
    return_date: Optional[str] = None,
    passengers: int = 1,
    max_retries: int = 3,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Look up flight prices on Dohop using BrightData browser automation
    
    Args:
        origin: Origin airport code (3 letters, e.g., 'BER')
        destination: Destination airport code (3 letters, e.g., 'KEF') 
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date for round trips (optional)
        passengers: Number of passengers (1-9)
        max_retries: Maximum retry attempts (1-5)
        
    Returns:
        Dict containing:
        - success: bool
        - route: str
        - flights: List[Dict] with price, airline, departure_time
        - statistics: Dict with price ranges and averages
        - timing: Dict with performance metrics
        - error: str (if failed)
        
    Example:
        result = await lookup_dohop_flights('BER', 'KEF', '2025-07-18', '2025-07-20')
        if result['success']:
            print(f"Found {len(result['flights'])} flights")
            print(f"Cheapest: €{result['statistics']['cheapest_flight_eur']}")
    """
    
    # Validate arguments
    try:
        args = FlightLookupArgs(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            passengers=passengers
        )
    except Exception as e:
        return {
            'success': False,
            'error': f"Invalid arguments: {e}",
            'route': f"{origin} → {destination}",
            'scrape_timestamp': datetime.now().isoformat()
        }
    
    # Check BrightData configuration
    if not BR_ENDPOINT:
        return {
            'success': False,
            'error': "BrightData configuration missing. Set BRIGHTDATA_AUTH or BRIGHTDATA_ENDPOINT",
            'route': f"{args.origin} → {args.destination}",
            'scrape_timestamp': datetime.now().isoformat()
        }
    
    if "brd.superproxy.io" not in BR_ENDPOINT:
        return {
            'success': False,
            'error': "Invalid BrightData endpoint format",
            'route': f"{args.origin} → {args.destination}",
            'scrape_timestamp': datetime.now().isoformat()
        }
    
    start_time = time.time()
    timing_data = {}
    target_url = build_dohop_url(args.origin, args.destination, args.departure_date, args.return_date, args.passengers)
    
    logger.info(f"🔗 {args.origin} → {args.destination} | {target_url}")
    
    # Send initial progress update
    if progress_callback:
        try:
            await progress_callback({
                "type": "tool_progress",
                "tool": "lookup_flight_prices",
                "message": f"🔗 Searching flights {args.origin} → {args.destination}",
                "status": "connecting"
            })
        except Exception:
            pass  # Don't fail if progress callback fails
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = (2 ** (attempt - 1)) + random.uniform(0, 2)
                await asyncio.sleep(wait_time)
            
            async with async_playwright() as pw:
                # Browser connection
                browser_start = time.time()
                browser = await pw.chromium.connect_over_cdp(BR_ENDPOINT)
                timing_data['browser_connection'] = time.time() - browser_start
                
                try:
                    page = await browser.new_page()
                    page.set_default_timeout(120000)
                    
                    # Navigation
                    nav_start = time.time()
                    logger.info(f"⏳ Navigating to Dohop...")
                    
                    # Send navigation progress
                    if progress_callback:
                        try:
                            await progress_callback({
                                "type": "tool_progress",
                                "tool": "lookup_flight_prices", 
                                "message": "⏳ Navigating to Dohop...",
                                "status": "navigating"
                            })
                        except Exception:
                            pass
                    
                    await page.goto(target_url, timeout=120_000)
                    timing_data['navigation'] = time.time() - nav_start
                    
                    # Verify page loaded correctly
                    current_url = page.url
                    if "error" in current_url.lower() or "not-found" in current_url.lower():
                        raise ValueError(f"Dohop returned error page: {current_url}")
                    
                    # Screenshot for debugging
                    screenshot_file = f"dohop_{args.origin}_{args.destination}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await page.screenshot(path=screenshot_file)
                    
                    # Handle cookie consent
                    cookie_start = time.time()
                    
                    # Send cookie handling progress
                    if progress_callback:
                        try:
                            await progress_callback({
                                "type": "tool_progress",
                                "tool": "lookup_flight_prices",
                                "message": "🍪 Handling cookie consent...",
                                "status": "initializing"
                            })
                        except Exception:
                            pass
                    
                    try:
                        await asyncio.sleep(3)  # Wait for cookie popup
                        
                        cookie_selectors = [
                            'button:has-text("Samþykkja")',     # Accept in Icelandic
                            'button:has-text("Accept")',
                            'button:has-text("Accept all")',
                            'button:has-text("OK")',
                            '[class*="cookie"] button',
                            '.cookie-consent button'
                        ]
                        
                        cookie_handled = False
                        for selector in cookie_selectors[:3]:  # Try top 3
                            try:
                                elements = await page.locator(selector).count()
                                if elements > 0:
                                    element = page.locator(selector).first
                                    if await element.is_visible(timeout=1000):
                                        await element.click()
                                        cookie_handled = True
                                        await asyncio.sleep(1)
                                        break
                            except:
                                continue
                        
                        # JavaScript fallback
                        if not cookie_handled:
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
                        
                    except Exception:
                        timing_data['cookie_handling'] = time.time() - cookie_start
                    
                    # Extract flight data
                    extract_start = time.time()
                    logger.info(f"📊 Extracting flight data...")
                    
                    # Send data extraction progress
                    if progress_callback:
                        try:
                            await progress_callback({
                                "type": "tool_progress",
                                "tool": "lookup_flight_prices",
                                "message": "📊 Extracting flight data...",
                                "status": "extracting"
                            })
                        except Exception:
                            pass
                    
                    flights = await extract_flight_data_from_page(page)
                    timing_data['data_extraction'] = time.time() - extract_start
                    timing_data['total_time'] = time.time() - start_time
                    
                    if not flights:
                        # Quick page analysis
                        page_analysis = await page.evaluate("""
                            () => {
                                const text = document.body.textContent || '';
                                return {
                                    hasPriceText: text.includes('ISK') || text.includes('€') || text.includes('kr'),
                                    textLength: text.length
                                };
                            }
                        """)
                        
                        if not page_analysis['hasPriceText']:
                            raise ValueError(f"No price information found on Dohop page")
                    
                    # Calculate statistics
                    prices = [flight['price_eur'] for flight in flights]
                    
                    if prices:
                        stats = {
                            'cheapest_flight_eur': min(prices),
                            'most_expensive_eur': max(prices),
                            'average_price_eur': int(sum(prices) / len(prices)),
                            'total_flights': len(flights),
                            'price_range': {'min': min(prices), 'max': max(prices)}
                        }
                    else:
                        stats = {
                            'cheapest_flight_eur': None,
                            'most_expensive_eur': None,
                            'average_price_eur': None,
                            'total_flights': 0,
                            'price_range': {'min': None, 'max': None}
                        }
                    
                    # Calculate stay duration
                    stay_duration = None
                    if args.return_date:
                        dep_date = datetime.strptime(args.departure_date, "%Y-%m-%d")
                        ret_date = datetime.strptime(args.return_date, "%Y-%m-%d")
                        stay_duration = (ret_date - dep_date).days
                    
                    # Success response
                    result = {
                        'success': True,
                        'route': f"{args.origin} → {args.destination}",
                        'origin': args.origin,
                        'destination': args.destination,
                        'departure_date': args.departure_date,
                        'return_date': args.return_date,
                        'stay_duration_days': stay_duration,
                        'passengers': args.passengers,
                        'trip_type': 'round_trip' if args.return_date else 'one_way',
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
                    
                    # Send completion progress
                    if progress_callback:
                        try:
                            await progress_callback({
                                "type": "tool_progress",
                                "tool": "lookup_flight_prices",
                                "message": f"✅ Found {len(flights)} flights! Processing results..." if flights else "❌ Flight search completed with errors",
                                "status": "completing" if flights else "error"
                            })
                        except Exception:
                            pass
                    
                    return result
                    
                finally:
                    await browser.close()
                
        except Exception as e:
            last_error = e
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries - 1:
                break
    
    # Error response
    return {
        'success': False,
        'route': f"{args.origin} → {args.destination}",
        'origin': args.origin,
        'destination': args.destination,
        'departure_date': args.departure_date,
        'return_date': args.return_date,
        'passengers': args.passengers,
        'dohop_url': target_url,
        'error': str(last_error),
        'timing': timing_data,
        'scrape_timestamp': datetime.now().isoformat(),
        'data_source': 'Dohop via BrightData Browser API'
    }

# Tool metadata for integration
TOOL_INFO = {
    'name': 'lookup_dohop_flights',
    'description': 'Look up flight prices and schedules on Dohop using BrightData browser automation',
    'function': lookup_dohop_flights,
    'args_schema': FlightLookupArgs,
    'category': 'travel',
    'requires_config': ['BRIGHTDATA_AUTH', 'BRIGHTDATA_ENDPOINT'],
    'examples': [
        {
            'description': 'Berlin to Reykjavik round trip',
            'args': {
                'origin': 'BER',
                'destination': 'KEF', 
                'departure_date': '2025-07-18',
                'return_date': '2025-07-20',
                'passengers': 1
            }
        },
        {
            'description': 'One-way London to Barcelona',
            'args': {
                'origin': 'LON',
                'destination': 'BCN',
                'departure_date': '2025-08-15',
                'passengers': 2
            }
        }
    ]
}

if __name__ == "__main__":
    """Test the Dohop flight lookup tool"""
    import asyncio
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_lookup():
        print("=== Dohop Flight Lookup Tool Test ===\n")
        
        # Test flight lookup
        result = await lookup_dohop_flights(
            origin='BER',
            destination='KEF',
            departure_date='2025-07-18',
            return_date='2025-07-20',
            passengers=1
        )
        
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Route: {result['route']}")
            print(f"Flights found: {result['statistics']['total_flights']}")
            
            timing = result.get('timing', {})
            print(f"Total time: {timing.get('total_time', 0):.1f}s")
            
            if result['statistics']['cheapest_flight_eur']:
                print(f"Price range: €{result['statistics']['price_range']['min']} - €{result['statistics']['price_range']['max']}")
                print("\nSample flights:")
                for flight in result['flights'][:3]:
                    print(f"  €{flight['price_eur']} - {flight['airline']} at {flight['departure_time']}")
        else:
            print(f"Error: {result['error']}")
    
    asyncio.run(test_lookup()) 