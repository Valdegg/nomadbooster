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
import os
import asyncio
import random
import sys
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
import logging
import statistics
from urllib.parse import urlencode
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

# BrightData configuration (same pattern as Numbeo script)
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

# City to country mapping for proper data attribution
CITY_TO_COUNTRY = {
    "Berlin": "Germany",
    "Munich": "Germany", 
    "Amsterdam": "Netherlands",
    "Barcelona": "Spain",
    "Madrid": "Spain",
    "Prague": "Czech Republic",
    "Lisbon": "Portugal",
    "Vienna": "Austria",
    "Rome": "Italy",
    "Paris": "France",
    "Copenhagen": "Denmark", 
    "Stockholm": "Sweden",
    "Brussels": "Belgium",
    "Zurich": "Switzerland",
    "Dublin": "Ireland",
    "Budapest": "Hungary",
    "Warsaw": "Poland",
    "Athens": "Greece",
    "Helsinki": "Finland",
    "Oslo": "Norway"
}

# Cities to search for accommodation pricing
SEARCH_CITIES = [
    "Berlin", "Amsterdam", "Barcelona", "Prague", "Lisbon",
    "Vienna", "Rome", "Paris", "Copenhagen", "Stockholm", 
    "Brussels", "Madrid", "Munich", "Zurich", "Dublin",
    "Budapest", "Warsaw", "Athens", "Helsinki", "Oslo"
]
import pandas as pd 
iata_data = pd.read_csv('../data/european_iatas_df.csv')
SEARCH_CITIES = set(iata_data['city'].tolist())

def build_airbnb_search_url(city: str, checkin_date: datetime, checkout_date: datetime, guests: int = 2, property_type: str = None) -> str:
    """
    Build Airbnb search URL with proper parameters
    
    Args:
        city: City name to search in
        checkin_date: Check-in date
        checkout_date: Check-out date  
        guests: Number of guests
        property_type: 'entire_place' or 'private_room' to filter by property type
        
    Returns:
        str: Formatted Airbnb search URL
    """
    try:
        # Base URL for Airbnb search
        base_url = "https://www.airbnb.com/s/"
        
        # Format dates for Airbnb (YYYY-MM-DD)
        checkin_str = checkin_date.strftime("%Y-%m-%d")
        checkout_str = checkout_date.strftime("%Y-%m-%d")
        
        # Search parameters - simplified for testing
        params = {
            'adults': guests,
            'checkin': checkin_str,
            'checkout': checkout_str,
            'currency': 'EUR',  # Force Euro currency display
            'locale': 'en',     # Force English language
            'c': 'EUR'          # Alternative currency parameter
        }
        
        # Add property type filter if specified
        if property_type == 'entire_place':
            params['room_types[]'] = 'Entire home/apt'
        elif property_type == 'private_room':
            params['room_types[]'] = 'Private room'
        
        # Build URL - Airbnb expects city in the path
        city_slug = city.lower().replace(' ', '-')
        search_url = f"{base_url}{city_slug}/homes?{urlencode(params)}"
        
        return search_url
        
    except Exception as e:
        logger.error(f"Error building Airbnb search URL: {e}")
        raise

async def fetch_airbnb_data_for_city(city: str, checkin_date: datetime, checkout_date: datetime, guests: int = 2, max_retries: int = 3, property_type: str = None) -> Dict:
    """
    Fetch Airbnb accommodation data for a specific city using BrightData Browser API
    
    Args:
        city: City name to fetch data for
        checkin_date: Check-in date
        checkout_date: Check-out date  
        guests: Number of guests
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dict: Accommodation data for the city
    """
    if not BR_ENDPOINT:
        raise ValueError("BrightData configuration missing. Set BRIGHTDATA_AUTH or BRIGHTDATA_ENDPOINT")
    
    if "brd.superproxy.io" not in BR_ENDPOINT:
        raise ValueError("Invalid BrightData endpoint format")
    
    target_url = build_airbnb_search_url(city, checkin_date, checkout_date, guests, property_type)
    logger.info(f"Target URL: {target_url}")
    
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
                    
                    # Navigate to Airbnb search page
                    await page.goto(target_url, timeout=120_000)
                
                    logger.info(f"Page loaded for {city}, waiting for listings...")
                    
                    # Wait a bit for JS to load
                    await asyncio.sleep(5)
                    
                    # Get page title and URL to check if we're on the right page
                    page_title = await page.title()
                    current_url = page.url
                    logger.info(f"Page title: '{page_title}', Current URL: {current_url}")
                    
                    # Save page screenshot for debugging (always save for now)
                    await page.screenshot(path=f"debug_airbnb_{city.lower()}.png")
                    logger.info(f"Saved debug screenshot: debug_airbnb_{city.lower()}.png")
                    
                    # Save HTML content immediately for inspection
                    content = await page.content()
                    html_file = f"debug_airbnb_{city.lower()}_content.html"
                    with open(html_file, "w") as f:
                        f.write(content)
                    logger.info(f"Saved full HTML content to: {html_file}")
                    
                    # Check for common Airbnb redirects/modals and handle them
                    try:
                        # Close any modal that might be blocking
                        modal_selectors = [
                            '[aria-label="Close"]',
                            '[data-testid="modal-close-button"]', 
                            'button:has-text("Close")',
                            '[aria-label="Dismiss"]'
                        ]
                        for modal_sel in modal_selectors:
                            try:
                                if await page.locator(modal_sel).is_visible():
                                    await page.locator(modal_sel).click()
                                    logger.info(f"Closed modal with selector: {modal_sel}")
                                    await asyncio.sleep(2)
                                    break
                            except:
                                continue
                    except Exception as e:
                        logger.info(f"No modals to close: {e}")
                    
                    # Wait longer for content to load
                    await asyncio.sleep(10)
                    
                    # Wait for Airbnb listings to load (try multiple selectors with more current ones)
                    working_selector = None
                    selectors_to_try = [
                        # Try more generic selectors first
                        '[data-testid*="listing"]',           # Any element with listing in testid
                        '[data-testid="card-container"]',      # Card containers
                        '[data-testid="listing-card-title"]',  # New Airbnb structure
                        '[role="group"]',                      # Any group role elements
                        '[aria-label*="bed"]',                 # Anything with "bed" in aria-label
                        '[data-plugin-in-point-id="SEARCH_RESULTS_LISTINGS"]', # Search results
                        '.t1jojoys',                          # Airbnb listing cards class
                        '._1h6x35l',                          # Alternative listing class
                        '.c1yo0219',                          # Another possible class
                        '[data-testid="card-container"] > div', # Children of card containers
                    ]
                    
                    for selector in selectors_to_try:
                        try:
                            elements = await page.locator(selector).count()
                            if elements > 0:
                                logger.info(f"Found {elements} elements with selector: {selector}")
                                working_selector = selector
                                break
                            else:
                                logger.warning(f"Selector '{selector}' found 0 elements")
                        except Exception as e:
                            logger.warning(f"Selector '{selector}' failed: {e}")
                            continue
                    
                    if not working_selector:
                        logger.error("Could not find any listings on the page")
                        # Log page content for debugging
                        content = await page.content()
                        logger.info(f"Page content preview: {content[:1000]}...")
                        
                        # Save full page content for debugging (always save now)
                        with open(f"debug_airbnb_{city.lower()}_content.html", "w") as f:
                            f.write(content)
                        logger.info(f"Saved debug HTML: debug_airbnb_{city.lower()}_content.html")
                        
                        # Log HTML structure to understand current Airbnb layout
                        logger.info("=== HTML STRUCTURE ANALYSIS ===")
                        html_analysis = await page.evaluate("""
                            () => {
                                // Find all elements that might contain listing data
                                const allDivs = [...document.querySelectorAll('div')];
                                const analysis = {
                                    totalDivs: allDivs.length,
                                    divsWithDataTestId: [],
                                    divsWithPrices: [],
                                    divsWithBeds: [],
                                    potentialListings: []
                                };
                                
                                for (const div of allDivs) {
                                    const testId = div.getAttribute('data-testid');
                                    const text = div.textContent || '';
                                    
                                    // Collect divs with data-testid
                                    if (testId) {
                                        analysis.divsWithDataTestId.push({
                                            testId: testId,
                                            text: text.slice(0, 50),
                                            hasChildren: div.children.length
                                        });
                                    }
                                    
                                    // Look for price indicators
                                    if (text.includes('€') || text.includes('$') || text.includes('night')) {
                                        analysis.divsWithPrices.push({
                                            className: div.className,
                                            testId: testId,
                                            text: text.slice(0, 100)
                                        });
                                    }
                                    
                                    // Look for bed indicators
                                    if (text.includes('bed') || text.includes('room')) {
                                        analysis.divsWithBeds.push({
                                            className: div.className,
                                            testId: testId,
                                            text: text.slice(0, 100)
                                        });
                                    }
                                    
                                    // Look for potential listing containers
                                    if (text.length > 50 && text.length < 500 && 
                                        (text.includes('bed') || text.includes('room') || text.includes('apartment'))) {
                                        analysis.potentialListings.push({
                                            className: div.className,
                                            testId: testId,
                                            text: text.slice(0, 200),
                                            innerHTML: div.innerHTML.slice(0, 300)
                                        });
                                    }
                                }
                                
                                // Limit arrays to prevent huge logs
                                analysis.divsWithDataTestId = analysis.divsWithDataTestId.slice(0, 20);
                                analysis.divsWithPrices = analysis.divsWithPrices.slice(0, 10);
                                analysis.divsWithBeds = analysis.divsWithBeds.slice(0, 10);
                                analysis.potentialListings = analysis.potentialListings.slice(0, 5);
                                
                                return analysis;
                            }
                        """)
                        
                        logger.info(f"Total divs on page: {html_analysis['totalDivs']}")
                        logger.info(f"Divs with data-testid: {len(html_analysis['divsWithDataTestId'])}")
                        logger.info(f"Divs with prices: {len(html_analysis['divsWithPrices'])}")
                        logger.info(f"Divs with beds: {len(html_analysis['divsWithBeds'])}")
                        logger.info(f"Potential listings: {len(html_analysis['potentialListings'])}")
                        
                        logger.info("=== DATA-TESTID ELEMENTS ===")
                        for item in html_analysis['divsWithDataTestId']:
                            logger.info(f"TestID: {item['testId']}, Text: '{item['text']}...', Children: {item['hasChildren']}")
                        
                        logger.info("=== PRICE ELEMENTS ===")
                        for item in html_analysis['divsWithPrices']:
                            logger.info(f"Class: {item['className'][:50]}, TestID: {item['testId']}, Text: '{item['text']}'")
                        
                        logger.info("=== BED ELEMENTS ===")
                        for item in html_analysis['divsWithBeds']:
                            logger.info(f"Class: {item['className'][:50]}, TestID: {item['testId']}, Text: '{item['text']}'")
                        
                        logger.info("=== POTENTIAL LISTINGS ===")
                        for item in html_analysis['potentialListings']:
                            logger.info(f"Class: {item['className'][:50]}, TestID: {item['testId']}")
                            logger.info(f"Text: '{item['text']}'")
                            logger.info(f"HTML: {item['innerHTML'][:100]}...")
                            logger.info("---")
                        
                        # Log all elements on page for debugging
                        all_elements = await page.evaluate("""
                            () => {
                                const elements = document.querySelectorAll('[data-testid], [class*="listing"], [aria-label*="bed"]');
                                return Array.from(elements).slice(0, 20).map(el => ({
                                    tagName: el.tagName,
                                    className: el.className,
                                    testId: el.getAttribute('data-testid'),
                                    ariaLabel: el.getAttribute('aria-label'),
                                    textContent: el.textContent.slice(0, 100)
                                }));
                            }
                        """)
                        logger.info(f"Found elements on page: {all_elements}")
                        
                        raise ValueError(f"No listings found on Airbnb page for {city}")
                    
                    # Find price elements directly (they're separate from listing cards)
                    logger.info("Looking for price elements directly...")
                    price_elements = await page.locator('[data-testid="price-availability-row"]').all()
                    logger.info(f"Found {len(price_elements)} price elements")
                    
                    # Also find listing cards for titles
                    listing_cards = await page.locator(working_selector).all()
                    logger.info(f"Found {len(listing_cards)} listing cards")
                    
                    # Extract price data directly from price elements
                    listings = []
                    max_items_to_process = min(len(price_elements), len(listing_cards), 20)
                    
                    logger.info(f"Processing first {max_items_to_process} price/listing pairs...")
                    
                    for i in range(max_items_to_process):
                        try:
                            # Get price from price element
                            price_element = price_elements[i]
                            price_text = await price_element.text_content() or ''
                            
                            # Get title from listing card
                            card = listing_cards[i] if i < len(listing_cards) else None
                            card_text = await card.text_content() if card else ''
                            
                            logger.info(f"Item {i}: price='{price_text[:50]}...', title='{card_text[:50]}...'")
                            
                            # Extract numeric price from price element
                            import re
                            price = None
                            
                            # Improved patterns for multilingual support (Arabic, Italian, etc.)
                            price_patterns = [
                                r'€\s*(\d{1,3}(?:,\d{3})*)',       # € 105 (most common)
                                r'(\d{1,3}(?:,\d{3})*)\s*€',       # 105 € (German/European)
                                r'€\s*(\d{1,3}(?:,\d{3})*)\s*€',   # € 105 € (duplicate euro in Arabic)
                                r'(\d{1,3}(?:,\d{3})*)\s*EUR',     # 105 EUR
                                r'EUR\s*(\d{1,3}(?:,\d{3})*)',     # EUR 105
                                r'(\d{1,3}(?:,\d{3})*)\s*\$',      # 105 $ (fallback)
                                r'\$\s*(\d{1,3}(?:,\d{3})*)',      # $ 105 (fallback)
                                # Handle cases where Arabic text separates prices
                                r'€\s*(\d{1,3}(?:,\d{3})*)[^€]*€\s*\d+',  # € 119 € 99 (take first)
                            ]
                            
                            for pattern in price_patterns:
                                match = re.search(pattern, price_text)
                                if match:
                                    candidate_price = int(match.group(1).replace(',', ''))
                                    if 15 <= candidate_price <= 500:  # Reasonable range
                                        price = candidate_price
                                        logger.info(f"Item {i}: Extracted price €{price} from '{price_text[:30]}...'")
                                        break
                            
                            if not price:
                                logger.warning(f"Item {i}: Could not extract price from '{price_text[:50]}...'")
                                continue
                            
                            # Extract title from card
                            title = ''
                            if card:
                                title_selectors = ['h3', 'h2', '[data-testid*="title"]', '[aria-label*="bed"]']
                                for title_sel in title_selectors:
                                    if await card.locator(title_sel).count() > 0:
                                        title = await card.locator(title_sel).first.text_content() or ''
                                        if len(title.strip()) > 5:
                                            break
                            
                            # Use the property type filter that was applied in the URL
                            if property_type == 'entire_place':
                                listing_property_type = 'Entire place'
                            elif property_type == 'private_room':
                                listing_property_type = 'Private room'
                            else:
                                # Fallback to text detection if no filter was applied
                                listing_property_type = 'Unknown'
                                if card_text:
                                    lower_text = card_text.lower()
                                    if 'entire home' in lower_text or 'entire place' in lower_text or 'entire apartment' in lower_text:
                                        listing_property_type = 'Entire place'
                                    elif 'private room' in lower_text:
                                        listing_property_type = 'Private room'
                                    elif 'shared room' in lower_text:
                                        listing_property_type = 'Shared room'
                                    elif any(word in lower_text for word in ['studio', 'apartment', 'house', 'loft']):
                                        listing_property_type = 'Entire place'
                            
                            # Add to results
                            listings.append({
                                'price': price,
                                'priceText': price_text.strip(),
                                'title': title.strip() or card_text[:50] + '...' if card_text else f'Listing {i+1}',
                                'propertyType': listing_property_type,
                                'fullText': card_text[:200] if card_text else ''
                            })
                            
                            # Log progress every 5 items
                            if (i + 1) % 5 == 0:
                                logger.info(f"Processed {i + 1}/{max_items_to_process} items, found {len(listings)} with prices")
                        
                        except Exception as e:
                            logger.warning(f"Error processing item {i}: {e}")
                            continue
                    
                    # Create final result
                    accommodation_data = {
                        'totalFound': len(listing_cards),
                        'processedCards': max_items_to_process,
                        'validListings': len(listings),
                        'listings': sorted(listings, key=lambda x: x['price'])  # Sort by price
                    }
                    
                    logger.info(f"Extracted {accommodation_data['validListings']} valid listings from {accommodation_data['totalFound']} total elements")
                    
                    # Log sample data for debugging
                    if accommodation_data['validListings'] > 0:
                        sample_listing = accommodation_data['listings'][0]
                        logger.info(f"Sample listing: €{sample_listing['price']} - {sample_listing['title'][:50]} ({sample_listing['propertyType']})")
                        logger.info(f"Sample price text: '{sample_listing['priceText']}'")
                    else:
                        logger.warning(f"No listings found. Sample page text: {accommodation_data.get('sampleText', 'No sample')[:200]}")
                    
                    if accommodation_data['validListings'] == 0:
                        logger.error(f"No valid accommodation data extracted for {city}")
                        
                        # Add HTML structure analysis when we find listings but no prices
                        logger.info("=== HTML STRUCTURE ANALYSIS (NO PRICES FOUND) ===")
                        html_analysis = await page.evaluate("""
                            () => {
                                // Find all elements that might contain pricing data
                                const allElements = [...document.querySelectorAll('*')];
                                const analysis = {
                                    elementsWithEuro: [],
                                    elementsWithDollar: [],
                                    elementsWithPrice: [],
                                    elementsWithNight: [],
                                    dataTestIdElements: []
                                };
                                
                                for (const el of allElements) {
                                    const text = el.textContent || '';
                                    const innerHTML = el.innerHTML || '';
                                    const testId = el.getAttribute('data-testid') || '';
                                    
                                    // Look for any elements with € symbol
                                    if (text.includes('€') || innerHTML.includes('€')) {
                                        analysis.elementsWithEuro.push({
                                            tag: el.tagName,
                                            className: (el.className || '').toString().slice(0, 50),
                                            testId: testId,
                                            text: text.slice(0, 100),
                                            innerHTML: innerHTML.slice(0, 150)
                                        });
                                    }
                                    
                                    // Look for any elements with $ symbol
                                    if (text.includes('$') || innerHTML.includes('$')) {
                                        analysis.elementsWithDollar.push({
                                            tag: el.tagName,
                                            className: (el.className || '').toString().slice(0, 50),
                                            testId: testId,
                                            text: text.slice(0, 100),
                                            innerHTML: innerHTML.slice(0, 150)
                                        });
                                    }
                                    
                                    // Look for elements with "price" in attributes
                                    const className = el.className || '';
                                    const ariaLabel = el.getAttribute('aria-label') || '';
                                    if (testId.includes('price') || className.toString().includes('price') || 
                                        ariaLabel.includes('price')) {
                                        analysis.elementsWithPrice.push({
                                            tag: el.tagName,
                                            className: (el.className || '').toString().slice(0, 50),
                                            testId: testId,
                                            ariaLabel: el.getAttribute('aria-label'),
                                            text: text.slice(0, 100)
                                        });
                                    }
                                    
                                    // Look for elements with "night"
                                    if (text.includes('night') || innerHTML.includes('night')) {
                                        analysis.elementsWithNight.push({
                                            tag: el.tagName,
                                            className: (el.className || '').toString().slice(0, 50),
                                            testId: testId,
                                            text: text.slice(0, 100)
                                        });
                                    }
                                }
                                
                                // Also get all data-testid elements
                                const testIdElements = [...document.querySelectorAll('[data-testid]')];
                                for (const el of testIdElements.slice(0, 30)) {  // Limit to first 30
                                    analysis.dataTestIdElements.push({
                                        testId: el.getAttribute('data-testid'),
                                        tag: el.tagName,
                                        text: el.textContent.slice(0, 80)
                                    });
                                }
                                
                                // Limit arrays to prevent huge logs
                                analysis.elementsWithEuro = analysis.elementsWithEuro.slice(0, 10);
                                analysis.elementsWithDollar = analysis.elementsWithDollar.slice(0, 10);
                                analysis.elementsWithPrice = analysis.elementsWithPrice.slice(0, 10);
                                analysis.elementsWithNight = analysis.elementsWithNight.slice(0, 10);
                                
                                return analysis;
                            }
                        """)
                        
                        # Save HTML analysis to file for inspection
                        analysis_file = f"debug_airbnb_{city.lower()}_analysis.json"
                        with open(analysis_file, 'w') as f:
                            json.dump(html_analysis, f, indent=2)
                        
                        logger.info(f"Elements with €: {len(html_analysis['elementsWithEuro'])}")
                        logger.info(f"Elements with $: {len(html_analysis['elementsWithDollar'])}")
                        logger.info(f"Elements with 'price': {len(html_analysis['elementsWithPrice'])}")
                        logger.info(f"Elements with 'night': {len(html_analysis['elementsWithNight'])}")
                        logger.info(f"Saved detailed HTML analysis to: {analysis_file}")
                        
                        raise ValueError(f"No valid accommodation data extracted for {city}")
                    
                    # Calculate statistics
                    prices = [listing['price'] for listing in accommodation_data['listings']]
                    property_types = [listing['propertyType'] for listing in accommodation_data['listings']]
                    
                    stats = {
                        'median_price_eur': int(statistics.median(prices)),
                        'mean_price_eur': int(statistics.mean(prices)),
                        'min_price_eur': min(prices),
                        'max_price_eur': max(prices),
                        'sample_size': len(prices),
                        'property_types': list(set(property_types)),
                        'prices': prices  # For further analysis
                    }
                    
                    logger.info(f"Statistics for {city}: median €{stats['median_price_eur']}, range €{stats['min_price_eur']}-{stats['max_price_eur']}, {stats['sample_size']} listings")
                    
                    stay_duration = (checkout_date - checkin_date).days
                    stay_type = "short" if stay_duration <= 7 else "long" if stay_duration >= 30 else "medium"
                    
                    result = {
                        "city": city,
                        "country": CITY_TO_COUNTRY.get(city, "Unknown"),
                        "accommodation_cost_eur": stats['median_price_eur'],
                        "stay_type": stay_type,
                        "stay_duration_days": stay_duration,
                        "travel_date": checkin_date.strftime('%Y-%m-%d'),
                        "guests": guests,
                        "property_types": stats['property_types'],
                        "sample_size": stats['sample_size'],
                        "price_range": {
                            'min': stats['min_price_eur'],
                            'max': stats['max_price_eur'],
                            'mean': stats['mean_price_eur']
                        },
                        "statistics": stats,
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

def load_existing_data(output_path: str = "../data/sources/airbnb_accommodation_costs.json") -> Dict:
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
        "data_source": "Airbnb Accommodation Costs via BrightData",
        "url": "https://www.airbnb.com/s/",
        "last_updated": datetime.now().isoformat(),
        "description": "Dynamic accommodation pricing with heavy JS handling",
        "method": "BrightData Browser API with IP rotation",
        "total_searches": 0,
        "accommodation_costs": []
    }

def save_single_search_data(search_data: Dict, output_path: str = "../data/sources/airbnb_accommodation_costs.json"):
    """
    Save or update data for a single search incrementally
    
    Args:
        search_data: Search result data
        output_path: Output file path
    """
    try:
        # Load existing data
        existing_data = load_existing_data(output_path)
        
        # Add the new search data
        existing_data['accommodation_costs'].append(search_data)
        
        # Update metadata
        existing_data['last_updated'] = datetime.now().isoformat()
        existing_data['total_searches'] = len(existing_data['accommodation_costs'])
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save updated data
        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        logger.info(f"Saved search data for {search_data['city']} (total: {len(existing_data['accommodation_costs'])} searches)")
        
    except Exception as e:
        logger.error(f"Error saving data for {search_data.get('city', '?')}: {e}")
        raise

async def fetch_accommodation_costs_for_cities(cities: List[str], travel_dates: List[tuple], guest_counts: List[int] = [1, 2]) -> List[Dict]:
    """
    Fetch accommodation costs for multiple cities, dates, and guest counts using BrightData
    
    Args:
        cities: List of cities to search
        travel_dates: List of (checkin, checkout) date tuples
        guest_counts: List of guest counts to test
        
    Returns:
        List[Dict]: Accommodation cost data for all searches
    """
    output_path = "../data/sources/airbnb_accommodation_costs.json"
    
    total_searches = len(cities) * len(travel_dates) * len(guest_counts)
    logger.info(f"Will perform {total_searches} searches ({len(cities)} cities × {len(travel_dates)} date ranges × {len(guest_counts)} guest counts)")
    
    search_count = 0
    
    try:
        for city in cities:
            for checkin_date, checkout_date in travel_dates:
                for guests in guest_counts:
                    search_count += 1
                    logger.info(f"Search {search_count}/{total_searches}: {city} ({checkin_date.date()} to {checkout_date.date()}, {guests} guests)")
                    
                    try:
                        # Fetch data for this specific search
                        search_data = await fetch_airbnb_data_for_city(city, checkin_date, checkout_date, guests)
                        
                        # Save immediately
                        save_single_search_data(search_data, output_path)
                        
                        # Add delay between requests to be respectful
                        await asyncio.sleep(5)  # 5 second delay between requests
                        
                    except Exception as e:
                        logger.error(f"Failed search {search_count}/{total_searches} for {city}: {e}")
                        continue
                        
    except Exception as e:
        logger.error(f"Error in batch fetching: {e}")
        
    # Return all collected data
    final_data = load_existing_data(output_path)
    return final_data['accommodation_costs']

async def fetch_both_property_types():
    """
    Fetch data for both entire places and private rooms for all target cities
    """
    logger.info("Starting Airbnb accommodation costs fetch via BrightData...")
    
    # European cities for nomad travel recommendations
    target_cities = [
        "Berlin", "Amsterdam", "Barcelona", "Prague", "Lisbon",
        "Vienna", "Rome", "Paris", "Copenhagen", "Stockholm", 
        "Brussels", "Madrid", "Munich", "Zurich", "Dublin",
        "Budapest", "Warsaw", "Athens", "Helsinki", "Oslo"
    ]
    target_cities = list(SEARCH_CITIES)  # Convert set to list for indexing
    
    # Simple: one date range, one guest count
    checkin_date = datetime.now() + timedelta(days=7)   # 1 week from now
    checkout_date = datetime.now() + timedelta(days=10) # 3-day stay
    guests = 1  # Solo traveler for simplicity
    
    # Search for both entire places and private rooms
    property_types = ['entire_place', 'private_room']
    
    output_path = "../data/sources/airbnb_accommodation_costs.json"
    results = []
    
    total_searches = len(target_cities) * len(property_types)
    search_count = 0
    
    for city in target_cities:
        logger.info(f"\n=== Processing city: {city} ===")
        
        for property_type in property_types:
            search_count += 1
            type_name = "Entire places" if property_type == 'entire_place' else "Private rooms"
            logger.info(f"Search {search_count}/{total_searches}: {type_name} in {city} ({checkin_date.date()} to {checkout_date.date()}, {guests} guest)")
            
            try:
                # Fetch accommodation costs for this city and property type
                search_data = await fetch_airbnb_data_for_city(city, checkin_date, checkout_date, guests, property_type=property_type)
                
                # Add property type info to result
                search_data['property_type_filter'] = property_type
                
                # Save the result immediately
                save_single_search_data(search_data, output_path)
                results.append(search_data)
                
                logger.info(f"✅ Completed {type_name} in {city} - median cost: €{search_data['accommodation_cost_eur']} ({search_data['sample_size']} listings)")
                
                # Wait between searches to avoid rate limiting
                await asyncio.sleep(15)  # Longer delay for multiple cities
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch {type_name} for {city}: {e}")
                # Continue with next search instead of stopping
                continue
        
        # Extra delay between cities
        if city != target_cities[-1]:  # Don't wait after last city
            logger.info(f"Completed {city}, waiting before next city...")
            await asyncio.sleep(30)
    
    # Summary by city
    if results:
        logger.info("\n=== FINAL SUMMARY ===")
        cities_processed = set(result['city'] for result in results)
        for city in cities_processed:
            city_results = [r for r in results if r['city'] == city]
            if len(city_results) == 2:  # Both property types
                entire = next((r for r in city_results if r.get('property_type_filter') == 'entire_place'), None)
                private = next((r for r in city_results if r.get('property_type_filter') == 'private_room'), None)
                if entire and private:
                    logger.info(f"{city}: Entire places €{entire['accommodation_cost_eur']}, Private rooms €{private['accommodation_cost_eur']}")
            else:
                logger.warning(f"{city}: Only {len(city_results)} property type(s) completed")
    
    logger.info(f"\nCompleted {len(results)} successful searches out of {total_searches} total")
    logger.info("Airbnb accommodation costs fetch completed successfully")
    return results

def main():
    """
    Main execution function for Airbnb accommodation costs fetching
    """
    try:
        results = asyncio.run(fetch_both_property_types())
        return results
        
    except Exception as e:
        logger.error(f"Failed to fetch Airbnb accommodation costs: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 