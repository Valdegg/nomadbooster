#!/usr/bin/env python3
"""
Simple Dohop Flight Scraper
Fast, direct approach without MCP complexity

Usage: python simple_dohop_scraper.py
"""

import requests
import json
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import os

def build_dohop_url(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1) -> str:
    """Build Dohop URL for flight search"""
    # Use the correct domain - just dohop.is, not en.dohop.is
    base_url = f"https://dohop.is/flights/{origin.upper()}/{destination.upper()}/{departure_date}"
    
    if return_date:
        base_url += f"/{return_date}"
    
    base_url += f"/adults-{passengers}"
    
    # Add direct flights filter
    base_url += "?stops=0"
    
    return base_url

def extract_prices_from_html(html_content: str) -> List[int]:
    """Extract flight prices from HTML content"""
    prices = []
    
    # Multiple price extraction strategies
    price_patterns = [
        # Currency symbols with numbers
        r'€(\d+(?:,\d{3})*)',          # €123, €1,234
        r'\$(\d+(?:,\d{3})*)',         # $123, $1,234
        r'(\d+(?:,\d{3})*)\s*EUR',     # 123 EUR, 1,234 EUR
        r'(\d+(?:,\d{3})*)\s*USD',     # 123 USD, 1,234 USD
        r'ISK\s*(\d+(?:,\d{3})*)',     # ISK 12,345
        r'(\d+(?:,\d{3})*)\s*ISK',     # 12,345 ISK
        
        # JSON-like patterns
        r'"price"\s*:\s*(\d+)',        # "price": 123
        r'"amount"\s*:\s*(\d+)',       # "amount": 123
        r'"total"\s*:\s*(\d+)',        # "total": 123
        
        # HTML data attributes
        r'data-price=["\'](\d+)',      # data-price="123"
        r'data-amount=["\'](\d+)',     # data-amount="123"
        
        # Generic price patterns (be careful with these)
        r'price[:\s]+(\d{2,5})',       # price: 123, price 1234
        r'(\d{3,5})\s*kr\.?',          # 1234 kr, 12345 kr.
    ]
    
    for pattern in price_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        for match in matches:
            try:
                # Clean up the price (remove commas, etc.)
                price_str = match.replace(',', '').replace('.', '')
                price = int(price_str)
                
                # Filter reasonable flight prices
                # ISK: 50,000 - 500,000 (roughly €300-3000)
                # EUR/USD: 50 - 3000
                if (50 <= price <= 3000) or (50000 <= price <= 500000):
                    prices.append(price)
            except ValueError:
                continue
    
    return sorted(list(set(prices)))

def scrape_dohop_simple(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1) -> Dict:
    """
    Simple Dohop scraper using requests
    
    Returns:
        dict: {
            "url": "https://...",
            "prices": [123, 456, ...],
            "html_saved": "filename.html",
            "success": True
        }
    """
    
    print(f"🛫 Scraping flights: {origin} → {destination}")
    
    # Build URL
    url = build_dohop_url(origin, destination, departure_date, return_date, passengers)
    print(f"🌐 URL: {url}")
    
    # Set up headers to look like a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    try:
        start_time = time.time()
        
        # Make request with session for better cookie/compression handling
        print("📡 Making HTTP request...")
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=30)
        request_time = time.time() - start_time
        
        print(f"✅ Response received in {request_time:.2f}s")
        print(f"📊 Status: {response.status_code}")
        print(f"📏 Content length: {len(response.content)} bytes")
        print(f"🔍 Content encoding: {response.headers.get('Content-Encoding', 'none')}")
        print(f"🔍 Content type: {response.headers.get('Content-Type', 'unknown')}")
        
        # Check if content is binary/compressed
        try:
            # Try to decode as text
            text_content = response.text
            print(f"✅ Successfully decoded text content ({len(text_content)} chars)")
            # Check if it looks like HTML
            if '<html' in text_content.lower() or '<!doctype' in text_content.lower():
                print("✅ Content appears to be HTML")
            else:
                print("⚠️ Content doesn't look like standard HTML")
                print(f"🔍 First 200 chars: {text_content[:200]}")
        except Exception as e:
            print(f"❌ Failed to decode text content: {e}")
            return {
                "url": url,
                "prices": [],
                "error": f"Content decoding failed: {e}",
                "success": False,
                "request_time": request_time
            }
        
        if response.status_code != 200:
            return {
                "url": url,
                "prices": [],
                "error": f"HTTP {response.status_code}",
                "success": False,
                "request_time": request_time
            }
        
        # Save HTML for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"dohop_html_{origin}_{destination}_{timestamp}.html"
        
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"💾 HTML saved to: {html_filename}")
        
        # Extract prices
        print("🔍 Extracting prices...")
        prices = extract_prices_from_html(response.text)
        
        # Convert ISK to EUR (approximate rate: 1 EUR = 140 ISK)
        isk_to_eur_rate = 140
        converted_prices = []
        
        for price in prices:
            if price > 10000:  # Likely ISK
                eur_price = int(price / isk_to_eur_rate)
                converted_prices.append(eur_price)
                print(f"💱 Converted {price:,} ISK → €{eur_price}")
            else:  # Likely already EUR/USD
                converted_prices.append(price)
        
        # Remove duplicates and filter reasonable range
        final_prices = sorted(list(set(converted_prices)))
        final_prices = [p for p in final_prices if 50 <= p <= 2000]
        
        total_time = time.time() - start_time
        
        print(f"✅ Found {len(final_prices)} flight prices in {total_time:.2f}s")
        if final_prices:
            print(f"💰 Price range: €{min(final_prices)} - €{max(final_prices)}")
            print(f"🏆 Cheapest: €{min(final_prices)}")
        
        return {
            "url": url,
            "prices": final_prices,
            "all_raw_prices": prices,  # Include raw prices for debugging
            "cheapest_price_eur": min(final_prices) if final_prices else None,
            "average_price_eur": int(sum(final_prices) / len(final_prices)) if final_prices else None,
            "html_saved": html_filename,
            "content_length": len(response.text),
            "request_time": request_time,
            "total_time": total_time,
            "success": True
        }
        
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {
            "url": url,
            "prices": [],
            "error": str(e),
            "success": False
        }

def scrape_with_selenium(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1) -> Dict:
    """
    Alternative scraper using Selenium for JavaScript-heavy pages
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        print("❌ Selenium not installed. Run: pip install selenium")
        return {"error": "Selenium not installed", "success": False}
    
    print(f"🤖 Using Selenium for {origin} → {destination}")
    
    # Build URL
    url = build_dohop_url(origin, destination, departure_date, return_date, passengers)
    print(f"🌐 URL: {url}")
    
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
    
    try:
        start_time = time.time()
        
        # Create driver
        print("🚀 Starting Chrome browser...")
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to page
        print("📡 Loading page...")
        driver.get(url)
        
        # Wait for page to load
        print("⏱️ Waiting for results to load...")
        time.sleep(8)  # Wait for JavaScript
        
        # Try to wait for specific elements
        try:
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".price, .fare, .flight-result, [data-price]")))
            print("✅ Flight results detected!")
        except:
            print("⚠️ No specific flight elements found, continuing...")
        
        # Get page source
        html_content = driver.page_source
        
        # Save HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"dohop_selenium_{origin}_{destination}_{timestamp}.html"
        
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"💾 HTML saved to: {html_filename}")
        
        # Extract prices
        prices = extract_prices_from_html(html_content)
        
        # Convert and filter prices (same as simple scraper)
        isk_to_eur_rate = 140
        converted_prices = []
        
        for price in prices:
            if price > 10000:  # Likely ISK
                eur_price = int(price / isk_to_eur_rate)
                converted_prices.append(eur_price)
            else:
                converted_prices.append(price)
        
        final_prices = sorted(list(set(converted_prices)))
        final_prices = [p for p in final_prices if 50 <= p <= 2000]
        
        total_time = time.time() - start_time
        
        print(f"✅ Found {len(final_prices)} flight prices in {total_time:.2f}s")
        if final_prices:
            print(f"💰 Cheapest: €{min(final_prices)}")
        
        driver.quit()
        
        return {
            "url": url,
            "prices": final_prices,
            "all_raw_prices": prices,
            "cheapest_price_eur": min(final_prices) if final_prices else None,
            "average_price_eur": int(sum(final_prices) / len(final_prices)) if final_prices else None,
            "html_saved": html_filename,
            "content_length": len(html_content),
            "total_time": total_time,
            "method": "selenium",
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Selenium scraping failed: {e}")
        return {
            "url": url,
            "prices": [],
            "error": str(e),
            "success": False,
            "method": "selenium"
        }

def main():
    """Test both scraping methods"""
    
    print("=== Simple Dohop Flight Scraper ===")
    print("Fast, direct approach without MCP\n")
    
    # Test flight
    test_params = {
        "origin": "BER",
        "destination": "KEF", 
        "departure_date": "2025-07-18",
        "return_date": "2025-07-20",
        "passengers": 1
    }
    
    print(f"Testing: {test_params['origin']} → {test_params['destination']}")
    print(f"Dates: {test_params['departure_date']} to {test_params['return_date']}")
    print()
    
    # Try simple scraping first
    print("=" * 50)
    print("METHOD 1: Simple HTTP Request")
    print("=" * 50)
    
    result1 = scrape_dohop_simple(**test_params)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"dohop_simple_results_{timestamp}.json"
    
    with open(json_filename, 'w') as f:
        json.dump(result1, f, indent=2)
    
    print(f"📄 Results saved to: {json_filename}")
    
    # Always try Selenium since simple HTTP is getting compressed/garbled content
    print("\n" + "=" * 50)
    print("METHOD 2: Selenium (JavaScript) - FORCED")
    print("=" * 50)
    print("⚠️ Simple HTTP got compressed content, trying Selenium...")
    
    result2 = scrape_with_selenium(**test_params)
    
    json_filename2 = f"dohop_selenium_results_{timestamp}.json"
    with open(json_filename2, 'w') as f:
        json.dump(result2, f, indent=2)
    
    print(f"📄 Selenium results saved to: {json_filename2}")
    
    print("\n🎯 SUMMARY:")
    print(f"Simple method: {len(result1.get('prices', []))} prices found")
    if 'result2' in locals():
        print(f"Selenium method: {len(result2.get('prices', []))} prices found")
        if result2.get('success') and result2.get('prices'):
            print(f"🏆 BEST RESULT: Selenium found {len(result2['prices'])} prices!")
            print(f"💰 Cheapest: €{result2['cheapest_price_eur']}")
        elif not result2.get('success'):
            print(f"❌ Selenium error: {result2.get('error', 'Unknown error')}")
    else:
        print("❌ Selenium not attempted")
    
    print("\n💡 Check the saved HTML files to see what content was actually scraped!")

if __name__ == "__main__":
    main() 