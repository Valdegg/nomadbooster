#!/usr/bin/env python3
"""
Tool for fetching flight costs from Dohop using Bright Data MCP browser automation.
Cleaner alternative to Skyscanner - no car rental ads, less noise

Tool Usage: {"origin": "Berlin", "destination": "Barcelona", "departure_date": "2025-07-15", "passengers": 1}
Returns: {"route": "Berlin → Barcelona", "cheapest_flight_eur": 89, "flight_prices": [...]}
"""

import json
import asyncio
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# MCP client imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ MCP library not installed. Run: pip install mcp")
    raise

# Load .env files for configuration
try:
    from dotenv import load_dotenv
    env_files = ['.env', 'mcp/.env', '../.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"📄 Loaded configuration from {env_file}")
            break
except ImportError:
    logger.info("📄 python-dotenv not installed, using manual .env loading")
    env_files = ['.env', 'mcp/.env', '../.env']
    for env_file in env_files:
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key.strip() not in os.environ:
                            os.environ[key.strip()] = value.strip()
            logger.info(f"📄 Manually loaded configuration from {env_file}")
            break
        except FileNotFoundError:
            continue

# Cache for environment variables
_env_cache = None

class FlightLookupArgs(BaseModel):
    """Arguments schema for Dohop flight cost lookup tool"""
    origin: str = Field(description="Origin city or airport code (e.g., 'Berlin', 'BER')")
    destination: str = Field(description="Destination city or airport code (e.g., 'Barcelona', 'BCN')")
    departure_date: str = Field(description="Departure date in YYYY-MM-DD format (e.g., '2025-07-15')")
    return_date: Optional[str] = Field(default=None, description="Return date for round-trip flights")
    passengers: int = Field(default=1, ge=1, le=9, description="Number of passengers (1-9)")

def _load_env_vars():
    """Load and cache environment variables from .env file"""
    global _env_cache
    
    if _env_cache is not None:
        return _env_cache
    
    env_vars = {}
    api_token = None
    
    # Try environment variables first
    api_token = os.getenv('API_TOKEN') or os.getenv('BRIGHTDATA_API_TOKEN') or os.getenv('BRIGHTDATA_API_KEY')
    
    # Try .env file in mcp directory and load ALL variables
    if not api_token:
        env_paths = ['mcp/.env', '../mcp/.env', '../../mcp/.env']
        for env_path in env_paths:
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
                            if key.strip() in ['API_KEY', 'API_TOKEN', 'BRIGHTDATA_API_TOKEN', 'BRIGHTDATA_API_KEY']:
                                api_token = value.strip()
                if api_token:
                    break
            except FileNotFoundError:
                continue
    
    if not api_token:
        raise ValueError("No Bright Data API token found. Set BRIGHTDATA_API_TOKEN or create mcp/.env file")
    
    # Set up environment with ALL variables from .env
    env = os.environ.copy()
    env.update(env_vars)
    env['API_TOKEN'] = api_token
    
    # Cache the result
    _env_cache = env
    return env

async def get_mcp_session():
    """Create an MCP session connection to Bright Data server"""
    env = _load_env_vars()
    
    server_params = StdioServerParameters(
        command="npx",
        args=["@brightdata/mcp"],
        env=env
    )
    
    return stdio_client(server_params)

def build_dohop_url(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1, direct_only: bool = True) -> str:
    """
    Build a Dohop URL for flight search - trying different URL formats
    
    Format options:
    1. https://dohop.is/flights/{origin}/{dest}/{departure_date}/{return_date}/adults-{passengers}?stops=0
    2. https://en.dohop.is/flights/... (English version)
    3. Alternative format based on actual Dohop URLs
    """
    
    # Try English version to avoid Icelandic interface
    use_english = os.getenv('DOHOP_USE_ENGLISH', 'true').lower() == 'true'
    base_domain = "en.dohop.is" if use_english else "dohop.is"
    
    # Check if we should use alternative URL format
    url_format = os.getenv('DOHOP_URL_FORMAT', 'standard').lower()
    
    if url_format == 'alternative':
        # Alternative format: try query parameter approach
        base_url = f"https://{base_domain}/flights/search"
        params = [
            f"origin={origin.upper()}",
            f"destination={destination.upper()}", 
            f"departure={departure_date}",
            f"adults={passengers}"
        ]
        if return_date:
            params.append(f"return={return_date}")
        if direct_only:
            params.append("stops=0")
        
        return base_url + "?" + "&".join(params)
    
    else:
        # Standard format with English subdomain
        base_url = f"https://{base_domain}/flights/{origin.upper()}/{destination.upper()}/{departure_date}"
        
        # Add return date if specified
        if return_date:
            base_url += f"/{return_date}"
        
        # Add passengers  
        base_url += f"/adults-{passengers}"
        
        # Add query parameters
        params = []
        
        # Add direct flights only filter
        if direct_only:
            params.append("stops=0")
        
        # Force English language
        if use_english:
            params.append("lang=en")
        
        if params:
            base_url += "?" + "&".join(params)
        
        return base_url

async def scrape_dohop_flights(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1) -> dict:
    """
    Use MCP browser automation to scrape flight data from Dohop
    
    Returns:
        dict: {
            "dohop_url": "https://...",
            "flight_prices": [89, 134, 167, ...],
            "cheapest_flight_eur": 89,
            "average_price_eur": 134,
            "scrape_successful": True
        }
    """
    
    start_time = time.time()
    
    # Build Dohop URL (with direct flights preference)
    direct_flights_only = os.getenv('DOHOP_DIRECT_ONLY', 'true').lower() == 'true'
    dohop_url = build_dohop_url(origin, destination, departure_date, return_date, passengers, direct_only=direct_flights_only)
    logger.info(f"🌐 Navigating to Dohop: {dohop_url}")
    logger.info(f"🛫 Direct flights only: {direct_flights_only}")
    
    try:
        async with await get_mcp_session() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Get available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools if hasattr(tools_result, 'tools') else []
                tool_names = [tool.name for tool in tools]
                logger.info(f"🛠️ Available tools: {len(tool_names)}")
                
                # Use browser automation if available, otherwise fall back to direct scraping
                if "scraping_browser_navigate" in tool_names:
                    logger.info("🚀 Using browser automation for Dohop...")
                    
                    # Navigate to Dohop
                    await session.call_tool("scraping_browser_navigate", arguments={"url": dohop_url})
                    logger.info("✅ Navigated to Dohop page")
                    
                    # Wait for JavaScript to load flight results
                    wait_time = int(os.getenv('DOHOP_JS_WAIT', '10'))  # Longer wait to ensure results load
                    logger.info(f"⏱️ Waiting {wait_time}s for flight results to load...")
                    
                    # Try to wait for specific elements that indicate results have loaded
                    try:
                        logger.info("🔍 Trying to wait for flight result elements...")
                        await session.call_tool("scraping_browser_wait_for", arguments={
                            "selector": ".flight-result, .price, .fare, [data-flight], .result-item",
                            "timeout": 8000  # 8 seconds max
                        })
                        logger.info("✅ Flight results elements detected!")
                    except:
                        logger.info("⚠️ No specific flight elements found, using fallback wait...")
                        await asyncio.sleep(wait_time)
                    
                    # Extract content
                    extraction_method = os.getenv('DOHOP_EXTRACT', 'text').lower()
                    if extraction_method == 'html':
                        logger.info("📄 Extracting HTML content...")
                        page_result = await session.call_tool("scraping_browser_get_html", arguments={"full_page": False})
                    else:
                        logger.info("📄 Extracting text content...")
                        page_result = await session.call_tool("scraping_browser_get_text", arguments={})
                    
                    if hasattr(page_result, 'content'):
                        content = page_result.content[0].text if hasattr(page_result.content, '__iter__') else str(page_result.content)
                    else:
                        # Fallback to direct scraping
                        logger.info("⚠️ Browser extraction failed, falling back to direct scraping...")
                        scrape_result = await session.call_tool("scrape_as_markdown", arguments={"url": dohop_url})
                        content = scrape_result.content[0].text if hasattr(scrape_result.content, '__iter__') else str(scrape_result.content)
                
                else:
                    logger.info("🌐 Using direct scraping (browser tools not available)...")
                    scrape_result = await session.call_tool("scrape_as_markdown", arguments={"url": dohop_url})
                    content = scrape_result.content[0].text if hasattr(scrape_result.content, '__iter__') else str(scrape_result.content)
                
                # Extract prices from Dohop content
                flight_prices = []
                page_content = content
                
                logger.info(f"🔍 Page content length: {len(content)} characters")
                logger.info(f"🔍 Content preview: {content[:200]}...")
                
                # Check if we're getting the search form instead of results
                if any(keyword in content.lower() for keyword in ['leitaðu', 'frá', 'til', 'brottför', 'search form']):
                    logger.warning("⚠️ Detected search form page instead of results - URL might be incorrect")
                
                if content:
                    logger.info(f"🔍 Page content length: {len(content)} characters")
                    logger.info(f"🔍 Content preview: {content[:200]}...")
                    
                    # Check if we're getting the search form instead of results
                    if any(keyword in content.lower() for keyword in ['leitaðu', 'frá', 'til', 'search form', 'enter destination']):
                        logger.warning("⚠️ Detected search form page instead of results - URL might be incorrect")
                    
                    # Dohop price patterns - more specific to avoid car rental prices
                    price_patterns = [
                        # Primary currency patterns
                        r'(\d+(?:,\d{3})*)\s*USD',  # USD prices: 123 USD
                        r'USD\s*(\d+(?:,\d{3})*)',  # USD prices: USD 123
                        r'(\d+(?:,\d{3})*)\s*EUR',  # EUR prices: 123 EUR  
                        r'EUR\s*(\d+(?:,\d{3})*)',  # EUR prices: EUR 123
                        r'\$(\d+(?:,\d{3})*)',      # Dollar sign: $123
                        r'€(\d+(?:,\d{3})*)',       # Euro sign: €123
                        r'ISK\s*(\d+(?:,\d{3})*)',  # Icelandic Krona: ISK 123
                        r'(\d+(?:,\d{3})*)\s*ISK',  # Icelandic Krona: 123 ISK
                        
                        # JSON/API patterns
                        r'"price"\s*:\s*(\d+)',     # "price": 123
                        r'"total"\s*:\s*(\d+)',     # "total": 123  
                        r'"fare"\s*:\s*(\d+)',      # "fare": 123
                        r'"amount"\s*:\s*(\d+)',    # "amount": 123
                        
                        # HTML data patterns
                        r'data-price=["\'](\d+)',   # data-price="123"
                        r'data-amount=["\'](\d+)',  # data-amount="123"
                        
                        # Generic number patterns (last resort)
                        r'price[:\s]*(\d{2,5})',    # price: 123 or price 1234
                        r'(\d{2,5})\s*kr',          # 1234 kr (Icelandic)
                    ]
                    
                    for pattern in price_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            try:
                                # Remove commas and convert to int
                                price = int(match.replace(',', ''))
                                # Filter reasonable flight prices (€50-€3000 for international flights)
                                if 50 <= price <= 3000:
                                    flight_prices.append(price)
                            except ValueError:
                                continue
                
                # Remove duplicates and sort
                flight_prices = sorted(list(set(flight_prices)))
                total_time = time.time() - start_time
                
                logger.info(f"✅ Extracted {len(flight_prices)} flight prices in {total_time:.2f}s")
                
                # Calculate statistics
                cheapest_flight = min(flight_prices) if flight_prices else None
                average_price = int(sum(flight_prices) / len(flight_prices)) if flight_prices else None
                
                return {
                    "dohop_url": dohop_url,
                    "flight_prices": flight_prices[:10],  # Limit to first 10 for readability
                    "all_prices_found": flight_prices,  # Full list for debugging
                    "total_prices_found": len(flight_prices),
                    "cheapest_flight_eur": cheapest_flight,
                    "average_price_eur": average_price,
                    "price_range": {
                        "min": min(flight_prices) if flight_prices else None,
                        "max": max(flight_prices) if flight_prices else None
                    },
                    "scrape_successful": True,
                    "page_content_preview": page_content[:500] + "..." if page_content else None,
                    "page_content_full": page_content,  # Full content for debugging
                    "scrape_timestamp": datetime.now().isoformat(),
                    "total_scrape_time": total_time
                }
                
    except Exception as e:
        logger.error(f"❌ Dohop scraping failed: {e}")
        return {
            "dohop_url": dohop_url,
            "flight_prices": [],
            "total_prices_found": 0,
            "cheapest_flight_eur": None,
            "average_price_eur": None,
            "scrape_successful": False,
            "error": str(e)
        }

async def tool_fetch_flight_cost(args: dict) -> dict:
    """
    Fetch flight costs from Dohop using MCP browser automation
    
    Args:
        args: Dictionary validated against FlightLookupArgs schema
        
    Returns:
        dict: {
            "route": "Berlin → Barcelona",
            "departure_date": "2025-07-15",
            "cheapest_flight_eur": 89,
            "average_price_eur": 134,
            "flight_prices": [89, 134, 167, ...],
            "dohop_url": "https://dohop.is/...",
            "status": "success"
        }
    """
    
    # Validate arguments
    try:
        validated_args = FlightLookupArgs(**args)
    except Exception as e:
        raise ValueError(f"Invalid arguments: {e}")
    
    origin = validated_args.origin
    destination = validated_args.destination
    departure_date = validated_args.departure_date
    return_date = validated_args.return_date
    passengers = validated_args.passengers
    
    logger.info(f"🛫 Fetching flights for {origin} → {destination} on {departure_date}")
    
    # Scrape Dohop using browser automation
    scrape_result = await scrape_dohop_flights(
        origin=origin,
        destination=destination, 
        departure_date=departure_date,
        return_date=return_date,
        passengers=passengers
    )
    
    # Prepare response
    result = {
        "route": f"{origin} → {destination}",
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "return_date": return_date,
        "passengers": passengers,
        "trip_type": "round_trip" if return_date else "one_way",
        "search_date": datetime.now().date().isoformat(),
        "last_updated": datetime.now().isoformat(),
        
        # Flight data from scraping
        "dohop_url": scrape_result["dohop_url"],
        "flight_prices": scrape_result["flight_prices"],
        "total_prices_found": scrape_result["total_prices_found"],
        "cheapest_flight_eur": scrape_result["cheapest_flight_eur"],
        "average_price_eur": scrape_result["average_price_eur"],
        "price_range": scrape_result.get("price_range"),
        
        # Status
        "status": "success" if scrape_result["scrape_successful"] else "failed",
        "scrape_successful": scrape_result["scrape_successful"],
        "data_source": "Dohop via BrightData MCP Browser"
    }
    
    # Add debugging data
    for key in ["page_content_full", "all_prices_found", "scrape_timestamp", "total_scrape_time"]:
        if scrape_result.get(key):
            result[key] = scrape_result[key]
    
    if scrape_result.get("error"):
        result["error"] = scrape_result["error"]
    
    logger.info(f"✅ Flight search completed for {origin} → {destination}: " +
                f"€{result['cheapest_flight_eur']} cheapest" if result['cheapest_flight_eur'] else "No prices found")
    
    return result

def main():
    """Test the tool with sample flight searches"""
    try:
        logging.basicConfig(level=logging.INFO)
        
        print("=== Dohop MCP Browser Automation Tool ===")
        print("Clean flight price scraping without car rental noise\n")
        
        # Test case - same as your example
        test_args = {
            "origin": "BER",  # Berlin Brandenburg
            "destination": "KEF",  # Keflavik (Iceland)  
            "departure_date": "2025-07-18",
            "return_date": "2025-07-20",
            "passengers": 1
        }
        
        print(f"Testing: {test_args['origin']} → {test_args['destination']}")
        print(f"Departure: {test_args['departure_date']}")
        print(f"Return: {test_args['return_date']}")
        
        result = asyncio.run(tool_fetch_flight_cost(test_args))
        
        # Save full results to JSON for debugging
        json_filename = f"dohop_results_{test_args['origin']}_{test_args['destination']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Full results saved to: {json_filename}")
        
        print(f"\n✅ Results:")
        print(f"Route: {result['route']}")
        print(f"Dohop URL: {result['dohop_url']}")
        print(f"Status: {result['status']}")
        print(f"Prices Found: {result['total_prices_found']}")
        
        if result['cheapest_flight_eur']:
            print(f"Cheapest Flight: €{result['cheapest_flight_eur']}")
            print(f"Average Price: €{result['average_price_eur']}")
            print(f"Price Range: €{result['price_range']['min']} - €{result['price_range']['max']}")
            
            if result['flight_prices']:
                print(f"\n💰 Sample Prices: {result['flight_prices']}")
            
            # Show all prices for debugging
            if result.get('all_prices_found') and len(result['all_prices_found']) > 10:
                print(f"\n🔍 All {len(result['all_prices_found'])} prices found: {result['all_prices_found']}")
        
        if result.get('error'):
            print(f"\n❌ Error: {result['error']}")
        
        # Show a sample of page content for manual verification
        if result.get('page_content_preview'):
            print(f"\n📄 Page Content Preview (first 500 chars):")
            print(result['page_content_preview'])
            
        print(f"\n🔍 Debug: Check {json_filename} for full page content and extracted prices")
        
        return True
        
    except Exception as e:
        logger.error(f"Tool test failed: {e}")
        raise

if __name__ == "__main__":
    main() 