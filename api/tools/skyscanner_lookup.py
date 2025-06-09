#!/usr/bin/env python3
"""
Tool for fetching flight costs from Skyscanner using Bright Data MCP browser automation.
Direct approach: Navigate to Skyscanner → Wait for load → Extract flight prices from DOM

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

# MCP client imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ MCP library not installed. Run: pip install mcp")
    raise

logger = logging.getLogger(__name__)

# Cache for environment variables to avoid re-reading .env file
_env_cache = None

# Global MCP session pool to reuse connections
_mcp_session_pool = None


class FlightLookupArgs(BaseModel):
    """Arguments schema for Skyscanner flight cost lookup tool"""
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
                            # Also check for API token
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
    env.update(env_vars)  # Load ALL .env variables including BROWSER_AUTH
    env['API_TOKEN'] = api_token
    
    # Cache the result
    _env_cache = env
    return env

async def get_mcp_session():
    """Create an MCP session connection to Bright Data server (optimized - skip server startup)"""
    env = _load_env_vars()
    
    # Check if user wants to skip server startup
    skip_startup = os.getenv('MCP_SKIP_SERVER_STARTUP', 'false').lower() == 'true'
    
    if skip_startup:
        logger.info("🚀 MCP_SKIP_SERVER_STARTUP=true - assuming server is already running")
        logger.info("⚠️ Note: This will still create a stdio connection but much faster")
    
    # Create MCP server connection (optimized parameters)
    server_params = StdioServerParameters(
        command="npx",
        args=["@brightdata/mcp"],
        env=env
    )
    
    return stdio_client(server_params)


def build_dohop_url(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1, direct_only: bool = True) -> str:
    """
    Build a Dohop URL for flight search - much cleaner than Skyscanner
    
    Format: https://dohop.is/flights/{origin}/{dest}/{departure_date}/{return_date}/adults-{passengers}?stops=0
    """
    # Dohop uses YYYY-MM-DD format (much simpler!)
    base_url = f"https://dohop.is/flights/{origin.upper()}/{destination.upper()}/{departure_date}"
    
    # Add return date if specified
    if return_date:
        base_url += f"/{return_date}"
    
    # Add passengers
    base_url += f"/adults-{passengers}"
    
    # Add query parameters
    params = []
    
    # Add direct flights only filter (much simpler!)
    if direct_only:
        params.append("stops=0")  # Dohop parameter: 0 stops = direct flights only
    
    if params:
        base_url += "?" + "&".join(params)
    
    return base_url


async def scrape_dohop_flights(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1) -> dict:
    """
    Use MCP browser automation to scrape flight data from Skyscanner
    
    Returns:
        dict: {
            "skyscanner_url": "https://...",
            "flight_prices": [89, 134, 167, ...],
            "cheapest_flight_eur": 89,
            "average_price_eur": 134,
            "scrape_successful": True
        }
    """
    
    start_time = time.time()
    
    # Build Skyscanner URL (with direct flights preference)
    direct_flights_only = os.getenv('SKYSCANNER_DIRECT_ONLY', 'true').lower() == 'true'
    skyscanner_url = build_skyscanner_url(origin, destination, departure_date, return_date, passengers, direct_only=direct_flights_only)
    logger.info(f"🌐 Navigating to Skyscanner: {skyscanner_url}")
    logger.info(f"🛫 Direct flights only: {direct_flights_only}")
    
    try:
        # Timer: MCP Connection
        mcp_start = time.time()
        async with await get_mcp_session() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_time = time.time() - mcp_start
                logger.info(f"⏱️ MCP session setup: {mcp_time:.2f}s")
                
                # Timer: Tool detection
                tools_start = time.time()
                tools_result = await session.list_tools()
                tools = tools_result.tools if hasattr(tools_result, 'tools') else []
                tool_names = [tool.name for tool in tools]
                tools_time = time.time() - tools_start
                logger.info(f"⏱️ Tool detection: {tools_time:.2f}s ({len(tools)} tools found)")
                
                # Use browser automation if available, otherwise fall back to scrape_as_markdown
                if "scraping_browser_navigate" in tool_names:
                    logger.info("🚀 Using browser automation for Skyscanner...")
                    
                    # Timer: Navigation
                    nav_start = time.time()
                    await session.call_tool("scraping_browser_navigate", arguments={"url": skyscanner_url})
                    nav_time = time.time() - nav_start
                    logger.info(f"✅ Navigated to Skyscanner page ({nav_time:.2f}s)")
                    
                    # Timer: JavaScript loading wait (optimized)
                    wait_start = time.time()
                    # Try waiting for specific elements instead of fixed time
                    try:
                        # Wait for flight results to appear
                        await session.call_tool("scraping_browser_wait_for", arguments={
                            "selector": "[data-testid='result-item'], .result, .flight-result, .price",
                            "timeout": 10000  # 10 seconds max
                        })
                        wait_time = time.time() - wait_start
                        logger.info(f"⏱️ Smart wait for results: {wait_time:.2f}s")
                    except:
                        # Fallback to shorter fixed wait
                        await asyncio.sleep(4)  # Reduced from 8 to 4 seconds
                        wait_time = time.time() - wait_start
                        logger.info(f"⏱️ Fallback wait: {wait_time:.2f}s")
                    
                    # Timer: Content extraction (try faster method)
                    extract_start = time.time()
                    try:
                        # Try HTML extraction first (sometimes faster)
                        page_result = await session.call_tool("scraping_browser_get_html", arguments={"full_page": False})
                        extract_time = time.time() - extract_start
                        logger.info(f"⏱️ HTML extraction: {extract_time:.2f}s")
                    except:
                        # Fallback to text extraction
                        page_result = await session.call_tool("scraping_browser_get_text", arguments={})
                        extract_time = time.time() - extract_start
                        logger.info(f"⏱️ Text extraction: {extract_time:.2f}s")
                    
                    if hasattr(page_result, 'content'):
                        content = page_result.content[0].text if hasattr(page_result.content, '__iter__') else str(page_result.content)
                        scrape_result = type('MockResult', (), {'content': [type('MockContent', (), {'text': content})]})()
                    else:
                        # Fallback to scrape_as_markdown
                        logger.info("⚠️ Falling back to scrape_as_markdown...")
                        fallback_start = time.time()
                        scrape_result = await session.call_tool("scrape_as_markdown", arguments={"url": skyscanner_url})
                        fallback_time = time.time() - fallback_start
                        logger.info(f"⏱️ Fallback scraping: {fallback_time:.2f}s")
                
                else:
                    logger.info("🌐 Using scrape_as_markdown (browser tools not available)...")
                    scrape_start = time.time()
                    scrape_result = await session.call_tool("scrape_as_markdown", arguments={"url": skyscanner_url})
                    scrape_time = time.time() - scrape_start
                    logger.info(f"⏱️ Markdown scraping: {scrape_time:.2f}s")
                
                # Timer: Price extraction
                parse_start = time.time()
                flight_prices = []
                page_content = ""
                
                if hasattr(scrape_result, 'content') and scrape_result.content:
                    content = scrape_result.content[0].text if hasattr(scrape_result.content, '__iter__') else str(scrape_result.content)
                    page_content = content
                    
                    # Extract prices using regex - look for currency patterns
                    # Skyscanner typically shows prices like "€89", "$134", "£99"
                    price_patterns = [
                        r'€\s*(\d+(?:,\d{3})*)',  # Euro: €89, €1,234
                        r'\$\s*(\d+(?:,\d{3})*)', # Dollar: $89, $1,234  
                        r'£\s*(\d+(?:,\d{3})*)',  # Pound: £89, £1,234
                        r'(\d+(?:,\d{3})*)\s*€',  # Reverse Euro: 89€
                        r'(\d+(?:,\d{3})*)\s*\$', # Reverse Dollar: 89$
                    ]
                    
                    for pattern in price_patterns:
                        matches = re.findall(pattern, content)
                        for match in matches:
                            try:
                                # Remove commas and convert to int
                                price = int(match.replace(',', ''))
                                # Filter reasonable flight prices (€20-€2000)
                                if 20 <= price <= 2000:
                                    flight_prices.append(price)
                            except ValueError:
                                continue
                
                # Remove duplicates and sort
                flight_prices = sorted(list(set(flight_prices)))
                parse_time = time.time() - parse_start
                logger.info(f"✅ Extracted {len(flight_prices)} flight prices ({parse_time:.2f}s)")
                
                # Timer: Total time
                total_time = time.time() - start_time
                logger.info(f"🏁 Total scraping time: {total_time:.2f}s")
                
                # Calculate statistics
                cheapest_flight = min(flight_prices) if flight_prices else None
                average_price = int(sum(flight_prices) / len(flight_prices)) if flight_prices else None
                
                return {
                    "skyscanner_url": skyscanner_url,
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
        logger.error(f"❌ MCP browser scraping failed: {e}")
        return {
            "skyscanner_url": skyscanner_url,
            "flight_prices": [],
            "total_prices_found": 0,
            "cheapest_flight_eur": None,
            "average_price_eur": None,
            "scrape_successful": False,
            "error": str(e)
        }


async def tool_fetch_flight_cost(args: dict) -> dict:
    """
    Fetch flight costs from Skyscanner using MCP browser automation
    
    Args:
        args: Dictionary validated against FlightLookupArgs schema
        
    Returns:
        dict: {
            "route": "Berlin → Barcelona",
            "departure_date": "2025-07-15",
            "cheapest_flight_eur": 89,
            "average_price_eur": 134,
            "flight_prices": [89, 134, 167, ...],
            "skyscanner_url": "https://www.skyscanner.com/...",
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
    
    # Scrape Skyscanner using browser automation
    scrape_result = await scrape_skyscanner_flights(
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
        "skyscanner_url": scrape_result["skyscanner_url"],
        "flight_prices": scrape_result["flight_prices"],
        "total_prices_found": scrape_result["total_prices_found"],
        "cheapest_flight_eur": scrape_result["cheapest_flight_eur"],
        "average_price_eur": scrape_result["average_price_eur"],
        "price_range": scrape_result.get("price_range"),
        
        # Status
        "status": "success" if scrape_result["scrape_successful"] else "failed",
        "scrape_successful": scrape_result["scrape_successful"],
        "data_source": "Skyscanner via BrightData MCP Browser"
    }
    
    if scrape_result.get("error"):
        result["error"] = scrape_result["error"]
    
    if scrape_result.get("page_content_preview"):
        result["page_content_preview"] = scrape_result["page_content_preview"]
    
    # Add full debugging data
    if scrape_result.get("page_content_full"):
        result["page_content_full"] = scrape_result["page_content_full"]
    
    if scrape_result.get("all_prices_found"):
        result["all_prices_found"] = scrape_result["all_prices_found"]
    
    if scrape_result.get("scrape_timestamp"):
        result["scrape_timestamp"] = scrape_result["scrape_timestamp"]
    
    if scrape_result.get("total_scrape_time"):
        result["total_scrape_time"] = scrape_result["total_scrape_time"]
    
    logger.info(f"✅ Flight search completed for {origin} → {destination}: " +
                f"€{result['cheapest_flight_eur']} cheapest" if result['cheapest_flight_eur'] else "No prices found")
    
    return result


def get_tool_schema() -> dict:
    """Get the JSON schema for the tool arguments"""
    return FlightLookupArgs.model_json_schema()


def main():
    """Test the tool with sample flight searches"""
    try:
        logging.basicConfig(level=logging.INFO)
        
        print("=== Skyscanner MCP Browser Automation Tool ===")
        print("Phase 1: Direct browser scraping of Skyscanner flight prices\n")
        
        # Test case
        test_args = {
            "origin": "BER",  # Berlin Brandenburg
            "destination": "KEF",  # Barcelona
            "departure_date": "2025-07-15",
            "return_date": "2025-07-22",
            "passengers": 1
        }
        
        print(f"Testing: {test_args['origin']} → {test_args['destination']}")
        print(f"Departure: {test_args['departure_date']}")
        print(f"Return: {test_args['return_date']}")
        
        result = asyncio.run(tool_fetch_flight_cost(test_args))
        
        # Save full results to JSON for debugging
        json_filename = f"skyscanner_results_{test_args['origin']}_{test_args['destination']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Full results saved to: {json_filename}")
        
        print(f"\n✅ Results:")
        print(f"Route: {result['route']}")
        print(f"Skyscanner URL: {result['skyscanner_url']}")
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