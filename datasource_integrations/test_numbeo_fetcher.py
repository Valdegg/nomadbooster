#!/usr/bin/env python3
"""
Test script for Numbeo cost index fetcher with BrightData
"""

import asyncio
import logging
import os
from numbeo_cost_index import fetch_cost_index_for_city, tool_fetch_cost_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_single_city():
    """Test fetching cost index for a single city"""
    try:
        print("🧪 Testing single city fetch...")
        
        # Test the direct function
        result = await fetch_cost_index_for_city("Lisbon")
        print(f"✅ Direct function result: {result}")
        
        # Test the MCP tool wrapper
        tool_result = await tool_fetch_cost_index({"city": "Lisbon"})
        print(f"✅ Tool wrapper result: {tool_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_multiple_cities():
    """Test fetching cost index for multiple cities"""
    try:
        print("🧪 Testing multiple cities fetch...")
        
        test_cities = ["Lisbon", "Berlin", "Amsterdam"]
        results = []
        
        for city in test_cities:
            try:
                result = await fetch_cost_index_for_city(city)
                results.append(result)
                print(f"✅ {city}: {result['cost_index']}")
                
                # Add delay to be respectful
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ Failed to fetch {city}: {e}")
                continue
        
        print(f"✅ Successfully fetched {len(results)} out of {len(test_cities)} cities")
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment...")
    
    # Check BrightData configuration
    auth = os.getenv("BRIGHTDATA_AUTH")
    endpoint = os.getenv("BRIGHTDATA_ENDPOINT")
    
    # Try to extract auth from endpoint if needed
    if not auth and endpoint and "@brd.superproxy.io" in endpoint:
        try:
            auth_part = endpoint.split("@")[0].replace("wss://", "")
            if auth_part.startswith("brd-customer-"):
                auth = auth_part
                print(f"📝 Extracted auth from endpoint: {auth[:20]}...")
        except Exception:
            pass
    
    if not auth and not endpoint:
        print("❌ Neither BRIGHTDATA_AUTH nor BRIGHTDATA_ENDPOINT is set")
        print("   Set one of:")
        print("   export BRIGHTDATA_AUTH='brd-customer-xxx-zone-xxx:password'")
        print("   export BRIGHTDATA_ENDPOINT='wss://brd-customer-xxx...'")
        return False
    
    if endpoint and "brd.superproxy.io" not in endpoint:
        print("❌ BRIGHTDATA_ENDPOINT doesn't look like a valid BrightData URL")
        return False
    
    if auth:
        print(f"✅ BRIGHTDATA_AUTH available (length: {len(auth)})")
    if endpoint:
        print(f"✅ BRIGHTDATA_ENDPOINT set (length: {len(endpoint)})")
    
    # Check if playwright is available
    try:
        from playwright.async_api import async_playwright
        print("✅ Playwright import successful")
    except ImportError:
        print("❌ Playwright not installed")
        print("   Install with: pip install playwright && playwright install")
        return False
    
    return True

async def main():
    """Main test function"""
    print("🚀 Starting Numbeo BrightData Integration Test")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please fix the issues above.")
        return
    
    print()
    
    # Test single city
    single_success = await test_single_city()
    print()
    
    # Test multiple cities if single test passed
    if single_success:
        multi_success = await test_multiple_cities()
    else:
        print("⏭️  Skipping multiple cities test due to single city test failure")
        multi_success = False
    
    print()
    print("=" * 50)
    
    if single_success and multi_success:
        print("🎉 All tests passed! Numbeo BrightData integration is working.")
    elif single_success:
        print("⚠️  Single city test passed, but multiple cities test had issues.")
    else:
        print("❌ Tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main()) 