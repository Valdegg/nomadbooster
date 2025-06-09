#!/usr/bin/env python3
"""
Test script for detailed Numbeo cost data extraction
"""

import asyncio
import json
from numbeo_cost_index import fetch_cost_index_for_city

async def test_detailed_extraction():
    """Test the new detailed cost extraction"""
    try:
        print("🧪 Testing detailed cost extraction for Lisbon...")
        
        result = await fetch_cost_index_for_city("Lisbon")
        
        print(f"✅ Successfully fetched data for {result['city']}")
        print(f"📊 Total items found: {result['total_items']}")
        print(f"💰 Calculated cost index: {result['cost_index']}")
        
        print("\n📋 Sample of detailed costs:")
        for i, (item, data) in enumerate(result['detailed_costs'].items()):
            if i >= 10:  # Show first 10 items
                break
            print(f"   {item}: {data['price']} {data['currency']} (Category: {data.get('category', 'N/A')})")
        
        print(f"\n... and {result['total_items'] - 10} more items")
        
        # Save sample to file for inspection
        with open('sample_detailed_costs.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print("💾 Saved sample data to 'sample_detailed_costs.json'")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_detailed_extraction()) 