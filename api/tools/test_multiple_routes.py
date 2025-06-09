#!/usr/bin/env python3
"""
Comprehensive Dohop Flight Scraper Test
Test multiple routes and dates for the latter half of 2025
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict

# Add current directory to path to import the scraper
sys.path.append(os.path.dirname(__file__))

from brightdata_dohop_scraper import fetch_dohop_flights_brightdata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test routes for European nomads
TEST_ROUTES = [
    # Germany routes
    {"origin": "BER", "destination": "KEF", "route_name": "Berlin → Reykjavik"},
    {"origin": "MUC", "destination": "LIS", "route_name": "Munich → Lisbon"},
    {"origin": "FRA", "destination": "BCN", "route_name": "Frankfurt → Barcelona"},
    
    # Netherlands routes  
    {"origin": "AMS", "destination": "PRG", "route_name": "Amsterdam → Prague"},
    {"origin": "AMS", "destination": "VIE", "route_name": "Amsterdam → Vienna"},
    
    # UK routes
    {"origin": "LHR", "destination": "DUB", "route_name": "London → Dublin"},
    {"origin": "LGW", "destination": "FCO", "route_name": "London Gatwick → Rome"},
    
    # Eastern Europe routes
    {"origin": "WAW", "destination": "ATH", "route_name": "Warsaw → Athens"},
    {"origin": "BUD", "destination": "ZRH", "route_name": "Budapest → Zurich"},
    
    # Scandinavia routes
    {"origin": "CPH", "destination": "OSL", "route_name": "Copenhagen → Oslo"},
    {"origin": "ARN", "destination": "HEL", "route_name": "Stockholm → Helsinki"},
]

# Test dates in latter half of 2025
def generate_test_dates():
    """Generate test dates for latter half of 2025"""
    test_dates = []
    
    # August 2025 - summer travel
    test_dates.append({
        "departure_date": "2025-08-15",
        "return_date": "2025-08-22", 
        "season": "Summer",
        "trip_length": "7 days"
    })
    
    # September 2025 - shoulder season
    test_dates.append({
        "departure_date": "2025-09-10",
        "return_date": "2025-09-17",
        "season": "Shoulder",
        "trip_length": "7 days"
    })
    
    # October 2025 - autumn
    test_dates.append({
        "departure_date": "2025-10-05",
        "return_date": "2025-10-12",
        "season": "Autumn",
        "trip_length": "7 days"
    })
    
    # November 2025 - off-season
    test_dates.append({
        "departure_date": "2025-11-20",
        "return_date": "2025-11-27",
        "season": "Off-season",
        "trip_length": "7 days"
    })
    
    # December 2025 - holiday season
    test_dates.append({
        "departure_date": "2025-12-15",
        "return_date": "2025-12-22",
        "season": "Holiday",
        "trip_length": "7 days"
    })
    
    return test_dates

async def test_single_route(route: Dict, date_info: Dict, passengers: int = 1) -> Dict:
    """Test a single route with specific dates"""
    
    logger.info(f"\n🛫 Testing: {route['route_name']}")
    logger.info(f"📅 Dates: {date_info['departure_date']} to {date_info['return_date']} ({date_info['season']})")
    logger.info(f"👥 Passengers: {passengers}")
    
    try:
        result = await fetch_dohop_flights_brightdata(
            origin=route['origin'],
            destination=route['destination'],
            departure_date=date_info['departure_date'],
            return_date=date_info['return_date'],
            passengers=passengers
        )
        
        # Add test metadata
        result['test_metadata'] = {
            'route_name': route['route_name'],
            'season': date_info['season'],
            'trip_length': date_info['trip_length'],
            'test_timestamp': datetime.now().isoformat()
        }
        
        # Log results
        if result['success']:
            stats = result['statistics']
            logger.info(f"✅ Success: {stats['total_flights']} flights found")
            if stats['cheapest_flight_eur']:
                logger.info(f"💰 Best price: €{stats['cheapest_flight_eur']}")
                logger.info(f"📊 Price range: €{stats['price_range']['min']} - €{stats['price_range']['max']}")
                
                # Log flight details
                for i, flight in enumerate(result['flights'][:3]):  # Show top 3
                    logger.info(f"  {i+1}. €{flight['price_eur']} - {flight['airline']} dep: {flight['departure_time']}")
        else:
            logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Exception testing {route['route_name']}: {e}")
        return {
            'success': False,
            'route': route['route_name'],
            'error': str(e),
            'test_metadata': {
                'route_name': route['route_name'], 
                'season': date_info['season'],
                'trip_length': date_info['trip_length'],
                'test_timestamp': datetime.now().isoformat()
            }
        }

async def run_comprehensive_test():
    """Run comprehensive test of multiple routes and dates"""
    
    print("=" * 60)
    print("🌍 COMPREHENSIVE DOHOP FLIGHT SCRAPER TEST")
    print("🗓️  Testing multiple routes across latter half of 2025")
    print("=" * 60)
    
    # Check BrightData configuration
    from brightdata_dohop_scraper import BR_ENDPOINT
    if not BR_ENDPOINT:
        print("❌ BrightData configuration missing!")
        print("Set BRIGHTDATA_AUTH or BRIGHTDATA_ENDPOINT environment variable")
        return
    
    print(f"🔗 BrightData endpoint: {BR_ENDPOINT[:50]}...")
    
    test_dates = generate_test_dates()
    all_results = []
    
    # Test subset of routes with different dates
    priority_routes = TEST_ROUTES[:5]  # Test first 5 routes
    
    total_tests = len(priority_routes) * len(test_dates)
    current_test = 0
    
    print(f"\n🧪 Running {total_tests} total tests...")
    print(f"📍 Routes: {len(priority_routes)}")
    print(f"📅 Date ranges: {len(test_dates)}")
    
    for route in priority_routes:
        for date_info in test_dates:
            current_test += 1
            
            print(f"\n📊 Test {current_test}/{total_tests}")
            print("-" * 40)
            
            result = await test_single_route(route, date_info)
            all_results.append(result)
            
            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
            temp_file = f"temp_results_{current_test:02d}_{timestamp}.json"
            with open(temp_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Add delay between tests to be respectful
            await asyncio.sleep(30)  # 30 second delay between tests
    
    # Save final comprehensive results
    final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_flight_test_results_{final_timestamp}.json"
    
    comprehensive_results = {
        'test_summary': {
            'total_tests': total_tests,
            'successful_tests': len([r for r in all_results if r['success']]),
            'failed_tests': len([r for r in all_results if not r['success']]),
            'test_timestamp': datetime.now().isoformat(),
            'routes_tested': len(priority_routes),
            'date_ranges_tested': len(test_dates)
        },
        'results': all_results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
    
    # Generate summary report
    print(f"\n🏁 COMPREHENSIVE TEST COMPLETE")
    print("=" * 60)
    print(f"📄 Results saved to: {results_file}")
    print(f"✅ Successful tests: {comprehensive_results['test_summary']['successful_tests']}")
    print(f"❌ Failed tests: {comprehensive_results['test_summary']['failed_tests']}")
    
    # Price analysis
    successful_results = [r for r in all_results if r['success'] and r.get('statistics', {}).get('cheapest_flight_eur')]
    
    if successful_results:
        all_prices = [r['statistics']['cheapest_flight_eur'] for r in successful_results]
        
        print(f"\n💰 PRICE ANALYSIS")
        print(f"Cheapest flight: €{min(all_prices)}")
        print(f"Most expensive: €{max(all_prices)}")
        print(f"Average price: €{int(sum(all_prices) / len(all_prices))}")
        
        # Price by season
        season_prices = {}
        for result in successful_results:
            season = result['test_metadata']['season']
            price = result['statistics']['cheapest_flight_eur']
            if season not in season_prices:
                season_prices[season] = []
            season_prices[season].append(price)
        
        print(f"\n📊 AVERAGE PRICES BY SEASON:")
        for season, prices in season_prices.items():
            avg_price = int(sum(prices) / len(prices))
            print(f"  {season}: €{avg_price} (from {len(prices)} flights)")
        
        # Top routes by price
        route_prices = {}
        for result in successful_results:
            route = result['test_metadata']['route_name'] 
            price = result['statistics']['cheapest_flight_eur']
            if route not in route_prices:
                route_prices[route] = []
            route_prices[route].append(price)
        
        print(f"\n🛫 CHEAPEST ROUTES:")
        sorted_routes = sorted(route_prices.items(), key=lambda x: min(x[1]))
        for route, prices in sorted_routes[:5]:
            min_price = min(prices)
            avg_price = int(sum(prices) / len(prices))
            print(f"  {route}: from €{min_price} (avg €{avg_price})")
    
    print(f"\n🎯 Test completed successfully!")
    print(f"Check {results_file} for detailed results")
    
    return comprehensive_results

if __name__ == "__main__":
    # Run the comprehensive test
    results = asyncio.run(run_comprehensive_test()) 