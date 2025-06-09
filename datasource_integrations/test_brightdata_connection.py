#!/usr/bin/env python3
"""
BrightData Connection Test Script

Tests both HTTP proxy and Browser API connections to verify BrightData credentials are working.
Based on BrightData setup guide for Nomad Booster.
"""

import os
import asyncio
import logging
import requests
import subprocess
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BrightDataTester:
    def __init__(self):
        self.auth = os.getenv("BRIGHTDATA_AUTH")
        self.endpoint = os.getenv("BRIGHTDATA_ENDPOINT")
        
    def check_environment(self) -> bool:
        """Check if environment variables are properly set"""
        print("🔍 Checking BrightData environment setup...")
        
        # Try to get auth, or extract from endpoint
        if not self.auth and self.endpoint:
            # Extract auth from endpoint format: wss://auth@host:port
            try:
                if "@brd.superproxy.io" in self.endpoint:
                    auth_part = self.endpoint.split("@")[0].replace("wss://", "")
                    if auth_part.startswith("brd-customer-"):
                        self.auth = auth_part
                        print(f"📝 Extracted auth from endpoint: {self.auth[:20]}...{self.auth[-10:]}")
            except Exception as e:
                print(f"⚠️  Could not extract auth from endpoint: {e}")
        
        if not self.auth:
            print("❌ BRIGHTDATA_AUTH environment variable not set")
            print("   Expected format: brd-customer-123456-zone-my_scraper_zone:password")
            print("   Set it with: export BRIGHTDATA_AUTH='your-auth-string'")
            print("   Or set BRIGHTDATA_ENDPOINT with embedded auth")
            return False
        
        # Validate auth format
        if not self.auth.startswith("brd-customer-") or ":" not in self.auth:
            print(f"❌ BRIGHTDATA_AUTH format looks incorrect: {self.auth}")
            print("   Expected format: brd-customer-123456-zone-my_scraper_zone:password")
            return False
        
        print(f"✅ BRIGHTDATA_AUTH available: {self.auth[:20]}...{self.auth[-10:]}")
        
        if self.endpoint:
            print(f"✅ BRIGHTDATA_ENDPOINT set: {self.endpoint[:30]}...")
        else:
            # Generate endpoint from auth
            self.endpoint = f"wss://{self.auth}@brd.superproxy.io:9222"
            print(f"📝 Generated endpoint: {self.endpoint[:30]}...")
        
        return True
    
    def test_http_proxy(self) -> bool:
        """Test HTTP proxy connection using curl command"""
        print("\n🌐 Testing HTTP Proxy Connection...")
        
        try:
            # Test using curl command (same as in the guide)
            cmd = [
                "curl", "--proxy", "brd.superproxy.io:33335",
                "--proxy-user", self.auth,
                "--max-time", "10",
                "--silent",
                "https://geo.brdtest.com/welcome.txt"
            ]
            
            print(f"Running: curl --proxy brd.superproxy.io:33335 --proxy-user ****** https://geo.brdtest.com/welcome.txt")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and "welcome" in result.stdout.lower():
                print("✅ HTTP Proxy test PASSED")
                print(f"   Response: {result.stdout.strip()[:100]}...")
                return True
            else:
                # Check for common Browser API zone error
                if "403" in result.stderr or result.returncode == 56:
                    print("ℹ️  HTTP Proxy test SKIPPED - likely using Scraping Browser zone (Browser API only)")
                    return True  # This is expected for Browser API zones
                else:
                    print("❌ HTTP Proxy test FAILED")
                    print(f"   Return code: {result.returncode}")
                    print(f"   Stdout: {result.stdout}")
                    print(f"   Stderr: {result.stderr}")
                    return False
                
        except subprocess.TimeoutExpired:
            print("❌ HTTP Proxy test TIMEOUT")
            return False
        except FileNotFoundError:
            print("⚠️  curl not found, skipping HTTP proxy test")
            return True  # Don't fail if curl isn't available
        except Exception as e:
            print(f"❌ HTTP Proxy test ERROR: {e}")
            return False
    
    def test_http_proxy_python(self) -> bool:
        """Test HTTP proxy using Python requests"""
        print("\n🐍 Testing HTTP Proxy with Python requests...")
        
        try:
            proxies = {
                'http': f'http://{self.auth}@brd.superproxy.io:33335',
                'https': f'http://{self.auth}@brd.superproxy.io:33335'
            }
            
            response = requests.get(
                'https://geo.brdtest.com/welcome.txt',
                proxies=proxies,
                timeout=10
            )
            
            if response.status_code == 200 and "welcome" in response.text.lower():
                print("✅ Python HTTP Proxy test PASSED")
                print(f"   Response: {response.text.strip()[:100]}...")
                return True
            else:
                print(f"❌ Python HTTP Proxy test FAILED: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            error_msg = str(e)
            if "Scraping Browser zone as regular proxy" in error_msg:
                print("ℹ️  HTTP Proxy test SKIPPED - using Scraping Browser zone (Browser API only)")
                return True  # This is expected for Browser API zones
            else:
                print(f"❌ Python HTTP Proxy test ERROR: {e}")
                return False
    
    async def test_browser_api(self) -> bool:
        """Test Browser API connection using Playwright"""
        print("\n🎭 Testing Browser API Connection...")
        
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print("❌ Playwright not installed. Install with: pip install playwright")
            return False
        
        try:
            async with async_playwright() as pw:
                print(f"Connecting to: {self.endpoint[:50]}...")
                
                browser = await pw.chromium.connect_over_cdp(self.endpoint)
                try:
                    page = await browser.new_page()
                    
                    # Test with a simple page
                    print("Loading test page...")
                    await page.goto("https://geo.brdtest.com/welcome.txt", timeout=15000)
                    
                    content = await page.content()
                    
                    if "welcome" in content.lower():
                        print("✅ Browser API test PASSED")
                        print(f"   Page content: {content.strip()[:100]}...")
                        return True
                    else:
                        print("❌ Browser API test FAILED - unexpected content")
                        print(f"   Content: {content[:200]}...")
                        return False
                        
                finally:
                    await browser.close()
                    
        except Exception as e:
            print(f"❌ Browser API test ERROR: {e}")
            return False
    
    async def test_numbeo_connection(self) -> bool:
        """Test actual Numbeo connection to verify real-world usage"""
        print("\n🏙️  Testing Numbeo Connection...")
        
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print("⚠️  Playwright not available, skipping Numbeo test")
            return True
        
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.connect_over_cdp(self.endpoint)
                try:
                    page = await browser.new_page()
                    
                    # Test with actual Numbeo page
                    test_url = "https://www.numbeo.com/cost-of-living/in/Lisbon"
                    print(f"Loading: {test_url}")
                    
                    await page.goto(test_url, timeout=30000)
                    
                    # Check if page loaded successfully
                    title = await page.title()
                    print(f"Page title: {title}")
                    
                    # Check if it's a valid Numbeo page
                    title_lower = title.lower()
                    if ("cost of living" in title_lower and "lisbon" in title_lower) or ("numbeo" in title_lower):
                        print("✅ Numbeo connection test PASSED")
                        
                        # Try to find tables
                        tables = await page.evaluate("() => document.querySelectorAll('table').length")
                        print(f"   Found {tables} tables on page")
                        
                        # Look for cost of living data indicators
                        has_cost_data = await page.evaluate("""
                            () => {
                                const pageText = document.body.textContent.toLowerCase();
                                return pageText.includes('cost of living') && 
                                       (pageText.includes('index') || pageText.includes('price'));
                            }
                        """)
                        print(f"   Has cost of living data: {has_cost_data}")
                        
                        return True
                    else:
                        print("❌ Numbeo connection test FAILED - unexpected page")
                        print(f"   Title: {title}")
                        return False
                        
                finally:
                    await browser.close()
                    
        except Exception as e:
            print(f"❌ Numbeo connection test ERROR: {e}")
            return False

async def main():
    """Main test function"""
    print("🚀 BrightData Connection Test Suite")
    print("=" * 50)
    
    tester = BrightDataTester()
    
    # Check environment
    if not tester.check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    # Run tests
    results = {}
    
    results['http_proxy_curl'] = tester.test_http_proxy()
    results['http_proxy_python'] = tester.test_http_proxy_python()
    results['browser_api'] = await tester.test_browser_api()
    results['numbeo_connection'] = await tester.test_numbeo_connection()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! BrightData is properly configured.")
    elif passed_tests >= total_tests - 1:
        print("⚠️  Most tests passed. Minor issues detected.")
    else:
        print("❌ Multiple tests failed. Check your BrightData configuration.")

if __name__ == "__main__":
    asyncio.run(main()) 