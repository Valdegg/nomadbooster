#!/usr/bin/env python3
"""
Real Bright Data MCP Client Test
Connects to and tests the actual Bright Data MCP server

Usage: python mcp_client_test.py
"""

import asyncio
import json
import os
import subprocess
import sys
from typing import Dict, List, Any, Optional
import signal
import time

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ MCP library not installed. Install with: pip install mcp")
    print("   Or run: pip install -r requirements.txt")
    sys.exit(1)

class BrightDataMCPTester:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.server_process = None
        
    async def start_mcp_server(self):
        """Start the Bright Data MCP server"""
        print("🚀 Starting Bright Data MCP Server...")
        
        # Check if API token is set
        api_token = os.getenv('API_TOKEN') or os.getenv('BRIGHTDATA_API_TOKEN')
        if not api_token:
            # Try to load from .env file
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('BRIGHTDATA_API_KEY=') or line.startswith('BRIGHTDATA_API_TOKEN='):
                            api_token = line.split('=', 1)[1].strip()
                            os.environ['API_TOKEN'] = api_token
                            break
            except FileNotFoundError:
                pass
                
        if not api_token:
            print("❌ No API token found. Please set BRIGHTDATA_API_TOKEN or API_TOKEN")
            print("   Or create a .env file with BRIGHTDATA_API_TOKEN=your_token")
            return False
            
        print(f"✅ Found API token: {api_token[:8]}...")
        
        # Set environment for the MCP server
        env = os.environ.copy()
        env['API_TOKEN'] = api_token
        
        try:
            # Start the MCP server process
            server_params = StdioServerParameters(
                command="npx",
                args=["@brightdata/mcp"],
                env=env
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    print("✅ Connected to Bright Data MCP server")
                    
                    # Initialize the session
                    await session.initialize()
                    print("✅ MCP session initialized")
                    
                    return True
                    
        except Exception as e:
            print(f"❌ Failed to connect to MCP server: {e}")
            return False
    
    async def list_tools(self):
        """List all available tools from the MCP server"""
        if not self.session:
            print("❌ No active MCP session")
            return []
            
        try:
            print("🛠️  Fetching available MCP tools...")
            tools_result = await self.session.list_tools()
            
            print(f"🔍 Raw tools result: {tools_result}")
            print(f"🔍 Tools result type: {type(tools_result)}")
            print(f"🔍 Tools result attributes: {dir(tools_result)}")
            
            tools = tools_result.tools if hasattr(tools_result, 'tools') else []
            
            print(f"📋 Found {len(tools)} available tools:")
            for tool in tools:
                name = tool.name if hasattr(tool, 'name') else str(tool)
                description = tool.description if hasattr(tool, 'description') else "No description"
                print(f"   • {name}: {description}")
                
            return tools
            
        except Exception as e:
            print(f"❌ Error listing tools: {e}")
            print(f"❌ Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    async def test_search_tool(self):
        """Test the search_engine tool"""
        if not self.session:
            print("❌ No active MCP session")
            return False
            
        try:
            print("🔍 Testing search_engine tool...")
            
            # Test search for Skyscanner flights
            search_query = "site:skyscanner.com flights Berlin Barcelona"
            
            print(f"🔍 Calling tool with query: {search_query}")
            result = await self.session.call_tool(
                "search_engine",
                arguments={
                    "query": search_query,
                    "max_results": 5
                }
            )
            
            print(f"🔍 Raw search result: {result}")
            print(f"🔍 Result type: {type(result)}")
            print(f"🔍 Result attributes: {dir(result)}")
            
            print(f"✅ Search successful! Found results for: {search_query}")
            print("📄 Sample results:")
            
            # Parse and display results
            if hasattr(result, 'content'):
                try:
                    content = json.loads(result.content) if isinstance(result.content, str) else result.content
                    for i, item in enumerate(content.get('results', [])[:3]):
                        title = item.get('title', 'No title')
                        url = item.get('url', 'No URL')
                        print(f"   {i+1}. {title}")
                        print(f"      {url}")
                except:
                    print(f"   Raw result: {str(result.content)[:200]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing search tool: {e}")
            print(f"❌ Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_scrape_tool(self):
        """Test the scraping tools"""
        if not self.session:
            print("❌ No active MCP session")
            return False
            
        try:
            print("🌐 Testing scrape_as_markdown tool...")
            
            # Test scraping Skyscanner homepage
            test_url = "https://www.skyscanner.com"
            
            result = await self.session.call_tool(
                "scrape_as_markdown", 
                arguments={
                    "url": test_url
                }
            )
            
            print(f"✅ Scraping successful! Scraped: {test_url}")
            
            if hasattr(result, 'content'):
                content_str = str(result.content)
                print(f"📄 Content length: {len(content_str)} characters")
                print("📄 Sample content:")
                # Show first few lines
                lines = content_str.split('\n')[:5]
                for line in lines:
                    if line.strip():
                        print(f"   {line[:80]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing scrape tool: {e}")
            return False
    
    async def test_browser_automation(self):
        """Test browser automation tools"""
        if not self.session:
            print("❌ No active MCP session")
            return False
            
        try:
            print("🤖 Testing browser automation tools...")
            
            # Test browser navigation
            result = await self.session.call_tool(
                "scraping_browser_navigate",
                arguments={
                    "url": "https://www.skyscanner.com"
                }
            )
            
            print("✅ Browser navigation successful!")
            print(f"📄 Navigation result: {str(result)[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing browser automation: {e}")
            print("   Note: Browser tools may require BROWSER_ZONE configuration")
            return False

async def main():
    """Main test function"""
    print("🚀 Bright Data MCP Real Client Test")
    print("=" * 50)
    
    tester = BrightDataMCPTester()
    
    try:
        # Start MCP server connection
        if not await tester.start_mcp_server():
            print("❌ Failed to start MCP server")
            return False
        
        print()
        
        # List available tools
        tools = await tester.list_tools()
        print()
        
        # Test search tool
        await tester.test_search_tool()
        print()
        
        # Test scraping tool
        await tester.test_scrape_tool()
        print()
        
        # Test browser automation
        await tester.test_browser_automation()
        print()
        
        print("✅ MCP testing completed!")
        print("\n🎯 Next steps:")
        print("   1. Integrate successful tools into travel recommendation system")
        print("   2. Implement Skyscanner flight search workflow")
        print("   3. Add error handling and retry logic")
        print("   4. Test with different search parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during MCP testing: {e}")
        return False
    finally:
        # Cleanup
        if tester.server_process:
            tester.server_process.terminate()

if __name__ == "__main__":
    asyncio.run(main()) 