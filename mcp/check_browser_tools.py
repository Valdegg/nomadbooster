#!/usr/bin/env python3
"""
Quick check to see what browser automation tools are available in Bright Data MCP
"""

import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def check_browser_tools():
    """Check what browser automation tools are available"""
    
    # Load ALL environment variables from .env file
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        print("❌ No .env file found")
        return
    
    # Get API token
    api_token = env_vars.get('API_TOKEN') or env_vars.get('BRIGHTDATA_API_TOKEN') or env_vars.get('BRIGHTDATA_API_KEY')
    if not api_token:
        print("❌ No API token found")
        return
    
    # Set environment with ALL variables from .env
    env = os.environ.copy()
    env.update(env_vars)  # Load ALL .env variables
    env['API_TOKEN'] = api_token  # Ensure API_TOKEN is set
    
    # Debug: Print if BROWSER_AUTH is loaded
    if 'BROWSER_AUTH' in env:
        print(f"✅ BROWSER_AUTH loaded: {env['BROWSER_AUTH'][:30]}...")
    else:
        print("❌ BROWSER_AUTH not found in environment")
    
    server_params = StdioServerParameters(
        command="npx",
        args=["@brightdata/mcp"],
        env=env
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List all tools
            tools_result = await session.list_tools()
            tools = tools_result.tools if hasattr(tools_result, 'tools') else []
            
            print(f"🛠️  Found {len(tools)} total tools")
            
            # Filter for browser-related tools
            browser_tools = [tool for tool in tools if 'browser' in tool.name.lower() or 'scraping' in tool.name.lower()]
            
            print(f"\n🌐 Browser/Scraping Tools ({len(browser_tools)}):")
            for tool in browser_tools:
                print(f"   • {tool.name}: {tool.description}")
            
            # Check if browser activation is needed
            if any('activation' in tool.name.lower() for tool in browser_tools):
                print(f"\n🔍 Checking browser activation instructions...")
                try:
                    activation_result = await session.call_tool(
                        "scraping_browser_activation_instructions",
                        arguments={}
                    )
                    print(f"📋 Activation instructions:")
                    if hasattr(activation_result, 'content'):
                        content = activation_result.content[0].text if hasattr(activation_result.content, '__iter__') else str(activation_result.content)
                        print(content[:500] + "..." if len(content) > 500 else content)
                except Exception as e:
                    print(f"❌ Failed to get activation instructions: {e}")

if __name__ == "__main__":
    asyncio.run(check_browser_tools()) 