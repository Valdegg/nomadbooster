#!/usr/bin/env python3
"""
Simple MCP Test - Debug Version
"""

import asyncio
import os
import json

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ MCP library not installed. Run: pip install mcp")
    exit(1)

async def simple_test():
    """Simple test to debug MCP connection"""
    print("🚀 Simple MCP Connection Test")
    print("=" * 40)
    
    # Load API token
    api_token = None
    try:
        with open('.env', 'r') as f:
            for line in f:
                if 'API_KEY=' in line or 'API_TOKEN=' in line:
                    api_token = line.split('=', 1)[1].strip()
                    break
    except FileNotFoundError:
        print("❌ No .env file found")
        return
    
    if not api_token:
        print("❌ No API token found in .env")
        return
        
    print(f"✅ Found API token: {api_token[:8]}...")
    
    # Set environment
    env = os.environ.copy()
    env['API_TOKEN'] = api_token
    
    try:
        server_params = StdioServerParameters(
            command="npx",
            args=["@brightdata/mcp"],
            env=env
        )
        
        print("🔌 Connecting to MCP server...")
        async with stdio_client(server_params) as (read, write):
            print("✅ STDIO connection established")
            
            async with ClientSession(read, write) as session:
                print("✅ Client session created")
                
                # Initialize
                await session.initialize()
                print("✅ Session initialized")
                
                # Test 1: List tools
                print("\n🛠️  Testing list_tools()...")
                try:
                    tools_result = await session.list_tools()
                    print(f"📋 Tools result: {tools_result}")
                    
                    if hasattr(tools_result, 'tools'):
                        tools = tools_result.tools
                        print(f"📋 Found {len(tools)} tools:")
                        for tool in tools:
                            print(f"   • {tool.name}")
                    else:
                        print(f"📋 No 'tools' attribute. Available: {dir(tools_result)}")
                        
                except Exception as e:
                    print(f"❌ list_tools failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Test 2: Try a simple tool call
                print("\n🔍 Testing a simple tool call...")
                try:
                    # Try to call search_engine
                    result = await session.call_tool(
                        "search_engine",
                        arguments={"query": "test", "max_results": 1}
                    )
                    print(f"✅ Tool call successful: {result}")
                    
                except Exception as e:
                    print(f"❌ Tool call failed: {e}")
                    import traceback
                    traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_test()) 