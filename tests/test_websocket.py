#!/usr/bin/env python3
"""
Simple WebSocket test client for Nomad Booster API
Usage: python test_websocket.py [message]
"""

import asyncio
import websockets
import json
import sys
from datetime import datetime

async def test_chat(message="I want to find a place that's not too cold and not too warm."):
    """Test the chat WebSocket endpoint with follow-up messages"""
    
    client_id = "test_client_123"
    uri = f"ws://localhost:8000/ws/{client_id}"
    
    print(f"🔗 Connecting to {uri}")
    print(f"💬 Will send: {message}")
    print("-" * 50)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")
            
            # Send the initial test message
            test_message = {
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send(json.dumps(test_message))
            print(f"📤 Sent: {message}")
            print("-" * 50)
            print("🤖 Bot response:")
            
            # Listen for initial response
            await listen_for_response(websocket, "Initial")
            
            # Send follow-up message with specific temperatures
            followup_message = {
                "content": "I'd say between 15 and 25 degrees Celsius would be perfect. Call the tool right away! Fill in the values in the args!!!",
                "timestamp": datetime.now().isoformat()
            }
            
            print("\n" + "=" * 50)
            print("📤 Follow-up: I'd say between 15 and 25 degrees Celsius would be perfect.")
            print("-" * 50)
            print("🤖 Bot response:")
            
            await websocket.send(json.dumps(followup_message))
            await listen_for_response(websocket, "Follow-up")
                
    except ConnectionRefusedError:
        print("❌ Connection refused. Make sure the API is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

async def listen_for_response(websocket, response_label="Response"):
    """Helper function to listen for and display a complete response"""
    response_count = 0
    try:
        while True:
            response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            response_data = json.loads(response)
            response_count += 1
            
            # Handle different response types
            if response_data.get("type") == "stream":
                # Streaming text response
                content = response_data.get("content", "")
                if content:
                    print(content, end="", flush=True)
            
            elif response_data.get("type") == "tool_call":
                # Tool execution
                tool_name = response_data.get("tool")
                args = response_data.get("args", {})
                print(f"\n🔧 Executing tool: {tool_name}")
                print(f"   Args: {args}")
            
            elif response_data.get("type") == "tool_result":
                # Tool result
                tool_name = response_data.get("tool")
                result = response_data.get("result", {})
                status = response_data.get("status")
                print(f"\n✅ Tool {tool_name} {status}")
                if "description" in result:
                    print(f"   Result: {result['description']}")
                if "state" in result:
                    state = result["state"]
                    print(f"   Cities remaining: {state.get('remaining_cities', 'unknown')}")
                    if state.get("remaining_city_names"):
                        cities = state["remaining_city_names"]
                        if len(cities) <= 10:
                            print(f"   Cities: {', '.join(cities)}")
            
            elif response_data.get("type") == "message_complete":
                # End of response
                print(f"\n\n✅ {response_label} complete ({response_count} chunks received)")
                if "state" in response_data:
                    state = response_data["state"]
                    print(f"🏙️ Search state: {state.get('remaining_cities', 'unknown')} cities remaining")
                    if state.get("applied_filters"):
                        print(f"🔍 Applied filters: {', '.join(state['applied_filters'].keys())}")
                    if state.get("remaining_city_names"):
                        cities = state["remaining_city_names"]
                        if len(cities) <= 10:
                            print(f"🏙️ Remaining cities: {', '.join(cities)}")
                break
            
            elif response_data.get("type") == "error":
                # Error response
                print(f"\n❌ Error: {response_data.get('content')}")
                break
            
            else:
                # Unknown response type
                print(f"\n🤔 Unknown response: {response_data}")
                
    except asyncio.TimeoutError:
        print(f"\n⏰ Timeout waiting for response (received {response_count} chunks)")

def main():
    # Get message from command line argument or use default climate-focused message
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        message = "I want to find a place that's not too cold and not too warm."
    
    print("🚀 Nomad Booster WebSocket Test Client - Climate Filter Test")
    print("=" * 60)
    
    # Run the async test
    asyncio.run(test_chat(message))

if __name__ == "__main__":
    main() 