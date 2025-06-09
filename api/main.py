from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import os
from typing import Dict, List, Any
import logging
import redis.asyncio as redis
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from orchestrator import ChatOrchestrator
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Nomad Booster API...")
    
    # Initialize Redis connection
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        app.state.redis = redis.from_url(redis_url, decode_responses=True)
        await app.state.redis.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Running without cache.")
        app.state.redis = None
    
    # Initialize orchestrator with Redis
    global orchestrator
    orchestrator = ChatOrchestrator(redis_client=app.state.redis)
    logger.info("Orchestrator initialized with Redis support")
    
    yield
    
    # Shutdown
    if hasattr(app.state, 'redis') and app.state.redis:
        await app.state.redis.close()
    logger.info("Shutdown complete")

app = FastAPI(
    title="Nomad Booster API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React/Vite dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            
            # Log critical data for message_complete events
            if message.get("type") == "message_complete":
                logger.info(f"📡 SENDING message_complete to {client_id}")
                state = message.get("state", {})
                logger.info(f"📡 State keys: {list(state.keys())}")
                cities_complete = state.get("cities_complete_data", [])
                logger.info(f"📡 cities_complete_data length before JSON: {len(cities_complete)}")
                if cities_complete:
                    logger.info(f"📡 Sample city before JSON: {cities_complete[0].get('city', 'NO_CITY')}")
                
                # Test JSON serialization explicitly
                try:
                    json_str = json.dumps(message)
                    logger.info(f"📡 JSON serialization successful, length: {len(json_str)}")
                    
                    # Parse it back to verify
                    parsed = json.loads(json_str)
                    parsed_state = parsed.get("state", {})
                    parsed_cities = parsed_state.get("cities_complete_data", [])
                    logger.info(f"📡 After JSON round-trip, cities_complete_data length: {len(parsed_cities)}")
                    
                except Exception as e:
                    logger.error(f"📡 JSON serialization failed: {e}")
                    logger.error(f"📡 Problem message type: {type(message)}")
                    logger.error(f"📡 Problem state type: {type(state)}")
                    logger.error(f"📡 Problem cities type: {type(cities_complete)}")
            
            await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

# Initialize LangChain orchestrator (will be initialized with Redis in lifespan)
orchestrator = None

@app.get("/")
async def root():
    return {
        "message": "Nomad Booster API", 
        "version": "1.0.0",
        "websocket_endpoint": "/ws/{client_id}",
        "features": ["langchain", "streaming", "tool_binding"]
    }

@app.get("/health")
async def health_check():
    redis_status = "connected" if hasattr(app.state, 'redis') and app.state.redis else "disconnected"
    return {
        "status": "healthy",
        "redis": redis_status,
        "active_connections": len(manager.active_connections),
        "orchestrator": "langchain_ready"
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        # Send initialization message to trigger session loading
        init_message = {
            "content": "",
            "type": "init",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Process initialization to load session history
        async for response_chunk in orchestrator.process_message_stream(client_id, init_message):
            await manager.send_personal_message(response_chunk, client_id)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info(f"Received from {client_id}: {message}")
            
            # Stream response from orchestrator
            async for response_chunk in orchestrator.process_message_stream(client_id, message):
                await manager.send_personal_message(response_chunk, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Error in websocket for {client_id}: {e}")
        # Send error message to client
        error_response = {
            "type": "error",
            "content": f"Connection error: {str(e)}"
        }
        try:
            await manager.send_personal_message(error_response, client_id)
        except:
            pass  # Connection might be closed
        finally:
            manager.disconnect(client_id)

@app.get("/cities")
async def get_cities():
    """Get static cities data"""
    try:
        cities_df = pd.read_csv("../data/cities_static_properties.csv")
        return {"cities": cities_df.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading cities: {str(e)}")

@app.get("/sessions")
async def list_sessions(user_id: str = None):
    """List available chat sessions"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    sessions = await orchestrator.list_user_sessions(user_id)
    return {"sessions": sessions}

@app.post("/sessions/{session_id}/resume")
async def resume_session(session_id: str):
    """Resume a chat session by loading it from Redis"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        session, search_state = await orchestrator.load_session_from_redis(session_id)
        if not session.get("messages"):
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "message_count": len(session.get("messages", [])),
            "search_state": search_state.get_state_summary(),
            "last_updated": session.get("updated_at"),
            "status": "loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading session: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session from Redis"""
    if not orchestrator.redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        session_key = f"chat_session:{session_id}"
        search_key = f"search_state:{session_id}"
        
        deleted_session = await orchestrator.redis_client.delete(session_key)
        deleted_search = await orchestrator.redis_client.delete(search_key)
        
        if deleted_session == 0 and deleted_search == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 