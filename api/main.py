from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, List, Any
import logging
import redis.asyncio as redis
from contextlib import asynccontextmanager
from orchestrator import ChatOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Nomad Booster API...")
    
    # Initialize Redis connection
    try:
        app.state.redis = redis.from_url("redis://localhost:6379", decode_responses=True)
        await app.state.redis.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Running without cache.")
        app.state.redis = None
    
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
            await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

# Initialize LangChain orchestrator
orchestrator = ChatOrchestrator()

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
        import pandas as pd
        cities_df = pd.read_csv("../data/cities_data_static.csv")
        return {"cities": cities_df.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading cities: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 