import asyncio
import websockets
import json
from models import WebSocketMessage

async def test_websocket():
    uri = "ws://localhost:8000/ws/generate"
    async with websockets.connect(uri) as websocket:
        # Send a valid message
        message = WebSocketMessage(
            prompt="The future of AI is",
            max_length=50,
            client_id="test-client"
        )
        await websocket.send(json.dumps(message.dict()))
        response = await websocket.recv()
        print(f"Received: {response}")
        
        # Send an invalid message
        await websocket.send(json.dumps({"prompt": "I hate this"}))
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(test_websocket())
