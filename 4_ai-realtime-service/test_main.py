import pytest
import websockets
import json
from models import WebSocketMessage

@pytest.mark.asyncio
async def test_websocket_generate():
    uri = "ws://localhost:8000/ws/generate"
    async with websockets.connect(uri) as websocket:
        # Test valid message
        message = WebSocketMessage(
            prompt="Test prompt",
            max_length=50,
            client_id="test-client"
        )
        await websocket.send(json.dumps(message.dict()))
        response = json.loads(await websocket.recv())
        assert "generated_text" in response
        assert response["model_name"] == "gpt2"
        assert "client_id" in response
        
        # Test forbidden words
        await websocket.send(json.dumps({"prompt": "I hate this", "max_length": 50, "client_id": "test-client"}))
        response = json.loads(await websocket.recv())
        assert "error" in response
        assert "inappropriate content" in response["error"]

@pytest.mark.asyncio
async def test_websocket_invalid_format():
    uri = "ws://localhost:8000/ws/generate"
    async with websockets.connect(uri) as websocket:
        await websocket.send("invalid json")
        response = json.loads(await websocket.recv())
        assert "error" in response
        assert "Invalid message format" in response["error"]
