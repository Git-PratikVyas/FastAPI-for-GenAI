import pytest
import httpx
from fastapi.testclient import TestClient
from main import app
import asyncio

client = TestClient(app)

@pytest.mark.asyncio
async def test_generate_text_concurrent():
    async with httpx.AsyncClient(app=app, base_url="http://test") as async_client:
        # Send 5 concurrent requests
        tasks = [
            async_client.post("/generate", json={"prompt": f"Test prompt {i}", "max_length": 50})
            for i in range(5)
        ]
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            assert response.status_code == 200
            assert "generated_text" in response.json()
            assert response.json()["model"] == "gpt2"
            assert "request_id" in response.json()

def test_generate_text_invalid_prompt():
    response = client.post("/generate", json={"prompt": "", "max_length": 50})
    assert response.status_code == 422  # Pydantic validation error

def test_generate_text_forbidden_words():
    response = client.post("/generate", json={"prompt": "I hate this", "max_length": 50})
    assert response.status_code == 400
    assert "inappropriate content" in response.json()["detail"]
