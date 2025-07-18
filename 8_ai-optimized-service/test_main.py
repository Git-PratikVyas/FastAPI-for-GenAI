import pytest
import httpx
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_text_batch():
    response = client.post(
        "/generate",
        json={"prompts": ["Test prompt 1", "Test prompt 2"], "max_length": 50}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for item in data:
        assert "id" in item
        assert "prompt" in item
        assert "generated_text" in item
        assert item["model_name"] == "gpt2"

def test_generate_forbidden_prompt():
    response = client.post(
        "/generate",
        json={"prompts": ["I hate this"], "max_length": 50}
    )
    assert response.status_code == 400
    assert "inappropriate content" in response.json()["detail"]

def test_history_retrieval():
    # Generate some records
    client.post(
        "/generate",
        json={"prompts": ["Test history"], "max_length": 50}
    )
    response = client.get("/history")
    assert response.status_code == 200
    assert len(response.json()) > 0

def test_cache_hit():
    # First request
    response1 = client.post(
        "/generate",
        json={"prompts": ["Cached prompt"], "max_length": 50}
    )
    assert response1.status_code == 200
    
    # Second request (should hit cache)
    response2 = client.post(
        "/generate",
        json={"prompts": ["Cached prompt"], "max_length": 50}
    )
    assert response2.status_code == 200
    assert response1.json() == response2.json()  # Cached response should match
