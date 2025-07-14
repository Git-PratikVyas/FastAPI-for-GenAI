import pytest
import httpx
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_text_and_store():
    response = client.post("/generate", json={"prompt": "Test prompt", "max_length": 50})
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "generated_text" in data
    assert data["model_name"] == "gpt2"
    assert "created_at" in data
    
    # Verify history retrieval
    record_id = data["id"]
    history_response = client.get(f"/history/{record_id}")
    assert history_response.status_code == 200
    assert history_response.json()["id"] == record_id

def test_generate_text_invalid_prompt():
    response = client.post("/generate", json={"prompt": "", "max_length": 50})
    assert response.status_code == 422  # Pydantic validation error

def test_generate_text_forbidden_words():
    response = client.post("/generate", json={"prompt": "I hate this", "max_length": 50})
    assert response.status_code == 400
    assert "inappropriate content" in response.json()["detail"]

def test_history_not_found():
    response = client.get("/history/999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
