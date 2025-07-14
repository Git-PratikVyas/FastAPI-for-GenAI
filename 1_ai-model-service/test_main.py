import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_text_success():
    response = client.post("/generate", json={"prompt": "The future of AI", "max_length": 50})
    assert response.status_code == 200
    assert "generated_text" in response.json()
    assert response.json()["model_name"] == "gpt2"

def test_generate_text_invalid_prompt():
    response = client.post("/generate", json={"prompt": "", "max_length": 50})
    assert response.status_code == 422  # Pydantic validation error

def test_generate_text_forbidden_words():
    response = client.post("/generate", json={"prompt": "I hate this", "max_length": 50})
    assert response.status_code == 400
    assert "inappropriate content" in response.json()["detail"]
