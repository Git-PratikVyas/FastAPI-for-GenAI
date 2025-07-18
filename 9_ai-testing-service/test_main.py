import pytest
import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import app, run_model_inference
from database import get_db
from models import Base, GenerationRecord
import unittest.mock

# Test database setup
TEST_DATABASE_URL = "sqlite:///test_ai_testing.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)

@pytest.mark.asyncio
async def test_generate_text_success():
    response = client.post(
        "/generate",
        json={"prompt": "Test prompt", "max_length": 50}
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["prompt"] == "Test prompt"
    assert data["model_name"] == "gpt2"
    assert "generated_text" in data
    assert "created_at" in data

@pytest.mark.asyncio
async def test_generate_text_forbidden_prompt():
    response = client.post(
        "/generate",
        json={"prompt": "I hate this", "max_length": 50}
    )
    assert response.status_code == 400
    assert "inappropriate content" in response.json()["detail"]

@pytest.mark.asyncio
async def test_generate_text_invalid_input():
    response = client.post(
        "/generate",
        json={"prompt": "", "max_length": 50}
    )
    assert response.status_code == 422  # Pydantic validation error

@pytest.mark.asyncio
async def test_history_retrieval():
    # Generate a record
    client.post("/generate", json={"prompt": "History test", "max_length": 50})
    
    # Retrieve history
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["prompt"] == "History test"

@pytest.mark.asyncio
async def test_generate_text_mock_model(monkeypatch):
    # Mock the model inference
    async def mock_inference(prompt: str, max_length: int):
        return f"Mocked output for {prompt}"
    
    monkeypatch.setattr("main.run_model_inference", mock_inference)
    
    response = client.post(
        "/generate",
        json={"prompt": "Mocked prompt", "max_length": 50}
    )
    assert response.status_code == 200
    assert response.json()["generated_text"] == "Mocked output for Mocked prompt"
