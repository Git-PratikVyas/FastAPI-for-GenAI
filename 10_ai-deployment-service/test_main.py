import pytest
import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import app
from database import get_db
from models import Base

TEST_DATABASE_URL = "sqlite:///test_ai_deployment.db"
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
    assert data["model_name"] == "distilgpt2"

@pytest.mark.asyncio
async def test_generate_text_forbidden_prompt():
    response = client.post(
        "/generate",
        json={"prompt": "I hate this", "max_length": 50}
    )
    assert response.status_code == 400
    assert "inappropriate content" in response.json()["detail"]

@pytest.mark.asyncio
async def test_history_retrieval():
    client.post("/generate", json={"prompt": "History test", "max_length": 50})
    response = client.get("/history")
    assert response.status_code == 200
    assert len(response.json()) == 1
