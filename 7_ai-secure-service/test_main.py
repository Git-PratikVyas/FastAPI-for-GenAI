# test_main.py

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from main import app, Base, get_db

# --- Test Database Setup ---
# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # Use StaticPool for in-memory DB
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the tables in the in-memory database before tests run
Base.metadata.create_all(bind=engine)


# --- Dependency Override ---
# This function will replace get_db in our tests
def override_get_db():
    database = TestingSessionLocal()
    try:
        yield database
    finally:
        database.close()


# Tell the app to use our new test database function instead of the real one
app.dependency_overrides[get_db] = override_get_db

# --- Your Tests (Now Isolated) ---
client = TestClient(app)


def test_create_user_and_generate():
    # Create user
    response = client.post(
        "/users", json={"username": "testuser_gen", "password": "securepassword"}
    )
    assert response.status_code == 200
    assert response.json()["username"] == "testuser_gen"

    # Login
    response = client.post(
        "/token", data={"username": "testuser_gen", "password": "securepassword"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]

    # Generate text
    response = client.post(
        "/generate",
        json={"prompt": "Test prompt", "max_length": 50},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "generated_text" in data
    assert data["model_name"] == "gpt2"

    # Check history
    response = client.get("/history", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert len(response.json()) > 0
    # The database is fresh, so history will have exactly 1 item
    assert len(response.json()) == 1


def test_unauthorized_access():
    response = client.post(
        "/generate", json={"prompt": "Test prompt", "max_length": 50}
    )
    assert response.status_code == 401
    # The default message from FastAPI 0.100+ is "Not authenticated"
    assert response.json()["detail"] == "Not authenticated"


def test_forbidden_prompt():
    # Because each test is isolated, we need to create the user again
    client.post(
        "/users", json={"username": "testuser_forbidden", "password": "securepassword"}
    )
    response = client.post(
        "/token", data={"username": "testuser_forbidden", "password": "securepassword"}
    )
    token = response.json()["access_token"]

    response = client.post(
        "/generate",
        json={"prompt": "I hate this", "max_length": 50},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400
    assert "inappropriate content" in response.json()["detail"]


def test_sanitized_input():
    client.post(
        "/users", json={"username": "testuser_sanitize", "password": "securepassword"}
    )
    response = client.post(
        "/token", data={"username": "testuser_sanitize", "password": "securepassword"}
    )
    token = response.json()["access_token"]

    response = client.post(
        "/generate",
        json={"prompt": "Test <script> alert('hack');</script>", "max_length": 50},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    # The prompt itself is sanitized before being passed to the model,
    # but the model's output is what we should check.
    # Note: gpt2 might still generate special characters. This test might be flaky.
    # The important part is that the input to the model was sanitized.
    assert "<script>" not in response.json()["generated_text"]
