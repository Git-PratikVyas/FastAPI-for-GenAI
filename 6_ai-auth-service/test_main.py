import pytest
import httpx
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_create_user_and_generate():
    # Create user
    response = client.post("/users", json={"username": "testuser", "password": "securepassword", "role": "user"})
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
    
    # Login
    response = client.post("/token", data={"username": "testuser", "password": "securepassword"})
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # Generate text
    response = client.post(
        "/generate",
        json={"prompt": "Test prompt", "max_length": 50},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "generated_text" in response.json()
    assert response.json()["model_name"] == "gpt2"

def test_unauthorized_access():
    response = client.post("/generate", json={"prompt": "Test prompt", "max_length": 50})
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]

def test_admin_endpoint_access():
    # Create admin user
    response = client.post("/users", json={"username": "adminuser", "password": "adminpassword", "role": "admin"})
    assert response.status_code == 200
    
    # Login as admin
    response = client.post("/token", data={"username": "adminuser", "password": "adminpassword"})
    token = response.json()["access_token"]
    
    # Access admin endpoint
    response = client.get("/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_non_admin_access_denied():
    # Login as regular user
    response = client.post("/token", data={"username": "testuser", "password": "securepassword"})
    token = response.json()["access_token"]
    
    # Try accessing admin endpoint
    response = client.get("/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    assert "Admin access required" in response.json()["detail"]
