# Testing AI Services

## Background
Testing AI services is critical to verify functionality, performance, and reliability, especially when dealing with unpredictable outputs from generative models like GPT-2. This project will focuses on unit, integration, and performance testing for a FastAPI-based AI service, covering endpoints, database interactions, and model inference. This project will build a FastAPI service with a generative AI model (Hugging Face’s GPT-2), store results in SQLite, and test it using `pytest`, `httpx`, and `unittest.mock` for mocking model outputs. This guide assumes familiarity with Python, FastAPI, SQLAlchemy, and basic security. This project will test API endpoints, database operations, and edge cases to ensure a production-ready service.

**Why Testing Matters**:
- **Reliability**: Catch bugs before they reach production.
- **Performance**: Ensure low latency and high throughput under load.
- **Correctness**: Validate AI outputs and database interactions.
- **Real-World Use**: Ensure chatbots, content generators, or AI APIs perform consistently.

---

## Testing a FastAPI AI Service

Build a FastAPI service with a generative AI model (GPT-2), store results in SQLite, and implement comprehensive tests using `pytest` for unit, integration, and performance testing.

**Libs**:
- Python 3.9+ (the base for our recipe)
- FastAPI (for the API framework)
- Uvicorn (ASGI server)
- Hugging Face Transformers (for the AI model)
- Torch (for model computation)
- Pydantic (for data validation)
- SQLAlchemy (for database ORM)
- `pytest`, `pytest-asyncio`, `httpx`, `unittest.mock` (for testing)
- Optional: `locust` (for performance testing)

**Tools**:
- Terminal or command line
- Code editor (e.g., VS Code)
- Virtual environment (to keep dependencies clean)
- SQLite (built into Python)

---
**Request Flow Diagram**:

The following diagram illustrates the detailed flow of a request through the system components:

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Pydantic
    participant ContentFilter
    participant ThreadPool
    participant AIModel
    participant SQLAlchemy
    participant Database
    
    Client->>FastAPI: POST /generate with prompt
    FastAPI->>Pydantic: Validate request body
    
    alt Invalid Request
        Pydantic-->>FastAPI: Validation Error
        FastAPI-->>Client: 422 Unprocessable Entity
    else Valid Request
        Pydantic-->>FastAPI: Validated Request
        FastAPI->>ContentFilter: Check for forbidden words
        
        alt Contains Forbidden Words
            ContentFilter-->>FastAPI: Content Violation
            FastAPI-->>Client: 400 Bad Request
        else Content OK
            ContentFilter-->>FastAPI: Content Approved
            FastAPI->>ThreadPool: Submit model inference task
            ThreadPool->>AIModel: Generate text
            AIModel-->>ThreadPool: Generated text
            ThreadPool-->>FastAPI: Generated text
            
            FastAPI->>SQLAlchemy: Create record
            SQLAlchemy->>Database: Insert record
            Database-->>SQLAlchemy: Record ID
            SQLAlchemy-->>FastAPI: Record with ID
            
            FastAPI-->>Client: 200 OK with generated text
        end
    end
    
    Client->>FastAPI: GET /history
    FastAPI->>SQLAlchemy: Query all records
    SQLAlchemy->>Database: SELECT query
    Database-->>SQLAlchemy: Records
    SQLAlchemy-->>FastAPI: Records
    FastAPI-->>Client: 200 OK with records
```

**Key Component**

- **Client**: External user or system sending HTTP requests to the API endpoints
- **FastAPI**: Web framework handling HTTP requests, routing, and dependency injection
- **Pydantic**: Data validation library ensuring request data meets defined schemas
- **ContentFilter**: Component checking for inappropriate content in user prompts
- **ThreadPool**: ThreadPoolExecutor managing concurrent model inference tasks
- **AIModel**: Hugging Face Transformers pipeline running the GPT-2 model
- **SQLAlchemy**: ORM library managing database operations and object mapping
- **Database**: SQLite database storing generation records and history


---

## Step 1: Project Environment

**Instructions**:
1. Create a project directory:
   ```bash
   mkdir ai-testing-service
   cd ai-testing-service
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install fastapi uvicorn pydantic transformers torch sqlalchemy pytest pytest-asyncio httpx locust
   ```

4. Save dependencies to a `requirements.txt` file:
   ```bash
   pip freeze > requirements.txt
   ```

**What You Get**:
```
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2
transformers==4.44.2
torch==2.4.1
sqlalchemy==2.0.35
pytest==8.3.2
pytest-asyncio==0.23.8
httpx==0.27.2
locust==2.31.6
```

- **Key Point**: `pytest-asyncio` enables async testing, and `locust` supports performance testing.

---

## Step 2: Define Pydantic and SQLAlchemy Models

Pydantic models validate API inputs/outputs, while SQLAlchemy models define the database schema.

**Instructions**:
1. Create a `models.py` file.
2. Define Pydantic models for API requests and SQLAlchemy models for storage.

**Code Example**:
```python
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# SQLAlchemy Base
Base = declarative_base()

# SQLAlchemy Model
class GenerationRecord(Base):
    __tablename__ = "generation_records"
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String, nullable=False)
    generated_text = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Input prompt for text generation")
    max_length: Optional[int] = Field(50, ge=10, le=200, description="Maximum length of generated text")

class TextGenerationResponse(BaseModel):
    id: int = Field(..., description="Database record ID")
    prompt: str = Field(..., description="Input prompt")
    generated_text: str = Field(..., description="Text generated by the AI model")
    model_name: str = Field(..., description="Name of the AI model used")
    created_at: str = Field(..., description="Timestamp of record creation")
```

- **`GenerationRecord`**: This is the SQLAlchemy model that maps to the `generation_records` table in our SQLite database. It stores the prompt, the AI-generated text, the model name, and a timestamp.
- **`TextGenerationRequest`**: This Pydantic model defines the expected input for our API. It ensures that incoming requests have a `prompt` and an optional `max_length`, with validation rules.
- **`TextGenerationResponse`**: This Pydantic model defines the structure of the API's response, ensuring it's consistent and includes all the necessary fields from the database record.

---

## Step 3: Database Configuration


**Instructions**:
1. Create a `database.py` file for SQLAlchemy setup.

**Code Example**:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

DATABASE_URL = "sqlite:///ai_testing.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- **`DATABASE_URL`**: Specifies the connection string for our SQLite database file, `ai_testing.db`.
- **`create_engine`**: Creates the SQLAlchemy engine. `check_same_thread: False` is required for SQLite to allow connections from multiple threads, which is necessary for FastAPI.
- **`SessionLocal`**: A factory for creating new database sessions.
- **`get_db`**: A dependency that provides a database session to our API endpoints and ensures the session is closed after the request is finished.

---

## Step 4: FastAPI Service


**Instructions**:
1. Create a `main.py` file.
2. Set up FastAPI with endpoints for text generation and history retrieval.

**Code Example**:
```python
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from models import TextGenerationRequest, TextGenerationResponse, GenerationRecord
from database import get_db
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Testing Service", version="1.0.0")
generator = pipeline("text-generation", model="gpt2", device=-1)  # CPU for simplicity
executor = ThreadPoolExecutor(max_workers=4)
FORBIDDEN_WORDS = ["hate", "violence"]

def run_model_inference(prompt: str, max_length: int) -> str:
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]["generated_text"]

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest, db: Session = Depends(get_db)):
    if any(word in request.prompt.lower() for word in FORBIDDEN_WORDS):
        logger.warning(f"Rejected prompt: {request.prompt}")
        raise HTTPException(status_code=400, detail="Prompt contains inappropriate content")
    
    try:
        loop = asyncio.get_running_loop()
        generated_text = await loop.run_in_executor(
            executor, run_model_inference, request.prompt, request.max_length
        )
        record = GenerationRecord(
            prompt=request.prompt,
            generated_text=generated_text,
            model_name="gpt2"
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(f"Stored record ID {record.id} for prompt: {request.prompt[:20]}...")
        return TextGenerationResponse(
            id=record.id,
            prompt=record.prompt,
            generated_text=record.generated_text,
            model_name=record.model_name,
            created_at=record.created_at.isoformat()
        )
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

@app.get("/history", response_model=list[TextGenerationResponse])
async def get_history(db: Session = Depends(get_db)):
    records = db.query(GenerationRecord).all()
    logger.info("Retrieved generation history")
    return [
        TextGenerationResponse(
            id=r.id,
            prompt=r.prompt,
            generated_text=r.generated_text,
            model_name=r.model_name,
            created_at=r.created_at.isoformat()
        )
        for r in records
    ]

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)
    logger.info("Thread pool shutdown")
```

- **`/generate` endpoint**: Receives a prompt, validates it against a list of forbidden words, runs the AI model to generate text, stores the result in the database, and returns the result.
- **`/history` endpoint**: Retrieves all previous text generation records from the database.
- **`ThreadPoolExecutor`**: The AI model inference is a blocking, CPU-bound task. We run it in a separate thread pool to prevent it from blocking the main application and to handle concurrent requests efficiently.
- **`shutdown_event`**: Ensures that the thread pool is shut down gracefully when the application exits.

---

## Step 5: Unit and Integration Tests

**Instructions**:
1. Create a `test_main.py` file.
2. Write unit and integration tests using `pytest`, `httpx`, and `unittest.mock`.

**Code Example**:
```python
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
    def mock_inference(prompt: str, max_length: int):
        return f"Mocked output for {prompt}"
    
    monkeypatch.setattr("main.run_model_inference", mock_inference)
    
    response = client.post(
        "/generate",
        json={"prompt": "Mocked prompt", "max_length": 50}
    )
    assert response.status_code == 200
    assert response.json()["generated_text"] == "Mocked output for Mocked prompt"
```

- **Test Database**: A separate in-memory SQLite database is used for testing to isolate tests from the development database. `override_get_db` replaces the production database dependency with the test database.
- **`setup_database` fixture**: This `pytest` fixture creates the database tables before each test and drops them afterward, ensuring a clean state for every test.
- **Unit Tests**: `test_generate_text_success`, `test_generate_text_forbidden_prompt`, and `test_generate_text_invalid_input` test the business logic of the `/generate` endpoint for different scenarios.
- **Integration Test**: `test_history_retrieval` is an integration test that checks the interaction between the `/generate` endpoint and the `/history` endpoint, verifying that data is correctly saved and retrieved.
- **Mocking**: `test_generate_text_mock_model` uses `monkeypatch` to replace the actual AI model inference with a simple function. This makes the test faster and more predictable, as it doesn't rely on the actual model's output.

**Command**:
```bash
pytest test_main.py
```

---

## Step 6: Performance Testing with Locust


**Instructions**:
1. Create a `locustfile.py` for performance testing.
2. Simulate multiple users sending requests to the `/generate` endpoint.

**Code Example**:
```python
from locust import HttpUser, task, between

class AIUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(1, 5)

    @task
    def generate_text(self):
        self.client.post(
            "/generate",
            json={"prompt": "Performance test prompt", "max_length": 50},
            headers={"Content-Type": "application/json"}
        )
```

- **`AIUser` class**: Defines a "user" for the load test. `wait_time` simulates a user waiting between 1 and 5 seconds before making another request.
- **`@task`**: The `generate_text` method is decorated with `@task`, marking it as a task that Locust will execute for its simulated users.
- **`self.client.post`**: This sends a POST request to the `/generate` endpoint with a sample payload.

Before running the Locust test, you need to start the FastAPI application.

3. **Start the FastAPI server**:
   In your terminal, from the `ai-testing-service` directory, run:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Run Locust**:
   In a separate terminal, from the same directory, run:
   ```bash
   locust -f locustfile.py
   ```

5. **Start the test**:
   Open `http://localhost:8089` in your browser, set the number of users (e.g., 50) and spawn rate (e.g., 5 users/second), and start the test.

---

## Project Structure
```
ai-testing-service/
├── main.py
├── models.py
├── database.py
├── test_main.py
├── locustfile.py
├── requirements.txt
├── Procfile
├── ai_testing.db (generated by SQLite)
└── venv/
```

---

