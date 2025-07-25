# Optimizing AI Services


## Background
AI services, especially those using generative models like GPT-2, can be resource-intensive, demanding careful optimization to ensure speed, scalability, and cost-efficiency. This project will likely covers techniques like model caching, async processing, batch inference, and resource management to optimize FastAPI-based AI services. We’ll integrate a generative AI model (Hugging Face’s GPT-2) with FastAPI, optimize it using caching (`fastapi-cache2`), batch processing, and efficient model loading, and store results in SQLite for persistence. This guide assumes familiarity with Python, FastAPI, SQLAlchemy, and security practices. Project will use Redis for caching and `concurrent.futures` for thread management to maximize performance.

**Why Optimization Matters**:
- **Speed**: Reduce latency for AI inference to improve user experience.
- **Scalability**: Handle more requests without crashing or slowing down.
- **Cost-Efficiency**: Minimize compute resources for cloud deployments.
- **Real-World Use**: Power high-traffic chatbots, content generators, or real-time AI apps.


---

## Optimizing a FastAPI AI Service

Build a FastAPI service that integrates a generative AI model (GPT-2), optimizes it with caching, batch inference, and efficient resource management, and stores results in SQLite.

**Libs**:
- Python 3.9+ (the base for our recipe)
- FastAPI (for the API framework)
- Uvicorn (ASGI server)
- Hugging Face Transformers (for the AI model)
- Torch (for model computation)
- Pydantic (for data validation)
- SQLAlchemy (for database ORM)
- `fastapi-cache2` (for response caching with Redis)
- `redis` (Python client for Redis)
- `concurrent.futures` (for thread management)
- Optional: Pytest and `httpx` (for testing)

**Tools**:
- Terminal or command line
- Code editor (e.g., VS Code)
- Virtual environment (to keep dependencies clean)
- SQLite (built into Python)
- Redis (for caching, installed separately)
- Docker (for running Redis)


---

## Architecture

The system follows a layered architecture with the following components:

1. **API Layer**: FastAPI application that handles HTTP requests and responses
2. **Service Layer**: Business logic for text generation and optimization
3. **Data Layer**: Database access and caching mechanisms
4. **Model Layer**: AI model integration and inference

```mermaid
graph TD
    Client[Client] --> API[API Layer - FastAPI]
    API --> Cache[Redis Cache]
    API --> Service[Service Layer]
    Service --> Model[Model Layer - GPT-2]
    Service --> DataAccess[Data Access Layer]
    DataAccess --> DB[(SQLite Database)]
    Cache -.-> Service
```

### Request Flow Diagram

The following diagram illustrates the detailed flow of a request through the system components:

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Cache as Redis Cache
    participant Service
    participant ThreadPool
    participant Model as GPT-2 Model
    participant DB as SQLite Database
    
    Client->>FastAPI: POST /generate with prompts
    FastAPI->>FastAPI: Validate request with Pydantic
    FastAPI->>Cache: Check if response is cached
    
    alt Response in cache
        Cache->>FastAPI: Return cached response
        FastAPI->>Client: Return generated text
    else Cache miss
        FastAPI->>Service: Process prompts
        Service->>Service: Filter inappropriate content
        Service->>ThreadPool: Submit batch inference task
        ThreadPool->>Model: Run batch inference
        Model->>ThreadPool: Return generated texts
        ThreadPool->>Service: Return results
        Service->>DB: Store generation records
        DB->>Service: Confirm storage
        Service->>Cache: Store in cache
        Service->>FastAPI: Return results
        FastAPI->>Client: Return generated text
    end
```

---

## Step 1: Project Environment


**Instructions**:
1.  **Install Docker**: Make sure you have Docker Desktop installed and running on your system.

2.  **Run Redis with Docker**: Open your terminal and run the following command to start a Redis container. This is a modern, isolated way to run services.
    ```bash
    docker run --name my-redis-container -p 6379:6379 -d redis
    ```

3.  **Check if Redis is running**: To confirm that your Redis container is running correctly, use the following command:
    ```bash
    docker exec my-redis-container redis-cli ping
    ```
    If it's working, you should see the response: `PONG`

4. Create a project directory:
   ```bash
   mkdir ai-optimized-service
   cd ai-optimized-service
   ```

5. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

6. Install required packages:
   ```bash
   pip install fastapi uvicorn pydantic transformers torch sqlalchemy fastapi-cache2 redis pytest httpx
   ```

7. Save dependencies to a `requirements.txt` file:
   ```bash
   pip freeze > requirements.txt
   ```

**What You Get**:
```
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2
transformers==4.44.2
torch==2.7.1
sqlalchemy==2.0.35
fastapi-cache2
redis==5.0.8
pytest==8.3.2
httpx==0.27.2
```

---

## Step 2: Define Pydantic and SQLAlchemy Models

Pydantic models validate API inputs/outputs, while SQLAlchemy models define the database schema.
**Instructions**:
1. Create a `models.py` file.
2. Define Pydantic models for AI requests and SQLAlchemy models for storing results.

**Code Example**:
```python
from pydantic import BaseModel, Field
from typing import Optional, List
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import declarative_base
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
    created_at = Column(DateTime, default=func.now())

# Pydantic Models
class TextGenerationRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=10, description="List of input prompts for batch processing")
    max_length: Optional[int] = Field(50, ge=10, le=200, description="Maximum length of generated text")

class TextGenerationResponse(BaseModel):
    id: int = Field(..., description="Database record ID")
    prompt: str = Field(..., description="Input prompt")
    generated_text: str = Field(..., description="Text generated by the AI model")
    model_name: str = Field(..., description="Name of the AI model used")
    created_at: str = Field(..., description="Timestamp of record creation")
```
**Key Snippets**:
*   `GenerationRecord`: A SQLAlchemy model defining the `generation_records` table schema for storing prompts, generated text, model name, and a timestamp in SQLite.
*   `TextGenerationRequest`: A Pydantic model for validating API input. It requires a list of `prompts` and an optional `max_length`, ensuring the input data is well-formed.
*   `TextGenerationResponse`: A Pydantic model that defines the structure of the API's JSON response, ensuring consistency and providing clear documentation for API consumers.

---

## Step 3: Database and Cache Configuration


**Instructions**:
1. Create a `database.py` file for SQLAlchemy setup.
2. Create a `cache.py` file for Redis caching.

**Code Example (Database)**:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

DATABASE_URL = "sqlite:///ai_optimized.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```
**Key Snippets (Database)**:
*   `create_engine`: Establishes the connection to the SQLite database. `check_same_thread=False` is crucial for compatibility with FastAPI's async environment.
*   `SessionLocal`: Creates a factory for generating new database sessions.
*   `get_db`: A FastAPI dependency that provides a database session to API endpoints and guarantees the session is closed after the request is complete, preventing resource leaks.

**Code Example (Cache)**:
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis.asyncio import Redis

async def init_cache():
    redis = await Redis(host="localhost", port=6379, db=0, decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
```
**Key Snippets (Cache)**:
*   `init_cache`: An async function that sets up the connection to the Redis server.
*   `FastAPICache.init`: Initializes `fastapi-cache2` with the Redis backend. The `prefix` helps namespace keys, preventing collisions if the Redis instance is shared.

---

## Step 4: FastAPI with Optimization Features

Combining caching, batch inference, and efficient resource management creates a high-performance AI service.

**Instructions**:
1. Create a `main.py` file.
2. Set up FastAPI with caching, batch inference, and thread management.

**Code Example**:
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi_cache.decorator import cache
from sqlalchemy.orm import Session
from models import TextGenerationRequest, TextGenerationResponse, GenerationRecord
from database import get_db
from cache import init_cache
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Optimized AI Service", version="1.0.0")
generator = pipeline("text-generation", model="gpt2", device=-1)  # CPU for simplicity
executor = ThreadPoolExecutor(max_workers=4)
FORBIDDEN_WORDS = ["hate", "violence"]

def run_batch_inference(prompts: list[str], max_length: int) -> list[str]:
    """Batch inference for multiple prompts."""
    results = generator(prompts, max_length=max_length, num_return_sequences=1, batch_size=len(prompts))
    return [result[0]["generated_text"] for result in results]

@app.post("/generate", response_model=list[TextGenerationResponse])
@cache(expire=60)  # Cache responses for 60 seconds
async def generate_text(request: TextGenerationRequest, db: Session = Depends(get_db)):
    # Validate prompts
    sanitized_prompts = [prompt for prompt in request.prompts if not any(word in prompt.lower() for word in FORBIDDEN_WORDS)]
    if len(sanitized_prompts) != len(request.prompts):
        logger.warning("Some prompts were filtered out due to forbidden words")
        raise HTTPException(status_code=400, detail="Some prompts contain inappropriate content")
    
    try:
        # Run batch inference in thread pool
        loop = asyncio.get_event_loop()
        generated_texts = await loop.run_in_executor(
            executor, run_batch_inference, sanitized_prompts, request.max_length
        )
        
        # Store results in database
        responses = []
        for prompt, generated_text in zip(sanitized_prompts, generated_texts):
            record = GenerationRecord(
                prompt=prompt,
                generated_text=generated_text,
                model_name="gpt2"
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            logger.info(f"Stored record ID {record.id} for prompt: {prompt[:20]}...")
            responses.append(
                TextGenerationResponse(
                    id=record.id,
                    prompt=prompt,
                    generated_text=generated_text,
                    model_name="gpt2",
                    created_at=record.created_at.isoformat()
                )
            )
        
        return responses
    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

@app.get("/history", response_model=list[TextGenerationResponse])
@cache(expire=300)  # Cache history for 5 minutes
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

@app.on_event("startup")
async def startup_event():
    await init_cache()
    logger.info("FastAPI app started with cache and thread pool")

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)
    logger.info("Thread pool shutdown")
```
**Key Snippets**:
*   `generator = pipeline(...)`: Loads the GPT-2 model. `device=-1` specifies CPU usage for broader compatibility.
*   `executor = ThreadPoolExecutor(...)`: Creates a thread pool to run the synchronous, CPU-intensive model inference without blocking the main async event loop.
*   `run_batch_inference(...)`: A function that processes a list of prompts at once, which is significantly more efficient than one-by-one processing.
*   `@cache(expire=60)`: A decorator that caches the endpoint's response in Redis for 60 seconds. Identical subsequent requests are served from the cache, skipping model inference entirely.
*   `loop.run_in_executor(...)`: The core of the async optimization. It runs the blocking `run_batch_inference` function in a separate thread from the pool, keeping the API responsive.
*   `@app.on_event("startup")` / `@app.on_event("shutdown")`: Event handlers that manage the application's lifecycle, ensuring the cache is initialized on startup and the thread pool is shut down gracefully.

---

## Step 5: Run the FastAPI Server


**Instructions**:
1. Ensure Redis is running by pinging it through the Docker container:
   ```bash
   docker exec my-redis-container redis-cli ping
   ```

2. Start the server with Uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. Visit `http://127.0.0.1:8000/docs` for Swagger UI to test the API.

**Try It Out**:
Test batch inference:
```bash
curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"prompts": ["The future of AI is", "AI will revolutionize"], "max_length": 50}'
```

---

## Step 6: Test the Service


**Instructions**:
1. Create a `test_main.py` file to test the API.
2. Use `pytest` and `httpx` for HTTP testing.

**Code Example**:
```python
import pytest
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
```
**Key Snippets**:
*   `TestClient(app)`: Creates a test client to make requests to the FastAPI app directly within the test suite, avoiding the need for a live server.
*   `client.post(...)` and `client.get(...)`: Simulate HTTP requests to the API endpoints to test their behavior.
*   `assert response.status_code == 200`: Verifies that the request was successful.
*   The tests cover the main success case (batch generation), an error case (forbidden words), and data retrieval (`/history`), ensuring the core features work as expected.

3. Run the tests:
   ```bash
   pytest test_main.py
   ```

---

## Step 7: Deploy the Service

**Instructions**:
1. Create a `Procfile` for deployment (e.g., on Render):

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4
```

2. Deploy to a platform like Render, ensuring `requirements.txt` is included.
3. Configure Redis in production (e.g., use a managed Redis service like Redis Labs).
4. Set environment variables for Redis host/port if needed.

---

## Project Structure
```
ai-optimized-service/
├── main.py
├── models.py
├── database.py
├── cache.py
├── test_main.py
├── requirements.txt
├── Procfile
├── ai_optimized.db (generated by SQLite)
└── .venv/
```
