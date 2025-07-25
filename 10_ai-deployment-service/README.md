# Deploying AI Services

## Background for Developers
Deploying an AI service involves moving your FastAPI application, generative model, and database to a cloud platform, ensuring scalability, reliability, and security. This project will focuses on deploying a FastAPI-based AI service with a generative model (Hugging Face’s GPT-2) to a platform like Render, complete with a managed database (PostgreSQL), environment variables, and production-grade configurations. We’ll use Docker for containerization, Render for hosting, and PostgreSQL for persistence, building on concepts from authentication, optimization, testing. This guide assumes familiarity with Python, FastAPI, SQLAlchemy, and basic testing. We’ll also secure the service with environment variables and HTTPS.

**Why Deployment Matters**:
- **Accessibility**: Make your AI service available to users worldwide.
- **Scalability**: Handle high traffic with cloud infrastructure.
- **Reliability**: Ensure uptime and robust error handling.
- **Real-World Use**: Power public-facing chatbots, content generators, or AI APIs.


---

## Deploying a FastAPI AI Service to Render

Deploy a FastAPI service with a generative AI model (GPT-2) and PostgreSQL to Render, using Docker for containerization and environment variables for configuration.

**Libs**:
- Python 3.9+ (the base for our recipe)
- FastAPI (for the API framework)
- Uvicorn (ASGI server with Gunicorn for production)
- Hugging Face Transformers (for the AI model)
- Torch (for model computation)
- Pydantic (for data validation)
- SQLAlchemy (for database ORM)
- `psycopg2-binary` (for PostgreSQL)
- Docker (for containerization)
- Optional: Pytest and `httpx` (for pre-deployment testing)

**Tools**:
- Terminal or command line
- Code editor (e.g., VS Code)
- Virtual environment (for local development)
- Docker Desktop (for building containers)
- Render account (for cloud hosting)


**Request Flow Diagram**:

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant ThreadPool
    participant Model as AI Model
    participant DB as PostgreSQL
    
    Client->>FastAPI: POST /generate
    FastAPI->>FastAPI: Validate request
    FastAPI->>ThreadPool: Run model inference
    ThreadPool->>Model: Generate text
    Model-->>ThreadPool: Return generated text
    ThreadPool-->>FastAPI: Return result
    FastAPI->>DB: Store generation record
    DB-->>FastAPI: Confirm storage
    FastAPI-->>Client: Return response
    
    Client->>FastAPI: GET /history
    FastAPI->>DB: Query records
    DB-->>FastAPI: Return records
    FastAPI-->>Client: Return response
```

**Key Component**

- **Client**: End-user or application consuming the AI service API
- **FastAPI**: Web framework handling HTTP requests, validation, and responses
- **ThreadPool**: Executor managing concurrent model inference without blocking the main event loop
- **AI Model**: DistilGPT-2 model from Hugging Face for text generation
- **PostgreSQL**: Database for storing generation records and history
---

## Step 1: Project Environment


**Instructions**:
1. Install Docker:
   - Follow instructions at [docker.com](https://www.docker.com/get-started) for your OS.
   - Verify installation:
     ```bash
     docker --version
     ```

2. Create a project directory:
   ```bash
   mkdir ai-deployment-service
   cd ai-deployment-service
   ```

3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

4. Install required packages:
   ```bash
   pip install fastapi uvicorn gunicorn pydantic transformers torch sqlalchemy psycopg2-binary pytest httpx
   ```

5. Save dependencies to a `requirements.txt` file:
   ```bash
   pip freeze > requirements.txt
   ```

**What You Get**:

```
fastapi==0.115.0
uvicorn==0.30.6
gunicorn==23.0.0
pydantic==2.9.2
transformers==4.44.2
torch==2.4.1
sqlalchemy==2.0.35
psycopg2-binary==2.9.9
pytest==8.3.2
httpx==0.27.2
```

**Pro Tip**: `gunicorn` is used with Uvicorn for production-grade worker management, and `psycopg2-binary` connects to PostgreSQL.

---

## Step 2: Define Pydantic and SQLAlchemy Models

Pydantic models validate API inputs/outputs, and SQLAlchemy models define the database schema.

**Instructions**:
1. Create a `models.py` file.
2. Define Pydantic models for API requests and SQLAlchemy models for PostgreSQL.

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

**Key Code Explained**:
-   `Base = declarative_base()`: Sets up the base class for SQLAlchemy models.
-   `GenerationRecord`: Defines the database table schema with columns for `id`, `prompt`, `generated_text`, `model_name`, and `created_at`.
-   `TextGenerationRequest`: Pydantic model for validating the incoming API request body.
-   `TextGenerationResponse`: Pydantic model for validating the outgoing API response.



- `GenerationRecord` defines the PostgreSQL table schema.
- Pydantic models ensure strict input/output validation.

---

## Step 3: Database Configuration


**Instructions**:
1. Create a `database.py` file for SQLAlchemy setup with PostgreSQL support.

**Code Example**:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
import os

# Use environment variable for database URL
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///ai_deployment.db")

# Create engine (fallback to SQLite for local development)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```


-   `DATABASE_URL = os.environ.get(...)`: Fetches the database connection string from an environment variable, with a fallback to a local SQLite database. This is crucial for production environments.
-   `create_engine(...)`: Creates the SQLAlchemy engine, which manages connections to the database.
-   `SessionLocal`: A factory for creating new database sessions.
-   `get_db()`: A dependency that provides a database session to the API endpoints and ensures the session is closed after the request is finished.


- Supports PostgreSQL via `DATABASE_URL` (set in production) and SQLite locally.
- `get_db` provides a session for dependency injection.

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
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Deployment Service", version="1.0.0")
generator = pipeline("text-generation", model="distilgpt2", device=-1)  # Use distilgpt2 for smaller size
executor = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", 4)))
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
        generated_text = await app.state.loop.run_in_executor(
            executor, run_model_inference, request.prompt, request.max_length
        )
        record = GenerationRecord(
            prompt=request.prompt,
            generated_text=generated_text,
            model_name="distilgpt2"
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

@app.on_event("startup")
async def startup_event():
    app.state.loop = app.state.loop or __import__("asyncio").get_event_loop()
    logger.info("FastAPI app started with thread pool")

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)
    logger.info("Thread pool shutdown")
```

**Key Code Explained**:
-   `generator = pipeline(...)`: Loads the `distilgpt2` model from Hugging Face, which is a smaller and faster version of GPT-2.
-   `ThreadPoolExecutor`: Manages a pool of threads to run the synchronous model inference code without blocking the asyncio event loop.
-   `@app.post("/generate")`: The main endpoint that receives a prompt, runs the model, and saves the result to the database.
-   `run_in_executor`: Runs the blocking `run_model_inference` function in a separate thread from the thread pool.
-   `@app.get("/history")`: An endpoint to retrieve all the generation records from the database.
-   `@app.get("/health")`: A simple health check endpoint that Render uses to monitor the service's health.

**Notes**:
- Uses `distilgpt2` (smaller than GPT-2) to reduce memory usage and speed up inference.
- `WORKERS` environment variable controls thread pool size.
- Endpoints are identical to previous chapters for consistency.

---

## Step 5: Docker Setup


**Instructions**:
1. Create a `Dockerfile` for containerizing the app.
2. Create a `.dockerignore` to exclude unnecessary files.

**Code Example (Dockerfile)**:
```
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV WORKERS=4

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--workers", "$WORKERS"]
```

**Key Code Explained**:
-   `FROM python:3.9-slim`: Uses a lightweight Python base image to keep the final container size small.
-   `WORKDIR /app`: Sets the working directory inside the container.
-   `COPY requirements.txt .` and `RUN pip install ...`: Copies the requirements file and installs the dependencies.
-   `COPY . .`: Copies the rest of the application code into the container.
-   `CMD ["gunicorn", ...]`: The command that runs when the container starts. It uses `gunicorn` as a production-grade process manager for the `uvicorn` ASGI server.

**Code Example (.dockerignore)**:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
*.db
```

**Key Code Explained**:
-   This file lists files and directories that should not be copied into the Docker image, such as the virtual environment (`venv/`) and local databases (`*.db`). This helps to keep the image size down and avoid including unnecessary files.

3. Build the Docker image:
   ```bash
   docker build -t ai-deployment-service .
   ```

4. Test the container locally:
   ```bash
   docker run -p 8000:8000 -e DATABASE_URL="sqlite:///ai_deployment.db" ai-deployment-service
   ```

5. Verify at `http://localhost:8000/docs`.

**Pro Tip**: Use `python:3.9-slim` to reduce image size and `gunicorn` for production-grade worker management.

---

## Step 6: Validate Locally

**Instructions**:
1. Create a `test_main.py` file for unit and integration tests.

**Code Example**:
```python
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
```

**Key Code Explained**:
-   `TEST_DATABASE_URL`: Defines a separate in-memory SQLite database for testing to isolate tests from the development database.
-   `override_get_db`: A fixture that overrides the `get_db` dependency to use the test database during tests.
-   `TestClient(app)`: A test client provided by FastAPI for making requests to the application in tests without running a live server.
-   `@pytest.fixture(autouse=True)`: A pytest fixture that sets up and tears down the database for each test function, ensuring a clean state for every test.

2. Run tests:
   ```bash
   pytest test_main.py
   ```

**Expected Result**:
```
===================================== test session starts ======================================
collected 3 items

test_main.py ...                                                                [100%]

===================================== 3 passed in 2.30s ======================================
```

**Pro Tip**: Run tests in CI/CD (e.g., GitHub Actions) before deployment.

---

## Step 7: Deploy to Render

**Instructions**:
1. Create a Render account at [render.com](https://render.com).
2. Set up a new Web Service in Render:
   - Repository: Push your project to a GitHub repository.
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT --workers $WORKERS`
   - Environment Variables:
     - `DATABASE_URL`: Obtain from Render’s PostgreSQL service (e.g., `postgresql://user:password@host:port/dbname`).
     - `WORKERS`: Set to `4` (adjust based on instance size).
3. Create a `render.yaml` for blueprint deployment:
```yaml
services:
  - type: web
    name: ai-deployment-service
    env: docker
    repo: https://github.com/your-username/your-repo
    plan: standard
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: ai-deployment-db
          property: connectionString
      - key: WORKERS
        value: 4
    healthCheckPath: /health
databases:
  - name: ai-deployment-db
    databaseName: ai_deployment
    plan: standard
```

**Key Code Explained**:
-   `services`: Defines the web service to be deployed on Render.
-   `env: docker`: Specifies that the service should be built and run from a Dockerfile.
-   `envVars`: Defines environment variables. `DATABASE_URL` is dynamically pulled from a Render PostgreSQL database.
-   `databases`: Defines a managed PostgreSQL database instance to be created on Render.
-   `healthCheckPath: /health`: Tells Render to use the `/health` endpoint to check if the service is running correctly.

4. Deploy using Render CLI:
   ```bash
   render deploy
   ```

5. Add a `/health` endpoint for Render’s health checks:
   Update `main.py` by adding:
   ```python
   @app.get("/health")
   async def health_check():
       return {"status": "healthy"}
   ```

6. Verify deployment at the Render-provided URL (e.g., `https://your-service.onrender.com/docs`).

**Try It Out**:
Test the deployed service:
```bash
curl -X POST "https://your-service.onrender.com/generate" -H "Content-Type: application/json" -d '{"prompt": "The future of AI is", "max_length": 50}'
```

**Expected Result**:
```json
{
  "id": 1,
  "prompt": "The future of AI is",
  "generated_text": "The future of AI is bright, with machines creating innovative solutions...",
  "model_name": "distilgpt2",
  "created_at": "2025-07-13T16:45:00"
}
```

**Pro Tip**: Render automatically enables HTTPS, ensuring secure connections.

---

## Step 8: Production Considerations


**Instructions**:
1. **Environment Variables**: Store `DATABASE_URL` and `WORKERS` securely in Render’s dashboard.
2. **Logging**: The `main.py` logging setup helps debug issues in production.
3. **Scaling**: Adjust `WORKERS` based on Render instance size (e.g., 4 for Standard plan).
4. **Monitoring**: Enable Render’s metrics or integrate Prometheus for performance tracking.

**Taste Test**:
- Logs track database operations and inference errors.
- Health checks ensure Render restarts unhealthy instances.

---

## Tips
- **Optimize Model Size**: Use `distilgpt2` for faster deployment; consider xAI’s Grok API (https://x.ai/api) for production-grade model hosting.
- **Database Choice**: PostgreSQL is ideal for production; ensure indexes on `generation_records.created_at`.
- **CI/CD**: Automate deployment with GitHub Actions and run tests before pushing to Render.
- **Cost Management**: Monitor Render usage to avoid unexpected costs; scale workers based on traffic.
- **Security**: Add authentication for production APIs.

---

## Project Structure
```
ai-deployment-service/
├── main.py
├── models.py
├── database.py
├── test_main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── render.yaml
├── ai_deployment.db (local SQLite, not used in production)
└── venv/
```

---

