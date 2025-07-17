# Securing AI Services


## Background for Developers
AI services, especially those handling user inputs or generating sensitive outputs, are prime targets for attacks like injection, data leaks, or denial-of-service (DoS). This project will focuses on securing a FastAPI-based AI service using techniques like input validation, rate limiting, HTTPS, and secure headers, building on authentication/authorization concepts from earlier chapters. We’ll integrate a generative AI model (Hugging Face’s GPT-2) with FastAPI, secure it with rate limiting (`slowapi`), input sanitization, and secure headers, and store data in SQLite for persistence. This guide assumes familiarity with Python, FastAPI, SQLAlchemy, and basic authentication. We’ll use `python-jose`, `passlib`, and `slowapi` to enhance security.

**Why Securing AI Services Matters**:
- **Data Protection**: Safeguard user inputs and AI outputs from leaks or tampering.
- **Availability**: Prevent abuse (e.g., DoS attacks) with rate limiting.
- **Compliance**: Meet security standards for production-grade APIs.
- **Real-World Use**: Secure chatbots, content generators, or AI APIs handling sensitive data.


---

## Securing a FastAPI AI Service

**Objective**: Build a FastAPI service that integrates a generative AI model (GPT-2), secures it with JWT authentication, rate limiting, input sanitization, secure headers, and self-hosted documentation.

**Ingredients**:
- Python 3.9+ (the base for our recipe)
- FastAPI (for the API framework)
- Uvicorn (ASGI server)
- Hugging Face Transformers (for the AI model)
- Torch (for model computation)
- Pydantic (for data validation)
- SQLAlchemy (for database ORM)
- `python-jose[cryptography]` (for JWT handling)
- `passlib[bcrypt]` (for password hashing)
- `slowapi` (for rate limiting)
- `python-multipart` (for form data)
- Optional: Pytest and `httpx` (for testing)

**Tools**:
- Terminal or command line
- Code editor (e.g., VS Code)
- Virtual environment (to keep dependencies clean)
- SQLite (built into Python)


---

## Step 1: Project Environment


**Instructions**:
1. Create a project directory:
   ```bash
   mkdir 7_ai-secure-service
   cd 7_ai-secure-service
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install fastapi uvicorn pydantic transformers torch sqlalchemy "python-jose[cryptography]" "passlib[bcrypt]" slowapi pytest httpx python-multipart
   ```

4. Save dependencies to a `requirements.txt` file:
   ```bash
   pip freeze > requirements.txt
   ```

**What You Get** (`requirements.txt`):
```
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2
transformers==4.44.2
torch==2.6.0
sqlalchemy==2.0.35
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
slowapi==0.1.9
pytest==8.3.2
httpx==0.27.2
python-multipart==0.0.9
```

**Pro Tip**: The `slowapi` package adds rate limiting to prevent abuse, and `python-multipart` is needed for OAuth2 form data.

---

## Step 2: Define Pydantic and SQLAlchemy Models

Pydantic models validate API inputs/outputs, while SQLAlchemy models define the database schema.

**Instructions**:
1. Create a `models.py` file.
2. Define Pydantic models for authentication and AI requests, plus SQLAlchemy models for users and generation records.

**Code Example** (`models.py`):
```python
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# SQLAlchemy Base
Base = declarative_base()

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

class GenerationRecord(Base):
    __tablename__ = "generation_records"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    prompt = Column(String, nullable=False)
    generated_text = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="User's username")
    password: str = Field(..., min_length=6, description="User's password")

class UserResponse(BaseModel):
    id: int
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Input prompt for text generation")
    max_length: Optional[int] = Field(50, ge=10, le=200, description="Maximum length of generated text")

class TextGenerationResponse(BaseModel):
    id: int = Field(..., description="Database record ID")
    generated_text: str = Field(..., description="Text generated by the AI model")
    model_name: str = Field(..., description="Name of the AI model used")
    created_at: str = Field(..., description="Timestamp of record creation")
```


- `User` stores usernames and hashed passwords.
- `GenerationRecord` tracks AI interactions with user metadata.
- Pydantic models enforce strict validation for API inputs/outputs.

---

## Step 3: Database and Auth Configuration


**Instructions**:
1. Create a `database.py` file for SQLAlchemy setup.
2. Create an `auth.py` file for JWT handling and password hashing.

**Code Example (Database)** (`database.py`):
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

DATABASE_URL = "sqlite:///ai_secure.db"
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

**Code Example (Auth)** (`auth.py`):
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from models import User, UserResponse
from database import get_db
from datetime import datetime, timedelta
import os

SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key")  # Set in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user
```


- `database.py` sets up SQLite (`ai_secure.db`) for user and record storage.
- `auth.py` handles JWT creation, password hashing, and user authentication.

---

## Step 4: FastAPI with Self-Hosted Docs

We'll now modify the main application to serve Swagger UI documentation from local files instead of a CDN. This eliminates reliance on external servers, enhancing security and ensuring the app works offline.

**Instructions**:
1. Update `main.py` to disable default docs, serve static files, and add a custom docs endpoint.
2. The Content Security Policy (CSP) is updated to allow `unsafe-inline` scripts and styles, which is now secure because the files are served locally.

**Code Example** (`main.py`):
```python
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from models import Base, User, UserCreate, UserResponse, TextGenerationRequest, TextGenerationResponse, Token, GenerationRecord
from database import get_db, engine
from auth import verify_password, get_password_hash, create_access_token, get_current_user
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import re
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html

Base.metadata.create_all(bind=engine)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Secure AI Service",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline';"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

generator = pipeline("text-generation", model="gpt2")
executor = ThreadPoolExecutor(max_workers=4)
FORBIDDEN_WORDS = ["hate", "violence"]
SANITIZE_PATTERN = re.compile(r'[<>{};]')

def sanitize_input(text: str) -> str:
    return SANITIZE_PATTERN.sub("", text)

def run_model_inference(prompt: str, max_length: int) -> str:
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]["generated_text"]

@app.post("/users", response_model=UserResponse)
@limiter.limit("5/minute")
async def create_user(request: Request, user: UserCreate, db: Session = Depends(get_db)):
    # ... (user creation logic remains the same)
    if SANITIZE_PATTERN.search(user.username):
        raise HTTPException(status_code=400, detail="Invalid characters in username")
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"Created user: {user.username}")
    return UserResponse(id=db_user.id, username=db_user.username)


@app.post("/token", response_model=Token)
@limiter.limit("10/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # ... (login logic remains the same)
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    logger.info(f"User {user.username} logged in")
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/generate", response_model=TextGenerationResponse)
@limiter.limit("5/minute")
async def generate_text(
    req: TextGenerationRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # ... (generation logic remains the same)
    sanitized_prompt = sanitize_input(req.prompt)
    if any(word in sanitized_prompt.lower() for word in FORBIDDEN_WORDS):
        logger.warning(f"User {user.username} sent forbidden prompt: {sanitized_prompt}")
        raise HTTPException(status_code=400, detail="Prompt contains inappropriate content")
    
    try:
        loop = asyncio.get_event_loop()
        generated_text = await loop.run_in_executor(
            executor, run_model_inference, sanitized_prompt, req.max_length
        )
        record = GenerationRecord(
            username=user.username,
            prompt=sanitized_prompt,
            generated_text=generated_text,
            model_name="gpt2"
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(f"Generated text for user {user.username}, record ID {record.id}")
        return TextGenerationResponse(
            id=record.id,
            generated_text=generated_text,
            model_name="gpt2",
            created_at=record.created_at.isoformat()
        )
    except Exception as e:
        logger.error(f"Inference failed for user {user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")


@app.get("/history", response_model=list[TextGenerationResponse])
async def get_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # ... (history logic remains the same)
    records = db.query(GenerationRecord).filter(GenerationRecord.username == user.username).all()
    logger.info(f"User {user.username} accessed history")
    return [
        TextGenerationResponse(
            id=r.id,
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

---

## Step 5: Run the FastAPI Server

Running the server lets you test security features.

**Instructions**:
1.  **Download Swagger UI static files**:
    *   Create a `static` directory in your project root.
    *   Download `swagger-ui-bundle.js` from [here](https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js) and place it in the `static` directory.
    *   Download `swagger-ui.css` from [here](https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css) and place it in the `static` directory.

2.  Set the `SECRET_KEY` environment variable:
    ```bash
    export SECRET_KEY="your-secure-secret-key"  # Windows: set SECRET_KEY=your-secure-secret-key
    ```

3.  Start the server with Uvicorn:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

4.  Visit `http://127.0.0.1:8000/docs` for your self-hosted, secure Swagger UI.

**Try It Out**:
1.  Register a user:
    ```bash
    curl -X POST "http://127.0.0.1:8000/users" -H "Content-Type: application/json" -d '{"username": "testuser", "password": "securepassword"}'
    ```

2.  Get a token:
    ```bash
    curl -X POST "http://127.0.0.1:8000/token" -H "Content-Type: application/x-www-form-urlencoded" -d "username=testuser&password=securepassword"
    ```

3.  Generate text (using your token):
    ```bash
    curl -X POST "http://127.0.0.1:8000/generate" -H "Authorization: Bearer <your-token>" -H "Content-Type: application/json" -d '{"prompt": "The future of AI is", "max_length": 50}'
    ```

---

## Step 6: Test the Service


**Instructions**:
1. Create a `test_main.py` file to test the API.
2. Use `pytest` and `httpx` for HTTP testing.

**Code Example** (`test_main.py`):
```python
import pytest
from fastapi.testclient import TestClient
from main import app
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, User
from database import get_db

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables before tests
Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    # Run tests, then cleanup
    yield
    os.remove("./test.db")

def test_create_user_and_generate():
    # Create user
    response = client.post("/users", json={"username": "testuser", "password": "securepassword"})
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
    data = response.json()
    assert "id" in data
    assert "generated_text" in data
    
    # Check history
    response = client.get("/history", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert len(response.json()) > 0

def test_unauthorized_access():
    response = client.post("/generate", json={"prompt": "Test prompt", "max_length": 50})
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]

def test_docs_redirect():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text
```

3. Run the tests:
   ```bash
   pytest test_main.py
   ```

---

## Step 7: Deploy the Service


**Instructions**:
1. Create a `Procfile` for deployment (e.g., on Render):
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

2. Deploy to a platform like Render, ensuring `requirements.txt` and the `static/` directory are included.
3. Set the `SECRET_KEY` environment variable in your deployment environment.

---

## Tips
- **Strong Secrets**: Use a random, 32+ character `SECRET_KEY` for JWTs.
- **Database Security**: Switch to PostgreSQL for production and enable SSL.
- **Rate Limiting**: Adjust `slowapi` limits based on usage patterns.
- **Monitor Threats**: Use Prometheus or a WAF (e.g., Cloudflare) to detect and block attacks.

---

## Project Structure
```
7_ai-secure-service/
├── static/
│   ├── swagger-ui-bundle.js
│   └── swagger-ui.css
├── main.py
├── models.py
├── database.py
├── auth.py
├── test_main.py
├── requirements.txt
├── Procfile
├── ai_secure.db
└── venv/
```

---

