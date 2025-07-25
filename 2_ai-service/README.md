# Building Type-Safe AI Services with FastAPI


## Type-Safe AI Service with FastAPI

**Objective**: Build a FastAPI service that generates text using a Hugging Face model, with Pydantic for type safety and Uvicorn for serving.

**Libs**:
- Python 3.9+
- FastAPI (for the API framework)
- Pydantic (for type-safe models)
- Hugging Face Transformers (for the AI model)
- Uvicorn (for running the server)
- Optional: Pytest (for testing)

**Tools**:
- Terminal or command line
- Code editor (e.g., VS Code)
- Virtual environment (to keep things tidy)


---

### Request Flow Diagram

The following diagram illustrates the detailed flow of a request through each component of the system:

```mermaid
sequenceDiagram
    participant Client
    participant RateLimiter
    participant FastAPI
    participant PydanticValidator
    participant ContentFilter
    participant TextGenerator
    participant HuggingFaceModel
    participant Logger
    
    Client->>FastAPI: POST /generate with JSON payload
    FastAPI->>Logger: Log incoming request
    FastAPI->>RateLimiter: Check rate limit
    
    alt Rate limit exceeded
        RateLimiter->>FastAPI: 429 Too Many Requests
        FastAPI->>Logger: Log rate limit exceeded
        FastAPI->>Client: Return 429 error
    else Rate limit OK
        RateLimiter->>FastAPI: Proceed
        FastAPI->>PydanticValidator: Validate request body
        
        alt Invalid request
            PydanticValidator->>FastAPI: Validation error
            FastAPI->>Logger: Log validation error
            FastAPI->>Client: Return 422 error
        else Valid request
            PydanticValidator->>FastAPI: Validated TextGenerationRequest
            FastAPI->>ContentFilter: Check for inappropriate content
            
            alt Contains inappropriate content
                ContentFilter->>FastAPI: Content violation
                FastAPI->>Logger: Log content violation
                FastAPI->>Client: Return 400 error
            else Content OK
                ContentFilter->>FastAPI: Content approved
                FastAPI->>TextGenerator: Generate text with prompt
                TextGenerator->>HuggingFaceModel: Request text generation
                HuggingFaceModel->>TextGenerator: Return generated text
                
                alt Generation error
                    TextGenerator->>FastAPI: Generation error
                    FastAPI->>Logger: Log generation error
                    FastAPI->>Client: Return 500 error
                else Generation success
                    TextGenerator->>FastAPI: Generated text and model name
                    FastAPI->>PydanticValidator: Create and validate TextGenerationResponse
                    PydanticValidator->>FastAPI: Validated response
                    FastAPI->>Logger: Log successful generation
                    FastAPI->>Client: Return 200 with response
                end
            end
        end
    end
```

**Key Component**

- **Client**: External system or user that sends HTTP requests to the API.
- **RateLimiter**: Controls request frequency using client IP address, limiting to 5 requests per minute.
- **FastAPI**: Web framework that handles routing, request parsing, and response formatting.
- **PydanticValidator**: Validates request/response data against defined schemas with type checking.
- **ContentFilter**: Screens input prompts for inappropriate content using a list of forbidden words.
- **TextGenerator**: Orchestrates the text generation process and handles errors.
- **HuggingFaceModel**: Pre-trained GPT-2 model that performs the actual text generation.
- **Logger**: Records system events, requests, responses, and errors for monitoring and debugging.


---

## Step 1: Project Environment



**Instructions**:
1. Create a project directory:
   ```bash
   mkdir ai-service
   cd ai-service
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install the core ingredients:
   ```bash
   pip install fastapi uvicorn pydantic transformers torch
   ```

4. Save your dependencies:
   ```bash
   pip freeze > requirements.txt
   ```

**What You Get**:
A `requirements.txt` file like:

fastapi==0.116.1
uvicorn==0.35.0
pydantic==2.11.7
transformers==4.53.2
torch==2.7.1
slowapi==0.1.9


---

## Step 2: Define Pydantic Models

Pydantic models are ensure inputs and outputs are exactly what you expect, catching errors early.

**Instructions**:
1. Create a `models.py` file.
2. Define request and response models for text generation with validation rules.

**Code Example**:
```python
from pydantic import BaseModel, Field
from typing import Optional

class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Input prompt for text generation")
    max_length: Optional[int] = Field(50, ge=10, le=200, description="Maximum length of generated text")

class TextGenerationResponse(BaseModel):
    generated_text: str = Field(..., description="Text generated by the AI model")
    model_name: str = Field(..., description="Name of the AI model used")
```

**Taste Test**:
- `TextGenerationRequest` ensures the prompt is non-empty and under 500 characters, with an optional `max_length` between 10 and 200.
- `TextGenerationResponse` guarantees the response includes generated text and the model name.

---

## Step 3: Set Up FastAPI

FastAPI is bakes your API, combining speed, type safety, and automatic documentation.

**Instructions**:
1. Create a `main.py` file.
2. Set up a FastAPI app with an endpoint to generate text using a Hugging Face model.

**Code Example**:
```python
from fastapi import FastAPI, HTTPException, Request
from models import TextGenerationRequest, TextGenerationResponse
from transformers import pipeline

app = FastAPI(title="Type-Safe AI Service", version="1.0.0")

# Load the AI model (GPT-2 for text generation)
generator = pipeline("text-generation", model="gpt2")

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(body: TextGenerationRequest, request: Request):
    try:
        # Generate text
        result = generator(body.prompt, max_length=body.max_length, num_return_sequences=1)
        generated_text = result[0]["generated_text"]
        
        return TextGenerationResponse(
            generated_text=generated_text,
            model_name="gpt2"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")
```


- The `pipeline("text-generation", model="gpt2")` loads a pre-trained GPT-2 model.
- The `/generate` endpoint validates the request with `TextGenerationRequest` and returns a `TextGenerationResponse`.
- Error handling ensures the API doesn’t crash on invalid inputs.

---

## Step 4: Run the FastAPI Server

Running the server lets you taste-test your API and explore its interactive docs.

**Instructions**:
1. Start the server with Uvicorn:
   ```bash
   uvicorn main:app --reload
   ```

2. Visit `http://127.0.0.1:8000/docs` for Swagger UI to test the API.

**Try It Out**:
Send a test request using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Once upon a time", "max_length": 50}'
```

**Expected Dish**:
```json
{
  "generated_text": "Once upon a time, in a faraway land, a brave knight set out on a quest...",
  "model_name": "gpt2"
}
```

---

## Step 5: Error Handling and Validation

Proper error handling makes your service robust and user-friendly.

**Instructions**:
1. Update `main.py` to include a content filter for inappropriate prompts.
2. Use a simple list of forbidden words for validation.

**Code Example** (Updated `main.py`):
```python
from fastapi import FastAPI, HTTPException
from models import TextGenerationRequest, TextGenerationResponse
from transformers import pipeline

app = FastAPI(title="Type-Safe AI Service", version="1.0.0")
generator = pipeline("text-generation", model="gpt2")
FORBIDDEN_WORDS = ["hate", "violence"]

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(body: TextGenerationRequest, request: Request):
    # Check for inappropriate content
    if any(word in body.prompt.lower() for word in FORBIDDEN_WORDS):
        raise HTTPException(status_code=400, detail="Prompt contains inappropriate content")
    
    try:
        result = generator(body.prompt, max_length=body.max_length, num_return_sequences=1)
        generated_text = result[0]["generated_text"]
        
        return TextGenerationResponse(
            generated_text=generated_text,
            model_name="gpt2"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")
```

**Taste Test**:
- If the prompt contains “hate” or “violence,” the API returns a 400 error.
- Pydantic ensures `prompt` isn’t empty, and `max_length` is within bounds.

---

## Step 6: Test the Service

Testing to ensures everything works as expected.

**Instructions**:
1. Install testing tools:
   ```bash
   pip install pytest httpx
   ```

2. Create a `test_main.py` file to test the API.

**Code Example**:
```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_text_success():
    response = client.post("/generate", json={"prompt": "Hello world", "max_length": 50})
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
```

3. Run the tests:
   ```bash
   pytest test_main.py
   ```

**Expected Result**:
```
===================================== test session starts ======================================
collected 3 items

test_main.py ...                                                                 [100%]

===================================== 3 passed in 0.12s ======================================
```

---

## Step 7: Deploy the Service

**Why**: Deployment is like serving your dish to the world—make it accessible!

**Instructions**:
1. Create a `Procfile` for deployment (e.g., on Render):

web: uvicorn main:app --host 0.0.0.0 --port $PORT


2. Deploy to Render or another platform, ensuring `requirements.txt` is included.
3. Use environment variables for sensitive data (e.g., model paths).

**Command** (Render CLI example):
```bash
render deploy
```

---

## Step 8: Logging and Rate Limiting

**Why**: Logging and rate limiting are the finishing touches for a production-ready service, like a sprinkle of herbs.

**Instructions**:
1. Install `slowapi` for rate limiting:
   ```bash
   pip install slowapi
   ```

2. Update `main.py` with logging and rate limiting.

**Code Example** (Updated `main.py`):
```python
from fastapi import FastAPI, HTTPException, Request
from models import TextGenerationRequest, TextGenerationResponse
from transformers import pipeline
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Type-Safe AI Service", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

generator = pipeline("text-generation", model="gpt2")
FORBIDDEN_WORDS = ["hate", "violence"]


@app.post("/generate", response_model=TextGenerationResponse)
@limiter.limit("5/minute")  # 5 requests per minute per client
async def generate_text(body: TextGenerationRequest, request: Request):
    logger.info(f"Received request with prompt: {body.prompt}")

    if any(word in body.prompt.lower() for word in FORBIDDEN_WORDS):
        logger.warning(f"Inappropriate prompt detected: {body.prompt}")
        raise HTTPException(
            status_code=400, detail="Prompt contains inappropriate content"
        )

    try:
        result = generator(
            body.prompt, max_length=body.max_length, num_return_sequences=1
        )
        generated_text = result[0]["generated_text"]
        logger.info("Text generation successful")

        return TextGenerationResponse(generated_text=generated_text, model_name="gpt2")
    except Exception as e:
        logger.error(f"Text generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")
```


- Logging tracks requests, warnings, and errors for debugging.
- Rate limiting (5 requests/minute) prevents abuse.
- The `@limiter.limit` decorator enforces the rate limit per client IP.

---

## Tips
- **Optimize Performance**: Load the AI model once at startup to reduce latency.
- **Secure Secrets**: Use `python-dotenv` to manage API keys or model paths in a `.env` file.
- **Scale Up**: For production, consider a more efficient model or an API
- **Monitor**: Use tools like Prometheus to track API performance.

---

## Project Structure
```
ai-service/
├── main.py
├── models.py
├── test_main.py
├── requirements.txt
├── Procfile
└── venv/
```

---
