# Real-Time Communication with Generative Models



## Background
Real-time communication is essential for interactive applications like chatbots, live content generation, or collaborative AI tools. FastAPI supports WebSockets, which allow bidirectional, persistent connections between clients and servers—perfect for streaming AI model outputs or handling back-and-forth conversations. This project will  focuses on integrating a generative AI model (e.g., Hugging Face’s GPT-2) with FastAPI’s WebSocket capabilities to enable real-time text generation. This project will use WebSockets for a chat-like interface, where users send prompts and receive AI-generated responses in real time. This guide assumes familiarity with Python, FastAPI, and AI model integration. This project will also leverage Pydantic for data validation and `concurrent.futures` for efficient model inference.

**Why Real-Time Matters**:
- **Interactivity**: Enables dynamic, chat-like AI interactions.
- **Low Latency**: WebSockets reduce overhead compared to repeated HTTP requests.
- **Scalability**: Handle multiple clients concurrently with async programming.
- **Real-World Use**: Power chatbots, live assistants, or interactive content generators.

---

## Real-Time AI Communication with FastAPI WebSockets

**Objective**: Build a FastAPI service with WebSocket support to enable real-time text generation using a generative AI model (GPT-2).

**Libs**:
- Python 3.9+ (the base for our recipe)
- FastAPI (for WebSocket and API framework)
- Uvicorn (ASGI server with WebSocket support)
- Hugging Face Transformers (for the AI model)
- Torch (for model computation)
- Pydantic (for data validation)
- `concurrent.futures` (for thread-based inference)
- Optional: Pytest and `websockets` (for testing)

**Tools**:
- Terminal or command line
- Code editor (e.g., VS Code)
- Virtual environment (to keep dependencies clean)
- Browser or WebSocket client (e.g., JavaScript or Python client)


---

## Step 1: Project Environment

**Instructions**:
1. Create a project directory:
   ```bash
   mkdir ai-realtime-service
   cd ai-realtime-service
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install fastapi uvicorn pydantic transformers torch pytest websockets
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
torch==2.7.1
pytest==8.3.2
websockets==13.0.1
```

**Pro Tip**: The `websockets` package is included for testing WebSocket connections.

---

## Step 2: Define Pydantic Models

**Instructions**:
1. Create a `models.py` file.
2. Define models for WebSocket messages and AI responses.

**Code Example**:
```python
# models.py

from pydantic import BaseModel, Field

class WebSocketMessage(BaseModel):
    prompt: str
    max_length: int = Field(default=50, gt=10, le=250)

class WebSocketResponse(BaseModel):
    generated_text: str
    source_model: str
    client_id: str
```

**Taste Test**:
- `WebSocketMessage` validates incoming WebSocket messages with a prompt and `max_length`.
- `WebSocketResponse` ensures responses include generated text, model name, and client ID for tracking.

---

## Step 3: WebSocket Endpoint with AI Integration

WebSockets enable real-time, bidirectional communication, and integrating the AI model ensures dynamic responses.

**Instructions**:
1. Create a `main.py` file.
2. Set up a FastAPI app with a WebSocket endpoint that handles AI inference in a thread pool for concurrency.

**Code Example**:
```python
# main.py

import asyncio
import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import pipeline, Pipeline
from models import WebSocketMessage, WebSocketResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FORBIDDEN_WORDS = ["hate", "violence"]


def run_model_inference(
    generator_pipeline: Pipeline, prompt: str, max_length: int
) -> str:
    logger.info(f"Running inference for prompt: {prompt[:30]}...")
    result = generator_pipeline(prompt, max_length=max_length, num_return_sequences=1)
    logger.info("Inference complete.")
    return result[0]["generated_text"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing resources...")
    app.state.executor = ThreadPoolExecutor(max_workers=4)
    loop = asyncio.get_running_loop()
    app.state.generator = await loop.run_in_executor(
        None, pipeline, "text-generation", "gpt2"
    )
    logger.info("Resources initialized successfully.")
    yield
    logger.info("Application shutdown: Cleaning up resources...")
    app.state.executor.shutdown(wait=True)
    logger.info("Resources cleaned up successfully.")


app = FastAPI(title="Real-Time AI Service", version="1.0.0", lifespan=lifespan)


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    logger.info(f"Client {client_id} connected.")

    loop = asyncio.get_running_loop()
    executor = websocket.app.state.executor
    generator_pipeline = websocket.app.state.generator

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = WebSocketMessage.parse_raw(data)
            except Exception as e:
                await websocket.send_json({"error": f"Invalid message format: {e}"})
                continue

            if any(word in message.prompt.lower() for word in FORBIDDEN_WORDS):
                await websocket.send_json(
                    {"error": "Prompt contains inappropriate content"}
                )
                continue

            try:
                generated_text = await loop.run_in_executor(
                    executor,
                    run_model_inference,
                    generator_pipeline,
                    message.prompt,
                    message.max_length,
                )

                response = WebSocketResponse(
                    generated_text=generated_text,
                    source_model="gpt2",
                    client_id=client_id,
                )

                await websocket.send_json(response.model_dump())
                logger.info(
                    f"Client {client_id} received response for prompt: {message.prompt[:20]}..."
                )
            except Exception as e:
                error_message = f"Model inference failed: {e}"
                await websocket.send_json({"error": error_message})
                logger.error(f"Client {client_id} inference failed: {e}")

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected.")
    except Exception as e:
        error_message = f"An unexpected server error occurred: {e}"
        logger.error(f"Client {client_id} error: {e}")
    finally:
        if websocket.client_state != "DISCONNECTED":
            await websocket.close()
```

- The `/ws/generate` WebSocket endpoint accepts connections and processes messages in a loop.
- `WebSocketMessage.parse_raw` validates incoming JSON messages.
- `run_in_executor` offloads CPU-bound inference to a thread pool, keeping the async loop free.
- The client receives responses or errors as JSON over the WebSocket.

---

## Step 4: Run the FastAPI Server

Running the server lets you test real-time communication.

**Instructions**:
1. Start the server with Uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. Test the WebSocket using a client (see Step 5 for a sample client).

**Pro Tip**: Uvicorn supports WebSockets out of the box, making it ideal for real-time apps.

---

## Step 5: WebSocket Client Example

A client lets you interact with the WebSocket endpoint to ensure it’s perfect.

**Instructions**:
1. Create a `client.py` file to test the WebSocket connection.
2. Use the `websockets` library to send messages and receive responses.

**Code Example**:
```python
import asyncio
import websockets
import json
from models import WebSocketMessage

async def test_websocket():
    uri = "ws://localhost:8000/ws/generate"
    async with websockets.connect(uri) as websocket:
        # Send a valid message
        message = WebSocketMessage(
            prompt="The future of AI is",
            max_length=50
        )
        await websocket.send(json.dumps(message.dict()))
        response = await websocket.recv()
        print(f"Received: {response}")
        
        # Send an invalid message
        await websocket.send(json.dumps({"prompt": "I hate this"}))
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(test_websocket())
```

3. Run the client (with the server running):
   ```bash
   python client.py
   ```

**Expected Output**:
```
Received: {"generated_text":"The future of AI is bright, with machines creating innovative solutions...","source_model":"gpt2","client_id":"unique-client-id"}
Received: {"error":"Prompt contains inappropriate content"}
```

**Tip**: Use a browser-based WebSocket client (e.g., a JavaScript-based HTML page) for a more interactive test.

---

## Step 6: Content Filtering and Logging

Content filtering and logging ensure your service is safe and debuggable.

**Instructions**:
1. The `main.py` above already includes content filtering (`FORBIDDEN_WORDS`) and logging.
2. Enhance logging to track WebSocket connections and disconnections.

**Code Example** (Already included in `main.py`):
- `FORBIDDEN_WORDS` rejects prompts with “hate” or “violence.”
- Logging tracks client connections, inference, and errors with client IDs.

**Taste Test**:
- Invalid prompts return an error message over the WebSocket.
- Logs help debug issues in real-time interactions.

---

## Step 7: Test Real-Time Communication


**Instructions**:
1. Create a `test_main.py` file to test the WebSocket endpoint.
2. Use `pytest` and `websockets` for async testing.

**Code Example**:
```python
import pytest
import websockets
import json
from models import WebSocketMessage

@pytest.mark.asyncio
async def test_websocket_generate():
    uri = "ws://localhost:8000/ws/generate"
    async with websockets.connect(uri) as websocket:
        # Test valid message
        message = WebSocketMessage(
            prompt="Test prompt",
            max_length=50
        )
        await websocket.send(json.dumps(message.dict()))
        response = json.loads(await websocket.recv())
        assert "generated_text" in response
        assert response["source_model"] == "gpt2"
        assert "client_id" in response
        
        # Test forbidden words
        await websocket.send(json.dumps({"prompt": "I hate this", "max_length": 50}))
        response = json.loads(await websocket.recv())
        assert "error" in response
        assert "inappropriate content" in response["error"]

@pytest.mark.asyncio
async def test_websocket_invalid_format():
    uri = "ws://localhost:8000/ws/generate"
    async with websockets.connect(uri) as websocket:
        await websocket.send("invalid json")
        response = json.loads(await websocket.recv())
        assert "error" in response
        assert "Invalid message format" in response["error"]
```

3. Run the tests (with the server running):
   ```bash
   pytest test_main.py
   ```

**Expected Result**:
```
===================================== test session starts ======================================

collected 2 items

test_main.py ..                                                                 [100%]

===================================== 2 passed in 2.20s ======================================
```

**Pro Tip**: Tests verify valid responses, content filtering, and error handling for malformed messages.

---

## Step 8: Deploy the Service


**Instructions**:
1. Create a `Procfile` for deployment (e.g., on Render):
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4
```

2. Deploy to a platform like Render, ensuring `requirements.txt` is included.
3. Configure environment variables for production settings.

**Command** (Render CLI example):
```bash
render deploy
```

**Pro Tip**: Use multiple Uvicorn workers (`--workers 4`) to handle concurrent WebSocket connections.

---

## Tips
- **Optimize Model Loading**: Load the model once at startup to reduce latency (done in `main.py`).
- **Scale with Smaller Models**: Use `distilgpt2` for faster inference during development.
- **Cloud APIs**: For production, consider API to offload model hosting.
- **Monitor WebSockets**: Use tools like Prometheus to track connection counts and latency.
- **Secure Connections**: Enable WSS (secure WebSockets) in production with SSL/TLS.

---

## Project Structure
```
ai-realtime-service/
├── main.py
├── models.py
├── client.py
├── test_main.py
├── requirements.txt
├── Procfile
└── venv/
```

---
