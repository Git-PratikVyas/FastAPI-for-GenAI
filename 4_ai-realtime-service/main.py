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

                # Create the response with the CORRECT field name
                response = WebSocketResponse(
                    generated_text=generated_text,
                    source_model="gpt2",
                    client_id=client_id,
                )

                # Use .model_dump() for Pydantic v2
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
