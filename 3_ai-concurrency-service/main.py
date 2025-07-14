# # main.py




import asyncio
import uuid
import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from transformers import pipeline


from models import TextGenerationRequest, TextGenerationResponse


# --- 1. Lifespan Manager for Startup/Shutdown Logic ---
# This is the modern replacement for @app.on_event decorators.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("FastAPI service is starting up.")
    # The 'generator' and 'executor' are already initialized in the global scope.
    yield
    # Code to run on shutdown
    logger.info("Shutting down thread pool...")
    executor.shutdown(wait=True)
    logger.info("Thread pool shutdown complete.")


# --- 2. Application Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pass the lifespan manager to the FastAPI app instance.
app = FastAPI(
    title="Concurrent AI Service",
    version="1.0.0",
    lifespan=lifespan
)

# Initialize heavy objects once at startup.
# The generator is loaded into memory when the Python script is first run.
logger.info("Loading text-generation model...")
generator = pipeline("text-generation", model="gpt2")
logger.info("Model loaded successfully.")

# A thread pool to run our blocking ML model without blocking the event loop.
executor = ThreadPoolExecutor(max_workers=4)

FORBIDDEN_WORDS = ["hate", "violence"]


# --- 3. Core Functions ---
def run_model_inference(prompt: str, max_length: int) -> str:
    """
    This is a synchronous and CPU-bound function that runs the model.
    It will be executed in the thread pool.
    """
    logger.info(f"Starting model inference in a separate thread for prompt: '{prompt[:30]}...'")
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    logger.info("Model inference completed in thread.")
    return result[0]["generated_text"]

def background_inference(prompt: str, max_length: int, request_id: str):
    """A wrapper to run inference for a background task and handle logging."""
    try:
        # This function is also blocking, but FastAPI's BackgroundTasks
        # runs it in a thread pool automatically.
        result = run_model_inference(prompt, max_length)
        logger.info(f"Background task {request_id} completed: {result[:50]}...")
    except Exception as e:
        logger.error(f"Background task {request_id} failed: {str(e)}")


# --- 4. API Endpoints ---
@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generates text and waits for the result before responding."""
    request_id = str(uuid.uuid4())
    logger.info(f"Processing request {request_id} with prompt: {request.prompt}")
    
    if any(word in request.prompt.lower() for word in FORBIDDEN_WORDS):
        logger.warning(f"Request {request_id} rejected: Inappropriate content")
        raise HTTPException(status_code=400, detail="Prompt contains inappropriate content")
    
    try:
        # KEY CHANGE: Get the current running event loop directly.
        # No need to store it on app.state.
        loop = asyncio.get_running_loop()

        # Await the result from the thread pool executor.
        # This frees up the main thread to handle other requests while waiting.
        generated_text = await loop.run_in_executor(
            executor, run_model_inference, request.prompt, request.max_length
        )

        logger.info(f"Request {request_id} completed successfully")
        return TextGenerationResponse(
            generated_text=generated_text,
            model=generator.model.config.model_type,
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

@app.post("/generate_background", response_model=dict)
async def generate_text_background(request: TextGenerationRequest, background_tasks: BackgroundTasks):
    """Queues the text generation task to run in the background and responds immediately."""
    request_id = str(uuid.uuid4())
    logger.info(f"Queuing background task {request_id} with prompt: {request.prompt}")
    
    if any(word in request.prompt.lower() for word in FORBIDDEN_WORDS):
        logger.warning(f"Background task {request_id} rejected: Inappropriate content")
        raise HTTPException(status_code=400, detail="Prompt contains inappropriate content")
    
    # Add the blocking function to run in the background.
    # FastAPI handles the thread management for this.
    background_tasks.add_task(background_inference, request.prompt, request.max_length, request_id)
    
    return {"message": "Task queued for background processing", "request_id": request_id}

