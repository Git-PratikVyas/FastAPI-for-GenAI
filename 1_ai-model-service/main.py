from fastapi import FastAPI, HTTPException
from models import TextGenerationRequest, TextGenerationResponse
from transformers import pipeline
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Model Service", version="1.0.0")
FORBIDDEN_WORDS = ["hate", "violence"]

# Load model with timing
start_time = time.time()
logger.info("Loading AI model...")
generator = pipeline("text-generation", model="gpt2")
logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    if any(word in request.prompt.lower() for word in FORBIDDEN_WORDS):
        logger.warning(f"Inappropriate prompt detected: {request.prompt}")
        raise HTTPException(status_code=400, detail="Prompt contains inappropriate content")
    
    try:
        logger.info(f"Generating text for prompt: {request.prompt}")
        result = generator(request.prompt, max_length=request.max_length, num_return_sequences=1)
        generated_text = result[0]["generated_text"]
        
        return TextGenerationResponse(
            generated_text=generated_text,
            model_name="gpt2"
        )
    except Exception as e:
        logger.error(f"Model inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
