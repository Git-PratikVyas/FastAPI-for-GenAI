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
