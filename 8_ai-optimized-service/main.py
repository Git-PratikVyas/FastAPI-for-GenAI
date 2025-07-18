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