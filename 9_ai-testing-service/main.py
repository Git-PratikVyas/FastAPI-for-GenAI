from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from models import TextGenerationRequest, TextGenerationResponse, GenerationRecord
from database import get_db
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Testing Service", version="1.0.0")
generator = pipeline("text-generation", model="gpt2", device=-1)  # CPU for simplicity
executor = ThreadPoolExecutor(max_workers=4)
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
        loop = asyncio.get_running_loop()
        generated_text = await loop.run_in_executor(
            executor, run_model_inference, request.prompt, request.max_length
        )
        record = GenerationRecord(
            prompt=request.prompt,
            generated_text=generated_text,
            model_name="gpt2"
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

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)
    logger.info("Thread pool shutdown")
