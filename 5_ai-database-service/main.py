from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from models import TextGenerationRequest, TextGenerationResponse, GenerationRecord, Base
from database import get_db, engine
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create DB tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Database Service", version="1.0.0")

# Load model pipeline once on startup
generator = pipeline("text-generation", model="gpt2")

FORBIDDEN_WORDS = ["hate", "violence"]

def run_model_inference(prompt: str, max_length: int) -> str:
    """Runs the model inference in a blocking manner."""
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]["generated_text"]

@app.post("/generate", response_model=TextGenerationResponse)
def generate_text(request: TextGenerationRequest, db: Session = Depends(get_db)):
    """
    Generates text and stores the result.
    This is a synchronous endpoint, so FastAPI runs it in a thread pool.
    """
    # Content filtering
    if any(word in request.prompt.lower() for word in FORBIDDEN_WORDS):
        logger.warning(f"Rejected prompt: {request.prompt}")
        raise HTTPException(status_code=400, detail="Prompt contains inappropriate content")
    
    try:
        # Run inference directly. FastAPI handles the threading.
        generated_text = run_model_inference(request.prompt, request.max_length)
        
        # Store in database
        record = GenerationRecord(
            prompt=request.prompt,
            generated_text=generated_text,
            model="gpt2"  # Renamed from model_name
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        
        logger.info(f"Stored record ID {record.id} for prompt: {request.prompt[:20]}...")
        
        return TextGenerationResponse(
            id=record.id,
            generated_text=record.generated_text,
            model=record.model,  # Renamed from model_name
            created_at=record.created_at.isoformat()
        )
    except Exception as e:
        logger.error(f"Inference or database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/history/{record_id}", response_model=TextGenerationResponse)
def get_history(record_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a generation record from the database.
    This is a synchronous endpoint for database IO.
    """
    record = db.query(GenerationRecord).filter(GenerationRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return TextGenerationResponse(
        id=record.id,
        generated_text=record.generated_text,
        model=record.model,  # Renamed from model_name
        created_at=record.created_at.isoformat()
    )

logger.info("FastAPI app initialized.")