#  main.py 

from fastapi import FastAPI, Depends, HTTPException, status, Security, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from fastapi.staticfiles import StaticFiles  #  Import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html #  Import the docs generator

# Create database tables
Base.metadata.create_all(bind=engine)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Disable default docs URL to replace it with our own
app = FastAPI(
    title="Secure AI Service",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# <<<  Mount the static directory to serve the downloaded files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Secure headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # <<<  Update CSP to allow what Swagger UI needs (inline scripts/styles)
    # This is now secure because we are no longer relying on an external CDN.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline';"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# CORS for frontend integration (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# <<<  Add our own custom endpoint to serve the offline docs
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

# Initialize model and thread pool
generator = pipeline("text-generation", model="gpt2")
executor = ThreadPoolExecutor(max_workers=4)
FORBIDDEN_WORDS = ["hate", "violence"]
SANITIZE_PATTERN = re.compile(r'[<>{};]')  # Basic sanitization for malicious characters

def sanitize_input(text: str) -> str:
    return SANITIZE_PATTERN.sub("", text)

def run_model_inference(prompt: str, max_length: int) -> str:
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]["generated_text"]

@app.post("/users", response_model=UserResponse)
@limiter.limit("5/minute")
async def create_user(request: Request, user: UserCreate, db: Session = Depends(get_db)):
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

