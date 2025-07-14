from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from models import (
    User,
    UserCreate,
    UserResponse,
    TextGenerationRequest,
    TextGenerationResponse,
    Token,
)
from database import get_db
from auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
    get_current_admin,
)
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Secure AI Service", version="1.0.0")
generator = pipeline("text-generation", model="gpt2")
executor = ThreadPoolExecutor(max_workers=4)
FORBIDDEN_WORDS = ["hate", "violence"]


def run_model_inference(prompt: str, max_length: int) -> str:
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]["generated_text"]


@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username, hashed_password=hashed_password, role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"Created user: {user.username}")
    return UserResponse(id=db_user.id, username=db_user.username, role=db_user.role)


@app.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
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
async def generate_text(
    request: TextGenerationRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if any(word in request.prompt.lower() for word in FORBIDDEN_WORDS):
        logger.warning(f"User {user.username} sent forbidden prompt: {request.prompt}")
        raise HTTPException(
            status_code=400, detail="Prompt contains inappropriate content"
        )

    try:
        generated_text = await app.state.loop.run_in_executor(
            executor, run_model_inference, request.prompt, request.max_length
        )
        logger.info(f"Generated text for user {user.username}")
        # --- MODIFIED LINE TO MATCH MODEL CHANGE ---
        return TextGenerationResponse(generated_text=generated_text, model="gpt2")
    except Exception as e:
        logger.error(f"Inference failed for user {user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")


@app.get("/admin/users", response_model=list[UserResponse])
async def list_users(
    user: User = Depends(get_current_admin), db: Session = Depends(get_db)
):
    users = db.query(User).all()
    logger.info(f"Admin {user.username} accessed user list")
    return [UserResponse(id=u.id, username=u.username, role=u.role) for u in users]


# --- START: MODIFIED STARTUP/SHUTDOWN EVENTS ---
@app.on_event("startup")
async def startup_event():
    # Correctly get and assign the event loop
    app.state.loop = asyncio.get_event_loop()
    logger.info("FastAPI app started with thread pool")


@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)
    logger.info("Thread pool shutdown")


# --- END: MODIFIED STARTUP/SHUTDOWN EVENTS ---
