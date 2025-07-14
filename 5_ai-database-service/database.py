from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# SQLite database (file-based)
DATABASE_URL = "sqlite:///ai_service.db"

# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Note: The Base.metadata.create_all(bind=engine) call was moved to main.py
# to be part of the application's startup sequence.

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()