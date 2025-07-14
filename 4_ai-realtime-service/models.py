# models.py


from pydantic import BaseModel, Field


class WebSocketMessage(BaseModel):
    prompt: str
    max_length: int = Field(default=50, gt=10, le=250)


class WebSocketResponse(BaseModel):
    generated_text: str
    source_model: str  # Renamed to a completely non-conflicting name.
    client_id: str
