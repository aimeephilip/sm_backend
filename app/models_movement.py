# app/models_movement.py
from typing import Optional
from datetime import datetime
import uuid
from sqlmodel import SQLModel, Field

class MovementResult(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)

    user_id: str = Field(index=True)
    movement: str = Field(index=True)
    side: str

    min_angle: float
    max_angle: float
    rom: float

    created_at: datetime = Field(default_factory=datetime.utcnow)
