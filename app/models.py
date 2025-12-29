# app/models.py
from typing import Optional
from datetime import datetime
import uuid
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    """
    Application user.

    For now, this is just a placeholder so we can
    prove Postgres is connected and working.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    email: str = Field(index=True, unique=True)
    role: str = Field(default="patient", index=True)  # patient | clinician | admin
    created_at: datetime = Field(default_factory=datetime.utcnow)
