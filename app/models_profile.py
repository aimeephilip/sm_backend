# app/models_profile.py
from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field
import uuid

class UserProfile(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)

    # Firebase identity
    firebase_uid: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)

    # App profile
    full_name: str = Field(default="")
    role: str = Field(default="patient", index=True)  # patient | clinician | admin

    # Optional later (add when ready)
    dob: Optional[str] = Field(default=None)      # keep as string initially (YYYY-MM-DD)
    gender: Optional[str] = Field(default=None)   # optional
    photo_url: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
