# app/models_notes.py
from datetime import datetime
import uuid
from sqlmodel import SQLModel, Field

class ClinicalNote(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)

    clinician_id: str = Field(index=True)
    patient_id: str = Field(index=True)

    title: str = Field(default="Clinical Note")
    note: str

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
