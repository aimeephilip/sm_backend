from typing import Optional
from datetime import datetime
import uuid
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)

    # ✅ real identifier
    client_id: str = Field(index=True, unique=True)

    # optional / future use
    email: Optional[str] = Field(default=None, index=True)

    role: str = Field(default="patient", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
