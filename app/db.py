# app/db.py
import os
from typing import Optional
from sqlmodel import SQLModel, create_engine, Session

_engine = None  # lazy singleton

def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable not set")

    _engine = create_engine(database_url, pool_pre_ping=True)
    return _engine

def create_db_and_tables():
    engine = get_engine()
    SQLModel.metadata.create_all(engine)

def get_session():
    engine = get_engine()
    with Session(engine) as session:
        yield session
