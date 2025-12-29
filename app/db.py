import os
from sqlmodel import SQLModel, create_engine, Session

def get_engine():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable not set")

    # Render Postgres URLs are usually fine as-is for SQLAlchemy.
    return create_engine(database_url, pool_pre_ping=True)

engine = get_engine()

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
