# app/utils/users.py
from sqlmodel import Session, select
from app.models import User

def get_or_create_user(session: Session, client_id: str) -> User:
    """
    Temporary user bootstrap using client_id.
    This will be replaced later by real auth.
    """
    fake_email = f"{client_id}@local"

    user = session.exec(
        select(User).where(User.email == fake_email)
    ).first()

    if user:
        return user

    user = User(
        email=fake_email,
        role="patient",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
