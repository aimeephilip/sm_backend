from sqlmodel import Session, select
from app.models import User

def get_or_create_user(session: Session, client_id: str) -> User:
    """
    Bootstrap user using real client_id.
    """
    client_id = (client_id or "").strip().lower()

    user = session.exec(
        select(User).where(User.client_id == client_id)
    ).first()

    if user:
        return user

    user = User(
        client_id=client_id,
        role="patient",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
