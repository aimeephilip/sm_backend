from sqlmodel import Session, select
from app.models import User

def _client_id_to_email(client_id: str) -> str:
    cid = (client_id or "").strip().lower()
    if not cid:
        return ""
    # If it already looks like an email (including @local), don't append again.
    if "@" in cid:
        return cid
    return f"{cid}@local"

def get_or_create_user(session: Session, client_id: str) -> User:
    """
    User bootstrap using client_id.
    Stores identifier consistently in User.email:
      - if client_id contains '@' => use as-is
      - else => append '@local'
    """
    email = _client_id_to_email(client_id)
    if not email:
        raise ValueError("missing_client_id")

    user = session.exec(select(User).where(User.email == email)).first()
    if user:
        return user

    user = User(email=email, role="patient")
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
