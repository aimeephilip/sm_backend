from sqlmodel import Session, select
from app.models import User


def _normalize_client_id(client_id: str) -> str:
    return (client_id or "").strip().lower()


def _client_id_to_email(client_id: str) -> str:
    cid = _normalize_client_id(client_id)
    if not cid:
        return ""
    # If it already looks like an email (including @local), don't append again.
    if "@" in cid:
        return cid
    return f"{cid}@local"


def get_or_create_user(session: Session, client_id: str) -> User:
    """
    User bootstrap using client_id.

    Stores the identifier consistently:
      - User.client_id = normalized client_id
      - User.email = client_id converted to email form
        (e.g. "clinician1" -> "clinician1@local")
    """
    cid = _normalize_client_id(client_id)
    email = _client_id_to_email(cid)

    if not cid or not email:
        raise ValueError("missing_client_id")

    user = session.exec(select(User).where(User.email == email)).first()
    if user:
        # Backfill client_id if an older row is missing it
        if getattr(user, "client_id", None) != cid:
            user.client_id = cid
            session.add(user)
            session.commit()
            session.refresh(user)
        return user

    user = User(
        client_id=cid,
        email=email,
        role="patient",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
