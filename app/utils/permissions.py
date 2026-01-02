# app/utils/permissions.py
from sqlmodel import Session, select
from app.models import User
from app.models_clinician import ClinicianPatient


def to_user_email(identifier: str) -> str:
    """
    Accepts either:
      - "clinician1"
      - "clinician1@local"
      - "someone@domain.com"
    Returns a stable identifier used in User.email.
    """
    v = (identifier or "").strip().lower()
    if not v:
        return v
    return v if "@" in v else f"{v}@local"


def get_user_by_client_id(session: Session, client_id: str) -> User | None:
    """
    Backwards compatible:
    - If caller passes "abc" -> looks up "abc@local"
    - If caller passes "abc@local" -> looks up exactly that
    - If caller passes real email -> looks up exactly that
    """
    email = to_user_email(client_id)
    if not email:
        return None
    return session.exec(select(User).where(User.email == email)).first()


def clinician_can_access_patient(session: Session, clinician: User, patient: User) -> bool:
    if clinician.role != "clinician":
        return False

    link = session.exec(
        select(ClinicianPatient).where(
            ClinicianPatient.clinician_id == clinician.id,
            ClinicianPatient.patient_id == patient.id,
        )
    ).first()
    return link is not None
