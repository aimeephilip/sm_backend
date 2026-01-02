# app/utils/permissions.py
from sqlmodel import Session, select
from app.models import User
from app.models_clinician import ClinicianPatient

def norm_client_id(client_id: str) -> str:
    return (client_id or "").strip().lower()

def email_for_client_id(client_id: str) -> str | None:
    cid = norm_client_id(client_id)
    if not cid:
        return None
    return cid if "@" in cid else f"{cid}@local"

def get_user_by_client_id(session: Session, client_id: str) -> User | None:
    email = email_for_client_id(client_id)
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
