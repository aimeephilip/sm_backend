# app/utils/permissions.py
from sqlmodel import Session, select
from app.models import User
from app.models_clinician import ClinicianPatient

def get_user_by_client_id(session: Session, client_id: str) -> User | None:
    cid = (client_id or "").strip().lower()
    if not cid:
        return None
    email = cid if "@" in cid else f"{cid}@local"
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
