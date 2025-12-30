# app/utils/profile.py
from datetime import datetime
from sqlmodel import Session, select
from app.models_profile import UserProfile

CLINICIAN_DOMAIN = "@stretchmasters.co.uk"

def ensure_profile(session: Session, firebase_uid: str, email: str) -> UserProfile:
    prof = session.exec(
        select(UserProfile).where(UserProfile.firebase_uid == firebase_uid)
    ).first()

    if not prof:
        role = "clinician" if email.lower().endswith(CLINICIAN_DOMAIN) else "patient"
        prof = UserProfile(
            firebase_uid=firebase_uid,
            email=email.lower(),
            role=role,
            full_name="",
        )
        session.add(prof)
        session.commit()
        session.refresh(prof)
        return prof

    # keep email updated
    changed = False
    if prof.email != email.lower():
        prof.email = email.lower()
        changed = True

    # auto-promote clinicians only (never auto-demote)
    if email.lower().endswith(CLINICIAN_DOMAIN) and prof.role == "patient":
        prof.role = "clinician"
        changed = True

    if changed:
        prof.updated_at = datetime.utcnow()
        session.add(prof)
        session.commit()
        session.refresh(prof)

    return prof
