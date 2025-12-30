# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import os
import shutil
import json
import traceback

from sqlmodel import Session, select

from app.services.processor import process_video
from app.db import create_db_and_tables, get_session

# --- Firebase profile endpoints (safe to include even if Flutter not using yet) ---
from app.utils.auth import get_firebase_claims
from app.utils.profile import ensure_profile

# --- Register SQLModel tables (imports ensure tables are created) ---
import app.models                      # User
import app.models_movement             # MovementResult
import app.models_clinician            # ClinicianPatient
import app.models_notes                # ClinicalNote
import app.models_profile              # UserProfile (your new table)

from app.models import User
from app.models_notes import ClinicalNote
from app.utils.users import get_or_create_user
from app.utils.results import save_result
from app.utils.permissions import (
    get_user_by_client_id,
    clinician_can_access_patient,
)

app = FastAPI(
    title="StretchMasters Backend",
    version="0.1.1",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Root / Health
# -------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "service": "sm-backend"}

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}

# -------------------------------------------------------------------
# Firebase: profile + role (NEW)
# These are the endpoints your Flutter app should move to next.
# -------------------------------------------------------------------

@app.get("/me")
def me(
    claims=Depends(get_firebase_claims),
    session: Session = Depends(get_session),
):
    uid = claims.get("uid")
    email = claims.get("email")
    if not uid or not email:
        return {"error": "missing_claims"}

    prof = ensure_profile(session, uid, email)

    return {
        "uid": prof.firebase_uid,
        "email": prof.email,
        "full_name": prof.full_name,
        "role": prof.role,
        "needs_profile": (prof.full_name.strip() == ""),
    }

@app.post("/me/profile")
def update_profile(
    full_name: str,
    claims=Depends(get_firebase_claims),
    session: Session = Depends(get_session),
):
    uid = claims.get("uid")
    email = claims.get("email")
    if not uid or not email:
        return {"error": "missing_claims"}

    prof = ensure_profile(session, uid, email)
    prof.full_name = full_name.strip()

    # simple updated_at
    from datetime import datetime
    prof.updated_at = datetime.utcnow()

    session.add(prof)
    session.commit()
    session.refresh(prof)

    return {"ok": True, "full_name": prof.full_name}

@app.get("/me/role")
def me_role(
    client_id: str | None = None,
    session: Session = Depends(get_session),
):
    """
    Backwards-compatible endpoint.

    - If your Flutter still calls /me/role?client_id=XYZ, this will return role based on User table.
    - Once Flutter uses Firebase, you should switch to calling GET /me and read role from there.
    """
    if not client_id:
        return {"role": "patient"}

    user = session.exec(select(User).where(User.email == client_id)).first()
    if not user:
        return {"role": "patient"}
    return {"role": user.role}

# -------------------------------------------------------------------
# DEBUG (Keep for now, remove/lock down before launch)
# -------------------------------------------------------------------

@app.get("/debug/db", include_in_schema=False)
def debug_db(session: Session = Depends(get_session)):
    users = session.exec(select(User)).all()
    return {"ok": True, "user_count": len(users)}

@app.get("/debug/results", include_in_schema=False)
def debug_results(session: Session = Depends(get_session)):
    from app.models_movement import MovementResult
    rows = session.exec(select(MovementResult)).all()
    return {"count": len(rows)}

@app.post("/debug/make_clinician/{client_id}", include_in_schema=False)
def make_clinician(client_id: str, session: Session = Depends(get_session)):
    """
    Promotes an existing User (by email/client_id) to clinician.
    """
    user = session.exec(select(User).where(User.email == client_id)).first()
    if not user:
        return {"error": "user_not_found"}

    user.role = "clinician"
    session.add(user)
    session.commit()
    session.refresh(user)

    return {"ok": True, "client_id": client_id, "role": user.role}

@app.post("/debug/assign_patient", include_in_schema=False)
def assign_patient(
    clinician_client_id: str,
    patient_client_id: str,
    session: Session = Depends(get_session),
):
    """
    Links clinician -> patient using your existing User table.
    """
    from app.models_clinician import ClinicianPatient

    clinician = session.exec(select(User).where(User.email == clinician_client_id)).first()
    patient = session.exec(select(User).where(User.email == patient_client_id)).first()

    if not clinician or not patient:
        return {"error": "user_not_found"}

    if clinician.role != "clinician":
        return {"error": "not_a_clinician"}

    link = ClinicianPatient(clinician_id=clinician.id, patient_id=patient.id)
    session.add(link)
    session.commit()

    return {"ok": True}

# -------------------------------------------------------------------
# Clinician: list assigned patients (CURRENT FLOW)
# -------------------------------------------------------------------

@app.get("/clinician/patients")
def clinician_patients(
    clinician_client_id: str,
    session: Session = Depends(get_session),
):
    """
    Current Flutter uses clinician_client_id query param.
    Later we will switch this to Firebase token-only.
    """
    from app.models_clinician import ClinicianPatient

    clinician = session.exec(select(User).where(User.email == clinician_client_id)).first()
    if not clinician or clinician.role != "clinician":
        return {"error": "not_authorised"}

    links = session.exec(
        select(ClinicianPatient).where(ClinicianPatient.clinician_id == clinician.id)
    ).all()
    patient_ids = [l.patient_id for l in links]

    if not patient_ids:
        return {"clinician": clinician.email, "patients": []}

    patients = session.exec(select(User).where(User.id.in_(patient_ids))).all()

    return {
        "clinician": clinician.email,
        "patients": [
            {"id": p.id, "client_id": p.email}
            for p in patients
        ],
    }

# -------------------------------------------------------------------
# Clinician: notes (list + create)
# -------------------------------------------------------------------

@app.get("/clinician/patient/notes")
def clinician_list_notes(
    clinician_client_id: str,
    patient_client_id: str,
    session: Session = Depends(get_session),
):
    clinician = get_user_by_client_id(session, clinician_client_id)
    patient = get_user_by_client_id(session, patient_client_id)

    if not clinician or not patient:
        return {"error": "user_not_found"}

    if not clinician_can_access_patient(session, clinician, patient):
        return {"error": "not_authorised"}

    notes = session.exec(
        select(ClinicalNote)
        .where(ClinicalNote.patient_id == patient.id)
        .order_by(ClinicalNote.created_at.desc())
    ).all()

    return {
        "patient_client_id": patient_client_id,
        "notes": [
            {
                "id": n.id,
                "title": n.title,
                "note": n.note,
                "created_at": n.created_at.isoformat() + "Z",
            }
            for n in notes
        ],
    }

@app.post("/clinician/patient/notes")
def clinician_create_note(
    clinician_client_id: str,
    patient_client_id: str,
    title: str,
    note: str,
    session: Session = Depends(get_session),
):
    clinician = get_user_by_client_id(session, clinician_client_id)
    patient = get_user_by_client_id(session, patient_client_id)

    if not clinician or not patient:
        return {"error": "user_not_found"}

    if not clinician_can_access_patient(session, clinician, patient):
        return {"error": "not_authorised"}

    n = ClinicalNote(
        clinician_id=clinician.id,
        patient_id=patient.id,
        title=title.strip() or "Clinical Note",
        note=note.strip(),
    )
    session.add(n)
    session.commit()
    session.refresh(n)

    return {
        "ok": True,
        "note": {
            "id": n.id,
            "title": n.title,
            "note": n.note,
            "created_at": n.created_at.isoformat() + "Z",
        },
    }

# -------------------------------------------------------------------
# Upload & process video (CURRENT FLOW)
# -------------------------------------------------------------------

UPLOAD_DIR = os.path.join("app", "static", "processed")
HISTORY_DIR = os.path.join("app", "static", "history")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

def _parse_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    return default

@app.post("/upload/")
async def upload_video(
    file: UploadFile = File(...),
    movement_type: str = Form(...),
    side: str = Form("left"),
    client_id: str = Form(...),
    session_id: str = Form(None),
    compute_symmetry: str = Form("true"),
    db: Session = Depends(get_session),
):
    # Ensure user exists (legacy client_id flow)
    user = get_or_create_user(db, client_id)

    ext = os.path.splitext(file.filename or "video.mp4")[1]
    file_id = str(uuid.uuid4())
    upload_dir = os.path.join(UPLOAD_DIR, client_id, file_id)
    os.makedirs(upload_dir, exist_ok=True)

    original_path = os.path.join(upload_dir, f"original{ext}")
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = process_video(
            original_path,
            movement_type,
            side,
            client_id,
            session_id=session_id,
            compute_symmetry=_parse_bool(compute_symmetry, default=True),
        )
    except Exception as e:
        print("UPLOAD ERROR:", e)
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "processing_failed", "detail": str(e)},
        )

    # Save movement result to Postgres
    save_result(
        db,
        user,
        {
            "movement": movement_type,
            "side": side,
            "min_angle": result.get("min_angle"),
            "max_angle": result.get("max_angle"),
            "rom": result.get("rom"),
        },
    )

    return {
        "message": "Upload and processing successful",
        "file_id": file_id,
        "results": result,
    }

# -------------------------------------------------------------------
# Patient notes (read-only)
# -------------------------------------------------------------------

@app.get("/patient/notes")
def patient_list_notes(
    client_id: str,
    session: Session = Depends(get_session),
):
    patient = get_user_by_client_id(session, client_id)
    if not patient:
        return {"error": "user_not_found"}

    notes = session.exec(
        select(ClinicalNote)
        .where(ClinicalNote.patient_id == patient.id)
        .order_by(ClinicalNote.created_at.desc())
    ).all()

    return {
        "client_id": client_id,
        "notes": [
            {
                "id": n.id,
                "title": n.title,
                "note": n.note,
                "created_at": n.created_at.isoformat() + "Z",
            }
            for n in notes
        ],
    }

# -------------------------------------------------------------------
# History (legacy JSON)
# -------------------------------------------------------------------

@app.get("/history/{client_id}")
def get_client_history(client_id: str):
    history_file = os.path.join(HISTORY_DIR, f"{client_id}.json")
    if not os.path.exists(history_file):
        return JSONResponse(status_code=404, content={"message": "Client not found"})

    with open(history_file, "r") as f:
        history = json.load(f)

    return {"client_id": client_id, "history": history}

# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    create_db_and_tables()
