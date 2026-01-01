# app/main.py

from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import os
import shutil
import json
import traceback
from datetime import datetime
import threading
import time

from sqlmodel import Session, select

from app.services.processor import process_video
from app.db import create_db_and_tables, get_session

# --- Firebase profile endpoints (kept even if Flutter not using right now) ---
from app.utils.auth import get_firebase_claims
from app.utils.profile import ensure_profile

# --- Register SQLModel tables (imports ensure tables are created) ---
import app.models                      # User
import app.models_movement             # MovementResult
import app.models_clinician            # ClinicianPatient
import app.models_notes                # ClinicalNote
import app.models_profile              # UserProfile

from app.models import User
from app.models_notes import ClinicalNote
from app.models_movement import MovementResult

from app.utils.users import get_or_create_user
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
# Helpers
# -------------------------------------------------------------------

def norm_client_id(client_id: str) -> str:
    return (client_id or "").strip().lower()

def _parse_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    return default

# -------------------------------------------------------------------
# MediaPipe warmup (prevents first /upload paying multi-minute init)
# -------------------------------------------------------------------

def _warmup_mediapipe():
    """
    Loads MediaPipe/TFLite once so the first /upload doesn't pay the init cost.
    Runs in a background thread so Render startup isn't blocked.
    """
    try:
        t0 = time.time()
        import numpy as np
        import cv2
        import mediapipe as mp

        mp_pose = mp.solutions.pose
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)

        with mp_pose.Pose(static_image_mode=True) as pose:
            pose.process(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))

        print(f"WARMUP: mediapipe ready in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"WARMUP: failed: {e}")

# -------------------------------------------------------------------
# Root / Health
# -------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "service": "sm-backend"}

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}

# Optional: a warmup endpoint you can hit from Flutter if you want
@app.get("/warmup", include_in_schema=False)
def warmup():
    return {"ok": True}

# -------------------------------------------------------------------
# Firebase: profile + role (kept)
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
    prof.full_name = (full_name or "").strip()
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
    if not client_id:
        return {"role": "patient"}

    client_id = norm_client_id(client_id)

    user = session.exec(
        select(User)
        .where(User.email == client_id)
        .order_by(User.id.desc())
    ).first()

    if not user:
        return {"role": "patient"}

    # 🔒 authoritative role from User table
    role = (user.role or "patient").strip().lower()
    return {"role": role}


# -------------------------------------------------------------------
# DEBUG (keep for now)
# -------------------------------------------------------------------

@app.get("/debug/db", include_in_schema=False)
def debug_db(session: Session = Depends(get_session)):
    users = session.exec(select(User)).all()
    return {"ok": True, "user_count": len(users)}

@app.get("/debug/results", include_in_schema=False)
def debug_results(session: Session = Depends(get_session)):
    rows = session.exec(select(MovementResult)).all()
    return {"count": len(rows)}

@app.post("/debug/make_clinician/{client_id}", include_in_schema=False)
def make_clinician(client_id: str, session: Session = Depends(get_session)):
    client_id = norm_client_id(client_id)
    if not client_id:
        return {"error": "missing_client_id"}

    # ✅ create if missing
    user = get_or_create_user(session, client_id)

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

    clinician_client_id = norm_client_id(clinician_client_id)
    patient_client_id = norm_client_id(patient_client_id)

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
# Clinician: TOKEN ONLY (kept; not used if you reverted to client_id)
# -------------------------------------------------------------------

@app.get("/clinician/patients")
def clinician_patients(
    claims=Depends(get_firebase_claims),
    session: Session = Depends(get_session),
):
    """
    Token-based clinician list.
    """
    from app.models_clinician import ClinicianPatient

    uid = claims.get("uid")
    email = claims.get("email")
    if not uid or not email:
        return {"error": "missing_claims"}

    prof = ensure_profile(session, uid, email)
    if prof.role != "clinician":
        return {"error": "not_authorised"}

    clinician_user = session.exec(select(User).where(User.email == prof.email)).first()
    if not clinician_user:
        clinician_user = get_or_create_user(session, prof.email)
        clinician_user.role = "clinician"
        session.add(clinician_user)
        session.commit()
        session.refresh(clinician_user)

    links = session.exec(
        select(ClinicianPatient).where(ClinicianPatient.clinician_id == clinician_user.id)
    ).all()
    patient_ids = [l.patient_id for l in links]

    if not patient_ids:
        return {"clinician": prof.email, "patients": []}

    patients = session.exec(select(User).where(User.id.in_(patient_ids))).all()

    return {
        "clinician": prof.email,
        "patients": [{"id": p.id, "client_id": p.email} for p in patients],
    }

@app.get("/clinician/patient/notes")
def clinician_list_notes(
    patient_client_id: str,
    claims=Depends(get_firebase_claims),
    session: Session = Depends(get_session),
):
    patient_client_id = norm_client_id(patient_client_id)

    uid = claims.get("uid")
    email = claims.get("email")
    if not uid or not email:
        return {"error": "missing_claims"}

    prof = ensure_profile(session, uid, email)
    if prof.role != "clinician":
        return {"error": "not_authorised"}

    clinician_user = session.exec(select(User).where(User.email == prof.email)).first()
    patient_user = session.exec(select(User).where(User.email == patient_client_id)).first()

    if not clinician_user or not patient_user:
        return {"error": "user_not_found"}

    if not clinician_can_access_patient(session, clinician_user, patient_user):
        return {"error": "not_authorised"}

    notes = session.exec(
        select(ClinicalNote)
        .where(ClinicalNote.patient_id == patient_user.id)
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
    patient_client_id: str,
    title: str,
    note: str,
    claims=Depends(get_firebase_claims),
    session: Session = Depends(get_session),
):
    patient_client_id = norm_client_id(patient_client_id)

    uid = claims.get("uid")
    email = claims.get("email")
    if not uid or not email:
        return {"error": "missing_claims"}

    prof = ensure_profile(session, uid, email)
    if prof.role != "clinician":
        return {"error": "not_authorised"}

    clinician_user = session.exec(select(User).where(User.email == prof.email)).first()
    patient_user = session.exec(select(User).where(User.email == patient_client_id)).first()

    if not clinician_user or not patient_user:
        return {"error": "user_not_found"}

    if not clinician_can_access_patient(session, clinician_user, patient_user):
        return {"error": "not_authorised"}

    n = ClinicalNote(
        clinician_id=clinician_user.id,
        patient_id=patient_user.id,
        title=(title or "").strip() or "Clinical Note",
        note=(note or "").strip(),
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
# Clinician: LEGACY client_id-based (use these if you're back to client_id)
# -------------------------------------------------------------------

@app.get("/clinician/patients_legacy")
def clinician_patients_legacy(
    clinician_client_id: str,
    session: Session = Depends(get_session),
):
    from app.models_clinician import ClinicianPatient

    clinician_client_id = norm_client_id(clinician_client_id)

    clinician_user = session.exec(select(User).where(User.email == clinician_client_id)).first()
    if not clinician_user:
        return {"error": "user_not_found"}
    if clinician_user.role != "clinician":
        return {"error": "not_authorised"}

    links = session.exec(
        select(ClinicianPatient).where(ClinicianPatient.clinician_id == clinician_user.id)
    ).all()
    patient_ids = [l.patient_id for l in links]

    if not patient_ids:
        return {"clinician": clinician_user.email, "patients": []}

    patients = session.exec(select(User).where(User.id.in_(patient_ids))).all()
    return {
        "clinician": clinician_user.email,
        "patients": [{"id": p.id, "client_id": p.email} for p in patients],
    }

@app.get("/clinician/patient/notes_legacy")
def clinician_list_notes_legacy(
    clinician_client_id: str,
    patient_client_id: str,
    session: Session = Depends(get_session),
):
    clinician_client_id = norm_client_id(clinician_client_id)
    patient_client_id = norm_client_id(patient_client_id)

    clinician_user = session.exec(select(User).where(User.email == clinician_client_id)).first()
    patient_user = session.exec(select(User).where(User.email == patient_client_id)).first()

    if not clinician_user or not patient_user:
        return {"error": "user_not_found"}
    if clinician_user.role != "clinician":
        return {"error": "not_authorised"}
    if not clinician_can_access_patient(session, clinician_user, patient_user):
        return {"error": "not_authorised"}

    notes = session.exec(
        select(ClinicalNote)
        .where(ClinicalNote.patient_id == patient_user.id)
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

@app.post("/clinician/patient/notes_legacy")
def clinician_create_note_legacy(
    clinician_client_id: str,
    patient_client_id: str,
    title: str,
    note: str,
    session: Session = Depends(get_session),
):
    clinician_client_id = norm_client_id(clinician_client_id)
    patient_client_id = norm_client_id(patient_client_id)

    clinician_user = session.exec(select(User).where(User.email == clinician_client_id)).first()
    patient_user = session.exec(select(User).where(User.email == patient_client_id)).first()

    if not clinician_user or not patient_user:
        return {"error": "user_not_found"}
    if clinician_user.role != "clinician":
        return {"error": "not_authorised"}
    if not clinician_can_access_patient(session, clinician_user, patient_user):
        return {"error": "not_authorised"}

    n = ClinicalNote(
        clinician_id=clinician_user.id,
        patient_id=patient_user.id,
        title=(title or "").strip() or "Clinical Note",
        note=(note or "").strip(),
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
# Upload & process video (client_id form field) — saves to Postgres reliably
# -------------------------------------------------------------------

UPLOAD_DIR = os.path.join("app", "static", "processed")
HISTORY_DIR = os.path.join("app", "static", "history")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

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
    t0 = time.time()
    print("UPLOAD: received request")

    client_id = norm_client_id(client_id)
    if not client_id:
        return JSONResponse(status_code=400, content={"error": "missing_client_id"})

    # Ensure user exists
    user = get_or_create_user(db, client_id)

    ext = os.path.splitext(file.filename or "video.mp4")[1]
    file_id = str(uuid.uuid4())
    upload_dir = os.path.join(UPLOAD_DIR, client_id, file_id)
    os.makedirs(upload_dir, exist_ok=True)

    original_path = os.path.join(upload_dir, f"original{ext}")
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"UPLOAD: saved file in {time.time() - t0:.2f}s")

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

    print(f"UPLOAD: processing done in {time.time() - t0:.2f}s")

    # Hard save to Postgres
    try:
        row = MovementResult(
            user_id=user.id,
            movement=(movement_type or "").strip().lower(),
            side=(side or "").strip().lower(),
            min_angle=result.get("min_angle"),
            max_angle=result.get("max_angle"),
            rom=result.get("rom"),
        )
        if hasattr(row, "created_at"):
            setattr(row, "created_at", datetime.utcnow())

        db.add(row)
        db.commit()

    except Exception as e:
        db.rollback()
        print("DB SAVE ERROR:", e)
        print(traceback.format_exc())
        return {
            "message": "Upload+processing OK, but DB save failed",
            "file_id": file_id,
            "results": result,
            "db_error": str(e),
        }

    print(f"UPLOAD: DB save done in {time.time() - t0:.2f}s")

    return {
        "message": "Upload and processing successful",
        "file_id": file_id,
        "results": result,
    }

# -------------------------------------------------------------------
# Patient notes (legacy client_id param)
# -------------------------------------------------------------------

@app.get("/patient/notes")
def patient_list_notes(
    client_id: str,
    session: Session = Depends(get_session),
):
    client_id = norm_client_id(client_id)

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
# History (DB first, JSON fallback)
# -------------------------------------------------------------------

@app.get("/history/{client_id}")
def get_client_history(client_id: str, session: Session = Depends(get_session)):
    client_id = norm_client_id(client_id)

    # 1) DB first
    user = session.exec(select(User).where(User.email == client_id)).first()
    if user:
        rows = session.exec(
            select(MovementResult)
            .where(MovementResult.user_id == user.id)
            .order_by(MovementResult.created_at.asc())
        ).all()

        if rows:
            history = []
            for r in rows:
                created_at = getattr(r, "created_at", None)
                history.append({
                    "movement": getattr(r, "movement", None),
                    "side": getattr(r, "side", None),
                    "max_angle": getattr(r, "max_angle", None),
                    "min_angle": getattr(r, "min_angle", None),
                    "rom": getattr(r, "rom", None),
                    "created_at": created_at.isoformat() + "Z" if created_at else None,
                })
            return {"client_id": client_id, "history": history}

    # 2) JSON fallback (legacy)
    history_file = os.path.join(HISTORY_DIR, f"{client_id}.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
        return {"client_id": client_id, "history": history}

    return JSONResponse(status_code=404, content={"message": "Client not found"})

# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    create_db_and_tables()
    threading.Thread(target=_warmup_mediapipe, daemon=True).start()

from fastapi import Body
from datetime import datetime
from sqlmodel import select
from app.models import User
from app.models_movement import MovementResult
from app.utils.users import get_or_create_user

@app.post("/symmetry/save")
def save_symmetry(payload: dict = Body(...), session: Session = Depends(get_session)):
    client_id = (payload.get("client_id") or "").strip().lower()
    movement = (payload.get("movement") or "").strip().lower()
    symmetry = payload.get("symmetry", None)

    right_max = payload.get("right_max", None)
    left_max = payload.get("left_max", None)

    if not client_id or not movement or symmetry is None:
        return JSONResponse(status_code=400, content={"error": "missing_fields"})

    user = get_or_create_user(session, client_id)

    row = MovementResult(
        user_id=user.id,
        movement=f"symmetry_{movement}",
        side="both",
        min_angle=None,
        max_angle=float(symmetry),
        rom=None,
    )
    if hasattr(row, "created_at"):
        row.created_at = datetime.utcnow()

    session.add(row)
    session.commit()
    return {"ok": True}
