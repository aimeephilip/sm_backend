# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import os
import shutil
import json
import traceback
import app.models_clinician

from sqlmodel import Session, select

from app.services.processor import process_video

from app.db import create_db_and_tables, get_session
import app.models                      # registers User table
import app.models_movement             # registers MovementResult table
from app.models import User

from app.utils.users import get_or_create_user
from app.utils.results import save_result   # ✅ THIS WAS MISSING

app = FastAPI(
    title="StretchMasters Backend",
    version="0.1.1",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Root & health ---
@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "service": "sm-backend"}

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}

# --- TEMP Debug endpoint: users ---
@app.get("/debug/db", include_in_schema=False)
def debug_db(session: Session = Depends(get_session)):
    users = session.exec(select(User)).all()
    return {"ok": True, "user_count": len(users)}

# --- TEMP Debug endpoint: results ---
@app.get("/debug/results", include_in_schema=False)
def debug_results(session: Session = Depends(get_session)):
    from app.models_movement import MovementResult
    rows = session.exec(select(MovementResult)).all()
    return {"count": len(rows)}

# --- Storage paths ---
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

# --- Upload & process video ---
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
    # ✅ Ensure user exists (Postgres)
    user = get_or_create_user(db, client_id)

    # Create unique folder
    ext = os.path.splitext(file.filename or "video.mp4")[1]
    file_id = str(uuid.uuid4())
    upload_dir = os.path.join(UPLOAD_DIR, client_id, file_id)
    os.makedirs(upload_dir, exist_ok=True)

    # Save original video
    original_path = os.path.join(upload_dir, f"original{ext}")
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        size = os.path.getsize(original_path)
    except Exception:
        size = -1
    print(f"DEBUG saved video: {original_path} size={size} bytes")

    # Process video
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

    # ✅ Save movement result to Postgres
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

    result["folder"] = upload_dir
    print("DEBUG upload response:", result)

    return {
        "message": "Upload and processing successful",
        "file_id": file_id,
        "results": result,
    }

@app.post("/debug/make_clinician/{client_id}", include_in_schema=False)
def make_clinician(client_id: str, session: Session = Depends(get_session)):
    fake_email = f"{client_id}@local"
    user = session.exec(
        select(User).where(User.email == fake_email)
    ).first()

    if not user:
        return {"error": "user_not_found"}

    user.role = "clinician"
    session.add(user)
    session.commit()

    return {"ok": True, "user_id": user.id, "role": user.role}


# --- Fetch history (JSON-based, unchanged) ---
@app.get("/history/{client_id}")
def get_client_history(client_id: str):
    history_file = os.path.join(HISTORY_DIR, f"{client_id}.json")
    if not os.path.exists(history_file):
        return JSONResponse(status_code=404, content={"message": "Client not found"})

    with open(history_file, "r") as f:
        history = json.load(f)

    return {"client_id": client_id, "history": history}

# --- Startup: create DB tables ---
@app.on_event("startup")
async def _init_db():
    create_db_and_tables()

# --- Debug: list routes ---
@app.on_event("startup")
async def _log_routes():
    print("=== ROUTES ===")
    for r in app.routes:
        try:
            print(getattr(r, "path"))
        except Exception:
            pass
    print("==============")
