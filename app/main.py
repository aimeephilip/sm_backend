# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import os
import shutil
import json
import traceback

from app.services.processor import process_video  # your existing processing function

app = FastAPI(
    title="StretchMasters Backend",
    version="0.1.1",
)

# --- CORS (testing-friendly; tighten later) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # e.g., ["https://your-app.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Simple root + health endpoints ---
@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "service": "sm-backend"}

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}

# --- Storage paths (make sure they exist) ---
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
    session_id: str = Form(None),               # optional
    compute_symmetry: str = Form("true"),       # optional (string -> bool)
):
    # create a unique folder for this upload
    ext = os.path.splitext(file.filename or "video.mp4")[1]
    file_id = str(uuid.uuid4())
    upload_dir = os.path.join(UPLOAD_DIR, client_id, file_id)
    os.makedirs(upload_dir, exist_ok=True)

    # save original video
    original_path = os.path.join(upload_dir, f"original{ext}")
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # quick sanity log so you can see it in Render logs
    try:
        size = os.path.getsize(original_path)
    except Exception:
        size = -1
    print(f"DEBUG saved video: {original_path} size={size} bytes")

    # process video with your pipeline
    try:
        result = process_video(
            original_path,
            movement_type,
            side,
            client_id,
            session_id=session_id,  # accepted by processor.py (currently unused)
            compute_symmetry=_parse_bool(compute_symmetry, default=True),  # accepted (unused)
        )
    except Exception as e:
        # Log full traceback for debugging in Render -> Logs
        print("UPLOAD ERROR:", e)
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "processing_failed", "detail": str(e)},
        )

    result["folder"] = upload_dir  # optional for frontend debugging
    print("DEBUG upload response:", result)  # shows max_angle etc. in Render logs

    return {
        "message": "Upload and processing successful",
        "file_id": file_id,
        "results": result,   # Flutter reads results.max_angle
    }

# --- Fetch history for a client ---
@app.get("/history/{client_id}")
def get_client_history(client_id: str):
    history_file = os.path.join(HISTORY_DIR, f"{client_id}.json")
    if not os.path.exists(history_file):
        return JSONResponse(status_code=404, content={"message": "Client not found"})

    with open(history_file, "r") as f:
        history = json.load(f)

    return {"client_id": client_id, "history": history}

# --- Debug: print routes at startup so we can see them in Render logs ---
@app.on_event("startup")
async def _log_routes():
    print("=== ROUTES ===")
    for r in app.routes:
        try:
            print(getattr(r, "path"))
        except Exception:
            pass
    print("==============")
