from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import os
import shutil
import json

from app.services.processor import process_video  # your existing import

app = FastAPI()

# Allow your mobile app (and browsers) to call this API during testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten later to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "app/static/processed"
HISTORY_DIR = os.path.join("app", "static", "history")

# Ensure folders exist (safe if already present)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload/")
async def upload_video(
    file: UploadFile = File(...),
    movement_type: str = Form(...),
    side: str = Form("left"),
    client_id: str = Form(...)
):
    # Create a unique folder for this upload
    ext = os.path.splitext(file.filename)[1]
    file_id = str(uuid.uuid4())
    upload_dir = os.path.join(UPLOAD_DIR, client_id, file_id)
    os.makedirs(upload_dir, exist_ok=True)

    # Save original video
    original_filename = f"original{ext}"
    file_path = os.path.join(upload_dir, original_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process video (your existing pipeline)
    result = process_video(file_path, movement_type, side, client_id)
    result["folder"] = upload_dir  # optional: return path to frontend

    return {
        "message": "Upload and processing successful",
        "file_id": file_id,
        "results": result
    }

@app.get("/history/{client_id}")
def get_client_history(client_id: str):
    history_file = os.path.join(HISTORY_DIR, f"{client_id}.json")
    if not os.path.exists(history_file):
        return JSONResponse(status_code=404, content={"message": "Client not found"})

    with open(history_file, "r") as f:
        history = json.load(f)

    return {"client_id": client_id, "history": history}
