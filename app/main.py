from fastapi import FastAPI, UploadFile, File, Form, Depends, Body
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

from sqlmodel import Session, select, delete

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

from io import BytesIO
from fastapi.responses import JSONResponse, StreamingResponse
from app.services.report_generator import build_progress_report_pdf

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

def email_for_client_id(client_id: str) -> str | None:
    """
    Accepts:
      - "clinician1" -> "clinician1@local"
      - "a@b.com"    -> "a@b.com"
      - "aimee_philip@local" -> "aimee_philip@local"
    """
    cid = norm_client_id(client_id)
    if not cid:
        return None
    return cid if "@" in cid else f"{cid}@local"

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

    cid = norm_client_id(client_id)

    # 🚨 TEMP HARD-CODE OVERRIDE (explicit, intentional)
    if cid == "clinician1":
        return {"role": "clinician"}

    return {"role": "patient"}

# -------------------------------------------------------------------
# DEBUG
# -------------------------------------------------------------------

@app.get("/debug/db", include_in_schema=False)
def debug_db(session: Session = Depends(get_session)):
    users = session.exec(select(User)).all()
    return {"ok": True, "user_count": len(users)}

@app.get("/debug/users", include_in_schema=False)
def debug_users(session: Session = Depends(get_session)):
    users = session.exec(select(User)).all()
    return {
        "count": len(users),
        "users": [{"id": u.id, "email": u.email, "role": u.role} for u in users],
    }

@app.get("/debug/results", include_in_schema=False)
def debug_results(session: Session = Depends(get_session)):
    rows = session.exec(select(MovementResult)).all()
    return {"count": len(rows)}

@app.post("/debug/make_clinician/{client_id}", include_in_schema=False)
def make_clinician(client_id: str, session: Session = Depends(get_session)):
    cid = norm_client_id(client_id)
    if not cid:
        return {"error": "missing_client_id"}

    # This creates "cid@local" user row (patient by default)
    user = get_or_create_user(session, cid)
    user.role = "clinician"
    session.add(user)
    session.commit()
    session.refresh(user)

    return {"ok": True, "client_id": cid, "email": user.email, "role": user.role}

@app.post("/debug/reset_users", include_in_schema=False)
def reset_users(session: Session = Depends(get_session)):
    session.exec(delete(User))
    session.commit()
    return {"ok": True}

@app.post("/debug/assign_patient", include_in_schema=False)
def assign_patient(
    # allow query OR form
    clinician_client_id: str | None = None,
    patient_client_id: str | None = None,
    clinician_client_id_form: str | None = Form(None),
    patient_client_id_form: str | None = Form(None),
    session: Session = Depends(get_session),
):

@app.get("/report/pdf/{client_id}")
def export_progress_report(client_id: str, session: Session = Depends(get_session)):
    cid_email = email_for_client_id(client_id)
    if not cid_email:
        return JSONResponse(status_code=400, content={"error": "missing_client_id"})

    user = session.exec(select(User).where(User.email == cid_email)).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "user_not_found"})

    rom_rows = session.exec(
        select(MovementResult)
        .where(MovementResult.user_id == user.id)
        .order_by(MovementResult.created_at.asc())
    ).all()

    notes_rows = session.exec(
        select(ClinicalNote)
        .where(ClinicalNote.patient_id == user.id)
        .order_by(ClinicalNote.created_at.desc())
    ).all()

    rom_history = [
        {
            "movement": getattr(r, "movement", None),
            "side": getattr(r, "side", None),
            "max_angle": getattr(r, "max_angle", None),
            "min_angle": getattr(r, "min_angle", None),
            "rom": getattr(r, "rom", None),
            "created_at": r.created_at.isoformat() + "Z" if getattr(r, "created_at", None) else None,
        }
        for r in rom_rows
    ]

    gait_history = load_gait_history(client_id)

    notes = [
        {
            "title": n.title,
            "note": n.note,
            "created_at": n.created_at.isoformat() + "Z" if n.created_at else None,
        }
        for n in notes_rows
    ]

    pdf_bytes = build_progress_report_pdf(
        client_id=cid_email,
        rom_history=rom_history,
        gait_history=gait_history,
        notes=notes,
    )

    filename = f"progress_report_{norm_client_id(client_id)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )
    """
    Links clinician -> patient.
    Minimal + safe:
    - Accepts clinician/patient as "clinician1" or "clinician1@local"
    - Auto-creates missing users using get_or_create_user()
    """
    from app.models_clinician import ClinicianPatient

    # prefer form if present, otherwise query
    clinician_raw = clinician_client_id_form or clinician_client_id
    patient_raw = patient_client_id_form or patient_client_id

    if not clinician_raw or not patient_raw:
        return JSONResponse(status_code=400, content={"error": "missing_fields"})

    clinician_cid = norm_client_id(clinician_raw)
    patient_cid = norm_client_id(patient_raw)

    if not clinician_cid or not patient_cid:
        return JSONResponse(status_code=400, content={"error": "missing_fields"})

    # IMPORTANT: create users if they don't exist
    clinician_user = get_or_create_user(session, clinician_cid.replace("@local", "").replace("@", "") if clinician_cid.endswith("@local") else clinician_cid.split("@")[0] if "@local" in clinician_cid else clinician_cid if "@" not in clinician_cid else clinician_cid)
    # ensure clinician email is correct for local ids
    # (get_or_create_user already uses client_id -> client_id@local)
    if clinician_user.role != "clinician":
        clinician_user.role = "clinician"
        session.add(clinician_user)
        session.commit()
        session.refresh(clinician_user)

    patient_user = get_or_create_user(session, patient_cid.replace("@local", "").replace("@", "") if patient_cid.endswith("@local") else patient_cid.split("@")[0] if "@local" in patient_cid else patient_cid if "@" not in patient_cid else patient_cid)

    # avoid duplicate link
    existing = session.exec(
        select(ClinicianPatient).where(
            ClinicianPatient.clinician_id == clinician_user.id,
            ClinicianPatient.patient_id == patient_user.id,
        )
    ).first()
    if existing:
        return {"ok": True, "already_linked": True}

    link = ClinicianPatient(clinician_id=clinician_user.id, patient_id=patient_user.id)
    session.add(link)
    session.commit()
    return {"ok": True}

# -------------------------------------------------------------------
# Clinician: LEGACY client_id-based
# -------------------------------------------------------------------

@app.get("/clinician/patients_legacy")
def clinician_patients_legacy(
    clinician_client_id: str,
    session: Session = Depends(get_session),
):
    from app.models_clinician import ClinicianPatient

    clinician_email = email_for_client_id(clinician_client_id)
    if not clinician_email:
        return {"error": "missing_clinician"}

    clinician_user = session.exec(select(User).where(User.email == clinician_email)).first()
    if not clinician_user:
        return {"error": "user_not_found", "clinician": clinician_email}
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
    clinician_email = email_for_client_id(clinician_client_id)
    patient_email = email_for_client_id(patient_client_id)

    if not clinician_email or not patient_email:
        return {"error": "missing_fields"}

    clinician_user = session.exec(select(User).where(User.email == clinician_email)).first()
    patient_user = session.exec(select(User).where(User.email == patient_email)).first()

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
        "patient_client_id": patient_email,
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
    clinician_email = email_for_client_id(clinician_client_id)
    patient_email = email_for_client_id(patient_client_id)

    if not clinician_email or not patient_email:
        return {"error": "missing_fields"}

    clinician_user = session.exec(select(User).where(User.email == clinician_email)).first()
    patient_user = session.exec(select(User).where(User.email == patient_email)).first()

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
# Gait analysis helpers (separate from ROM tracking)
# -------------------------------------------------------------------

GAIT_UPLOAD_DIR = os.path.join("app", "static", "gait_processed")
GAIT_HISTORY_DIR = os.path.join("app", "static", "gait_history")
os.makedirs(GAIT_UPLOAD_DIR, exist_ok=True)
os.makedirs(GAIT_HISTORY_DIR, exist_ok=True)

def _safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def _fill_nans(values):
    import numpy as np
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr

    valid = np.isfinite(arr)
    if not valid.any():
        return arr

    x = np.arange(arr.size)
    arr[~valid] = np.interp(x[~valid], x[valid], arr[valid])
    return arr

def _moving_average(values, window=5):
    import numpy as np
    if values.size == 0:
        return values
    window = max(1, int(window))
    if window == 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

def _detect_local_minima(values, fps):
    import numpy as np
    if values.size < 3:
        return []

    y = _fill_nans(values.tolist())
    y = _moving_average(y, window=5)

    if y.size < 3:
        return []

    minima = []
    for i in range(1, len(y) - 1):
        if y[i] <= y[i - 1] and y[i] < y[i + 1]:
            minima.append(i)

    if not minima:
        return []

    try:
        q40 = float(np.nanpercentile(y, 40))
        span = float(np.nanmax(y) - np.nanmin(y))
        threshold = q40 + (0.05 * span)
    except Exception:
        threshold = float(np.nanmedian(y))

    candidates = [i for i in minima if y[i] <= threshold]

    min_gap = max(1, int(round(fps * 0.35)))
    selected = []
    for idx in candidates:
        if not selected:
            selected.append(idx)
            continue

        if idx - selected[-1] >= min_gap:
            selected.append(idx)
        elif y[idx] < y[selected[-1]]:
            selected[-1] = idx

    return selected

def _mean_interval(event_indices, fps):
    import numpy as np
    if len(event_indices) < 2:
        return None
    times = np.array(event_indices, dtype=np.float32) / float(fps)
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    return float(np.mean(diffs))

def _confidence_score(pose_ratio, event_quality, visibility_quality):
    raw = (0.55 * pose_ratio) + (0.25 * event_quality) + (0.20 * visibility_quality)
    raw = max(0.0, min(1.0, raw))
    return round(raw * 100.0, 1)

def save_gait_history(client_id, entry):
    cid = norm_client_id(client_id)
    os.makedirs(GAIT_HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(GAIT_HISTORY_DIR, f"{cid}.json")

    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    history = loaded
        except Exception:
            history = []

    history.append(entry)

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def load_gait_history(client_id):
    cid = norm_client_id(client_id)
    history_file = os.path.join(GAIT_HISTORY_DIR, f"{cid}.json")
    if not os.path.exists(history_file):
        return []

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                return loaded
    except Exception:
        pass

    return []

def process_gait_video(filepath, client_id=None, view="side", save_output=True):
    """
    Side-view gait analysis using MediaPipe Pose.

    Outputs approximate gait metrics from one walking trial:
    - cadence
    - speed (approximate)
    - step_length
    - stride_length
    - stance_time_left
    - stance_time_right
    - symmetry
    - trunk_lean
    - peak_knee_flexion
    - confidence

    Prototype heuristic only.
    """
    import numpy as np
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    folder = os.path.dirname(filepath)
    output_path = os.path.join(folder, "gait_annotated.mp4")

    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    out = None
    if save_output and w > 0 and h > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    left_ankle_y = []
    right_ankle_y = []
    left_vis_series = []
    right_vis_series = []
    left_knee_flex_series = []
    right_knee_flex_series = []
    trunk_lean_series = []
    body_height_norm_series = []
    center_x_series = []

    total_frames = 0
    valid_pose_frames = 0

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_connections = mp.solutions.pose.POSE_CONNECTIONS

    def as_np(lm):
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    def angle(a, b, c):
        try:
            ba = as_np(a) - as_np(b)
            bc = as_np(c) - as_np(b)
            denom = np.linalg.norm(ba) * np.linalg.norm(bc)
            if denom == 0:
                return None
            cosang = float(np.dot(ba, bc) / denom)
            cosang = max(-1.0, min(1.0, cosang))
            return float(np.degrees(np.arccos(cosang)))
        except Exception:
            return None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            frame_t = total_frames / float(fps)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if not results.pose_landmarks:
                left_ankle_y.append(np.nan)
                right_ankle_y.append(np.nan)
                left_vis_series.append(0.0)
                right_vis_series.append(0.0)
                left_knee_flex_series.append(np.nan)
                right_knee_flex_series.append(np.nan)
                trunk_lean_series.append(np.nan)
                body_height_norm_series.append(np.nan)
                center_x_series.append(np.nan)

                if out is not None:
                    out.write(frame)
                continue

            valid_pose_frames += 1
            lm = results.pose_landmarks.landmark

            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            lk = lm[mp_pose.PoseLandmark.LEFT_KNEE]
            rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

            left_vis = float(np.mean([ls.visibility, lh.visibility, lk.visibility, la.visibility]))
            right_vis = float(np.mean([rs.visibility, rh.visibility, rk.visibility, ra.visibility]))
            left_vis_series.append(left_vis)
            right_vis_series.append(right_vis)

            shoulder_mid = np.array([(ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0, (ls.z + rs.z) / 2.0], dtype=np.float32)
            hip_mid = np.array([(lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0, (lh.z + rh.z) / 2.0], dtype=np.float32)
            ankle_mid = np.array([(la.x + ra.x) / 2.0, (la.y + ra.y) / 2.0, (la.z + ra.z) / 2.0], dtype=np.float32)

            dx = float(shoulder_mid[0] - hip_mid[0])
            dy = float(hip_mid[1] - shoulder_mid[1])
            trunk_lean = float(np.degrees(np.arctan2(abs(dx), abs(dy) if abs(dy) > 1e-6 else 1e-6)))
            trunk_lean_series.append(trunk_lean)

            body_height_norm = float(np.linalg.norm(shoulder_mid[:2] - ankle_mid[:2]))
            body_height_norm_series.append(body_height_norm)

            center_x_series.append(float(hip_mid[0]))

            left_knee_angle = angle(lh, lk, la)
            right_knee_angle = angle(rh, rk, ra)

            if left_knee_angle is not None:
                left_knee_flex = max(0.0, 180.0 - left_knee_angle)
            else:
                left_knee_flex = np.nan

            if right_knee_angle is not None:
                right_knee_flex = max(0.0, 180.0 - right_knee_angle)
            else:
                right_knee_flex = np.nan

            left_knee_flex_series.append(left_knee_flex)
            right_knee_flex_series.append(right_knee_flex)

            left_ankle_y.append(float(la.y))
            right_ankle_y.append(float(ra.y))

            if out is not None:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_connections)

                cv2.putText(
                    frame,
                    f"t={frame_t:.1f}s",
                    (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Lean={trunk_lean:.1f} deg",
                    (12, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                out.write(frame)

    cap.release()
    if out is not None:
        out.release()

    if total_frames == 0:
        raise Exception("No frames were read from the video")

    duration = total_frames / float(fps)

    left_events = _detect_local_minima(np.asarray(left_ankle_y, dtype=np.float32), fps)
    right_events = _detect_local_minima(np.asarray(right_ankle_y, dtype=np.float32), fps)

    all_events = sorted(set(left_events + right_events))
    cadence = (len(all_events) / duration) * 60.0 if duration > 0 and all_events else None

    left_step_interval = _mean_interval(left_events, fps)
    right_step_interval = _mean_interval(right_events, fps)

    step_time = None
    if len(all_events) >= 2:
        all_times = np.array(all_events, dtype=np.float32) / float(fps)
        step_diffs = np.diff(all_times)
        if step_diffs.size > 0:
            step_time = float(np.mean(step_diffs))

    stride_time = (step_time * 2.0) if step_time is not None else None

    center_x_filled = _fill_nans(center_x_series)
    body_height_filled = _fill_nans(body_height_norm_series)

    speed = None
    if center_x_filled.size >= 2:
        x_path = float(np.nansum(np.abs(np.diff(center_x_filled))))
        mean_height = float(np.nanmean(body_height_filled)) if np.isfinite(body_height_filled).any() else np.nan
        if np.isfinite(mean_height) and mean_height > 1e-6:
            scale_m_per_norm = 1.70 / mean_height
            speed = (x_path * scale_m_per_norm) / duration if duration > 0 else None

    stride_length = (speed * stride_time) if (speed is not None and stride_time is not None) else None
    step_length = (stride_length / 2.0) if stride_length is not None else None

    symmetry = None
    if left_step_interval is not None and right_step_interval is not None:
        base = max(left_step_interval, right_step_interval, 1e-6)
        diff_pct = abs(left_step_interval - right_step_interval) / base
        symmetry = max(0.0, 100.0 - (diff_pct * 100.0))
    elif left_events or right_events:
        base = max(len(left_events), len(right_events), 1)
        diff_pct = abs(len(left_events) - len(right_events)) / float(base)
        symmetry = max(0.0, 100.0 - (diff_pct * 100.0))

    trunk_lean_mean = None
    if len(trunk_lean_series) > 0 and np.isfinite(np.asarray(trunk_lean_series, dtype=np.float32)).any():
        trunk_lean_mean = float(np.nanmean(np.asarray(trunk_lean_series, dtype=np.float32)))

    peak_knee_flex = None
    knee_pool = np.concatenate([
        _fill_nans(left_knee_flex_series),
        _fill_nans(right_knee_flex_series),
    ]) if (left_knee_flex_series or right_knee_flex_series) else np.array([], dtype=np.float32)
    if knee_pool.size > 0 and np.isfinite(knee_pool).any():
        peak_knee_flex = float(np.nanmax(knee_pool))

    pose_ratio = valid_pose_frames / float(total_frames) if total_frames > 0 else 0.0
    event_quality = min(1.0, len(all_events) / 6.0)
    visibility_series = np.concatenate([
        np.asarray(left_vis_series, dtype=np.float32),
        np.asarray(right_vis_series, dtype=np.float32),
    ])
    visibility_quality = float(np.nanmean(visibility_series)) if visibility_series.size else 0.0
    confidence = _confidence_score(pose_ratio, event_quality, visibility_quality)

    summary_parts = []

    if cadence is None:
        summary_parts.append("Not enough gait cycles were detected for a confident summary.")
    else:
        summary_parts.append(f"Cadence estimated at {cadence:.0f} steps/min.")

    if speed is not None:
        summary_parts.append(f"Approximate walking speed: {speed:.2f} m/s.")

    if symmetry is not None:
        summary_parts.append(f"Step symmetry estimated at {symmetry:.1f}%.")

    if trunk_lean_mean is not None:
        summary_parts.append(f"Average trunk lean: {trunk_lean_mean:.1f}°.")

    if peak_knee_flex is not None:
        summary_parts.append(f"Peak knee flexion: {peak_knee_flex:.1f}°.")

    if not summary_parts:
        summary_parts.append("Side-view gait captured successfully.")

    result = {
        "view": view,
        "processed_video": output_path if save_output and os.path.exists(output_path) else None,
        "cadence": round(cadence, 1) if cadence is not None else None,
        "speed": round(speed, 2) if speed is not None else None,
        "step_length": round(step_length, 2) if step_length is not None else None,
        "stride_length": round(stride_length, 2) if stride_length is not None else None,
        "stance_time_left": round(left_step_interval, 2) if left_step_interval is not None else None,
        "stance_time_right": round(right_step_interval, 2) if right_step_interval is not None else None,
        "stride_time": round(stride_time, 2) if stride_time is not None else None,
        "symmetry": round(symmetry, 1) if symmetry is not None else None,
        "trunk_lean": round(trunk_lean_mean, 1) if trunk_lean_mean is not None else None,
        "peak_knee_flexion": round(peak_knee_flex, 1) if peak_knee_flex is not None else None,
        "confidence": confidence,
        "events_detected": len(all_events),
        "left_events": len(left_events),
        "right_events": len(right_events),
        "summary": " ".join(summary_parts),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if client_id:
        save_gait_history(
            client_id,
            {
                "client_id": norm_client_id(client_id),
                "view": view,
                "results": result,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "source_video": filepath,
            },
        )

    return result
# -------------------------------------------------------------------
# Upload & process video
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
# Gait upload & history
# -------------------------------------------------------------------

@app.post("/gait/upload")
async def gait_upload_video(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    view: str = Form("side"),
    session_id: str = Form(None),
    db: Session = Depends(get_session),
):
    t0 = time.time()
    print("GAIT UPLOAD: received request")

    client_id = norm_client_id(client_id)
    if not client_id:
        return JSONResponse(status_code=400, content={"error": "missing_client_id"})

    user = get_or_create_user(db, client_id)

    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    file_id = str(uuid.uuid4())
    upload_dir = os.path.join(GAIT_UPLOAD_DIR, client_id, file_id)
    os.makedirs(upload_dir, exist_ok=True)

    original_path = os.path.join(upload_dir, f"original{ext}")
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"GAIT UPLOAD: saved file in {time.time() - t0:.2f}s")

    try:
        result = process_gait_video(
            original_path,
            client_id=None,
            view=(view or "side").strip().lower(),
            save_output=True,
        )
    except Exception as e:
        print("GAIT UPLOAD ERROR:", e)
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "processing_failed", "detail": str(e)},
        )

    print(f"GAIT UPLOAD: processing done in {time.time() - t0:.2f}s")

    # Separate gait history file; does not touch ROM results table.
    try:
        entry = {
            "client_id": client_id,
            "file_id": file_id,
            "view": (view or "side").strip().lower(),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "results": result,
            "source_video": original_path,
        }
        save_gait_history(client_id, entry)
    except Exception as e:
        print("GAIT HISTORY SAVE ERROR:", e)
        print(traceback.format_exc())

    return {
        "message": "Upload and gait processing successful",
        "file_id": file_id,
        "results": result,
    }

@app.get("/gait/history/{client_id}")
def get_gait_history(client_id: str, session: Session = Depends(get_session)):
    cid_email = email_for_client_id(client_id)
    if not cid_email:
        return JSONResponse(status_code=400, content={"message": "missing_client_id"})

    user = session.exec(select(User).where(User.email == cid_email)).first()
    if not user:
        return {"client_id": cid_email, "history": []}

    history = load_gait_history(client_id)
    return {"client_id": cid_email, "history": history}

# -------------------------------------------------------------------
# Patient notes (legacy client_id param) — read-only for patient side
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
        "client_id": email_for_client_id(client_id),
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
    cid_email = email_for_client_id(client_id)
    if not cid_email:
        return JSONResponse(status_code=400, content={"message": "missing_client_id"})

    user = session.exec(select(User).where(User.email == cid_email)).first()
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
            return {"client_id": cid_email, "history": history}

    history_file = os.path.join(HISTORY_DIR, f"{norm_client_id(client_id)}.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
        return {"client_id": cid_email, "history": history}

    return JSONResponse(status_code=404, content={"message": "Client not found"})

# -------------------------------------------------------------------
# Symmetry save
# -------------------------------------------------------------------

@app.post("/symmetry/save")
def save_symmetry(payload: dict = Body(...), session: Session = Depends(get_session)):
    client_id = (payload.get("client_id") or "").strip().lower()
    movement = (payload.get("movement") or "").strip().lower()
    symmetry = payload.get("symmetry", None)

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

# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    create_db_and_tables()
    threading.Thread(target=_warmup_mediapipe, daemon=True).start()
