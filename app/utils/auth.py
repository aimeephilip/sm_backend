# app/utils/auth.py
import os
from fastapi import Header, HTTPException
import firebase_admin
from firebase_admin import auth, credentials

_initialized = False

def init_firebase():
    global _initialized
    if _initialized:
        return

    # Recommended: store Firebase service account JSON in Render as an env var
    # FIREBASE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
    sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON env var not set")

    import json
    cred = credentials.Certificate(json.loads(sa_json))
    firebase_admin.initialize_app(cred)
    _initialized = True

def get_firebase_claims(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing_bearer_token")

    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="empty_token")

    try:
        init_firebase()
        decoded = auth.verify_id_token(token)
        return decoded  # contains uid, email, etc
    except Exception:
        raise HTTPException(status_code=401, detail="invalid_token")
