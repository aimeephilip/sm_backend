# app/utils/auth.py
import os
import json
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def init_firebase():
    # Prevent double-init
    if firebase_admin._apps:
        return

    sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON not set")

    cred = credentials.Certificate(json.loads(sa_json))
    firebase_admin.initialize_app(cred)

# Initialise at import time (OK for Render)
init_firebase()

def get_firebase_claims(
    creds: HTTPAuthorizationCredentials = security,
) -> dict:
    """
    FastAPI dependency.
    Expects: Authorization: Bearer <Firebase ID Token>
    Returns: {"uid": ..., "email": ..., "claims": {...}}
    """
    try:
        decoded = auth.verify_id_token(creds.credentials)
        return {
            "uid": decoded.get("uid") or decoded.get("sub"),
            "email": decoded.get("email"),
            "claims": decoded,
        }
    except Exception:
        raise HTTPException(status_code=401, detail="invalid_token")
