# app/utils/auth.py
import os
import json
import time
import logging
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("uvicorn.error")

security = HTTPBearer(auto_error=False)

def init_firebase():
    # Prevent double-init
    if firebase_admin._apps:
        return

    sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON not set")

    try:
        cred_dict = json.loads(sa_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"FIREBASE_SERVICE_ACCOUNT_JSON invalid JSON: {e}")

    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

    # Helpful startup log: confirms which project your backend is actually using
    project_id = cred_dict.get("project_id")
    logger.info(f"[Firebase Admin] Initialized for project_id={project_id}")

# Initialise at import time (OK for Render)
init_firebase()


def get_firebase_claims(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    FastAPI dependency.
    Expects: Authorization: Bearer <Firebase ID Token>
    Returns: {"uid": ..., "email": ..., "claims": {...}}
    """
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="missing_token")

    token = creds.credentials.strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing_token")

    # If you set this env var, we enforce tokens come from the correct Firebase project.
    expected_project_id = os.getenv("FIREBASE_PROJECT_ID")  # e.g. sm-backend-eu-50498

    try:
        decoded = auth.verify_id_token(token, check_revoked=False)

        # Optional but strongly recommended: enforce correct Firebase project
        if expected_project_id:
            iss = decoded.get("iss", "")
            aud = decoded.get("aud", "")
            expected_iss = f"https://securetoken.google.com/{expected_project_id}"

            if aud != expected_project_id or iss != expected_iss:
                logger.warning(
                    "[Auth] Token project mismatch "
                    f"aud={aud} iss={iss} expected_aud={expected_project_id} expected_iss={expected_iss}"
                )
                raise HTTPException(status_code=401, detail="invalid_token_project")

        uid = decoded.get("uid") or decoded.get("sub")
        return {
            "uid": uid,
            "email": decoded.get("email"),
            "claims": decoded,
        }

    except HTTPException:
        # preserve our own errors
        raise

    except Exception as e:
        # Log the real reason for debugging on Render
        # Common reasons: wrong service account project, invalid signature, token expired, clock skew
        logger.exception(f"[Auth] verify_id_token failed: {type(e).__name__}: {e}")

        # Extra hint for clock skew debugging:
        try:
            # decode without verification to inspect iat/exp (safe for debugging, not trust)
            import jwt
            unverified = jwt.decode(token, options={"verify_signature": False})
            iat = unverified.get("iat")
            exp = unverified.get("exp")
            now = int(time.time())
            logger.warning(f"[Auth Debug] now={now} iat={iat} exp={exp} aud={unverified.get('aud')} iss={unverified.get('iss')}")
        except Exception:
            pass

        raise HTTPException(status_code=401, detail="invalid_token")
