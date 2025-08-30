# app/services/symmetry.py
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

def _parse_ts(ts: str) -> Optional[datetime]:
    try:
        # Expecting ISO with trailing Z from your processors
        if ts.endswith("Z"):
            ts = ts[:-1]
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def compute_symmetry(left_max: float, right_max: float) -> float:
    """
    Symmetry Index (0..100). 100 = perfectly matched.
    SI = 100 * (1 - abs(L - R) / max(L, R))
    Edge cases:
      - If L == R == 0 => SI = 100
      - If only one side is 0 but the other > 0 => formula still holds
    """
    L = float(left_max or 0.0)
    R = float(right_max or 0.0)
    if L == 0.0 and R == 0.0:
        return 100.0
    denom = max(L, R)
    if denom <= 0:
        return 0.0
    si = 100.0 * (1.0 - abs(L - R) / denom)
    return round(max(0.0, min(100.0, si)), 1)

def _best_candidate_by_time(
    entries: List[Dict[str, Any]],
    anchor_ts: Optional[datetime],
    window_minutes: int
) -> Optional[Dict[str, Any]]:
    """
    Given entries (already filtered for movement+opposite side), find the closest
    in time to anchor_ts within ±window_minutes. If none in window, return None.
    """
    if not entries or not anchor_ts:
        return None

    window = timedelta(minutes=window_minutes)
    best = None
    best_dt = None
    for e in entries:
        e_ts = _parse_ts(e.get("timestamp", ""))
        if not e_ts:
            continue
        dt = abs(e_ts - anchor_ts)
        if dt <= window and (best_dt is None or dt < best_dt):
            best = e
            best_dt = dt
    return best

def find_pair_for(
    entry: Dict[str, Any],
    history: List[Dict[str, Any]],
    window_minutes: int = 30
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Try to find the opposite-side pair for the given entry in order:
      1) Same session_id (if present)
      2) Closest-in-time within ±window_minutes
      3) Latest opposite-side record (fallback)
    Returns: (paired_entry_or_None, source_string)
             source_string in {"same_session","time_window","latest","none"}
    """
    movement = entry.get("movement")
    side = (entry.get("side") or "").lower()
    opposite = "left" if side == "right" else "right"
    session_id = entry.get("session_id")
    ts = _parse_ts(entry.get("timestamp", ""))

    # Filter by movement + opposite side + ignore the same record if identical
    opp_entries = [
        e for e in history
        if (e.get("movement") == movement
            and (e.get("side") or "").lower() == opposite)
    ]
    if not opp_entries:
        return None, "none"

    # 1) same_session
    if session_id:
        same_session = [e for e in opp_entries if e.get("session_id") == session_id]
        if same_session:
            # If multiple, pick the one closest in time
            candidate = _best_candidate_by_time(same_session, ts, window_minutes=10) or same_session[0]
            return candidate, "same_session"

    # 2) closest in time within window
    candidate = _best_candidate_by_time(opp_entries, ts, window_minutes)
    if candidate:
        return candidate, "time_window"

    # 3) latest opposite side (by timestamp)
    def _key(e):
        t = _parse_ts(e.get("timestamp", ""))
        return t or datetime.min

    latest = sorted(opp_entries, key=_key, reverse=True)[0]
    return latest, "latest"
