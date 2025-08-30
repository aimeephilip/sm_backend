# app/utils/history.py
import os
import json
from typing import List, Dict, Any

HISTORY_DIR = "app/static/history"

def save_to_history(client_id: str, movement_data: dict):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(HISTORY_DIR, f"{client_id}.json")

    # Load existing history if it exists
    history: List[Dict[str, Any]]
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []
    else:
        history = []

    # Append new entry
    history.append(movement_data)

    # Save updated history
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

def get_history(client_id: str) -> List[Dict[str, Any]]:
    """
    Read-only helper to fetch a client's full history list.
    Returns [] if none.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(HISTORY_DIR, f"{client_id}.json")
    if not os.path.exists(history_file):
        return []
    try:
        with open(history_file, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []
