import os
import json

HISTORY_DIR = "app/static/history"

def save_to_history(client_id: str, movement_data: dict):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(HISTORY_DIR, f"{client_id}.json")

    # Load existing history if it exists
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []

    # Append new entry
    history.append(movement_data)

    # Save updated history
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
