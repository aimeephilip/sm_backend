# app/services/ankle_dorsiflexion.py
import cv2
import mediapipe as mp
import os
import json
import numpy as np
from datetime import datetime
from app.utils.history import save_to_history

mp_pose        = mp.solutions.pose
mp_drawing     = mp.solutions.drawing_utils
mp_connections = mp.solutions.pose.POSE_CONNECTIONS

def process_ankle_dorsiflexion(
    filepath: str,
    side: str = "left",
    client_id: str = None,
    save_output: bool = True
):
    """
    Measures ANKLE DORSIFLEXION (side-on) as a positive angle relative to neutral.
    Geometry:
      - Raw angle at ankle between shank (knee→ankle) and foot (ankle→foot_index).
      - Neutral standing ~ 90°. Dorsiflexion decreases this raw angle below 90°.
      - We report DORSIFLEXION = max(0, 90 - raw_angle) so:
            neutral  -> ~0°
            dorsiflex-> positive degrees (e.g., 15-20°)
            plantar  -> 0° (not negative)

    Landmarks:
      - Knee, Ankle, Foot_Index on selected side.

    Returns min/max of dorsiflexion magnitude and ROM over the clip.
    """
    # Open video
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    # Prepare output paths
    folder      = os.path.dirname(filepath)
    output_path = os.path.join(folder, "pose.mp4")

    # Video writer setup
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps    = cap.get(cv2.CAP_PROP_FPS)
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    else:
        out = None

    # Track min/max dorsiflexion magnitude
    min_dorsi = float("inf")
    max_dorsi = float("-inf")

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Pose detection
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            if not results.pose_landmarks:
                if out:
                    out.write(frame)
                continue

            lm = results.pose_landmarks.landmark
            # Choose side landmarks
            if side.lower() == "right":
                kn = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
                an = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                ft = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            else:
                kn = lm[mp_pose.PoseLandmark.LEFT_KNEE]
                an = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                ft = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

            # Build vectors at the ankle vertex (use 3D to be consistent with other modules)
            a = np.array([kn.x - an.x, kn.y - an.y, kn.z - an.z], dtype=float)  # shank: ankle->knee
            b = np.array([ft.x - an.x, ft.y - an.y, ft.z - an.z], dtype=float)  # foot:  ankle->foot index

            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0.0 or nb == 0.0:
                raw_angle = 90.0  # safe fallback ~neutral
            else:
                cosv = float(np.dot(a, b) / (na * nb))
                cosv = np.clip(cosv, -1.0, 1.0)
                raw_angle = float(np.degrees(np.arccos(cosv)))  # ~90 neutral, <90 dorsiflexion

            # Dorsiflexion magnitude (0 at neutral; positive as foot approaches shin)
            dorsiflex = max(0.0, 90.0 - raw_angle)

            # Track extremes
            min_dorsi = min(min_dorsi, dorsiflex)
            max_dorsi = max(max_dorsi, dorsiflex)

            # Draw and label
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_connections)
            cv2.putText(
                frame,
                f"Ankle Dorsi ({side}): {int(dorsiflex)}\xb0",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            if out:
                out.write(frame)

    cap.release()
    if out:
        out.release()

    rom = max_dorsi - min_dorsi

    # Build summary
    summary_data = {
        "movement":    "ankle_dorsiflexion",
        "side":        side,
        "min_angle":   round(min_dorsi, 2),
        "max_angle":   round(max_dorsi, 2),
        "rom":         round(rom, 2),
        "timestamp":   datetime.utcnow().isoformat() + "Z"
    }

    # Save to history
    if client_id:
        save_to_history(client_id, summary_data)

    # Write JSON summary
    json_path = os.path.join(folder, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    return {
        "processed_video": output_path,
        "min_angle":       round(min_dorsi, 2),
        "max_angle":       round(max_dorsi, 2),
        "rom":             round(rom, 2),
        "metrics_file":    json_path
    }
