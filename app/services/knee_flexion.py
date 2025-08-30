# app/services/knee_flexion.py
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

def process_knee_flexion(
    filepath: str,
    side: str = "left",
    client_id: str = None,
    save_output: bool = True
):
    """
    Measures knee flexion as the internal angle at the knee (side-on recommended).

    Conventions:
      - 0° when the knee is fully straight.
      - Angle increases as the knee flexes.
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

    # Track min/max internal knee flexion
    min_int = float("inf")
    max_int = float("-inf")

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
                hp = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                kn = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
                an = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            else:
                hp = lm[mp_pose.PoseLandmark.LEFT_HIP]
                kn = lm[mp_pose.PoseLandmark.LEFT_KNEE]
                an = lm[mp_pose.PoseLandmark.LEFT_ANKLE]

            # Compute raw angle at knee using 3D points (hip–knee–ankle)
            a = np.array([hp.x, hp.y, hp.z], dtype=float)  # proximal (thigh)
            b = np.array([kn.x, kn.y, kn.z], dtype=float)  # vertex (knee)
            c = np.array([an.x, an.y, an.z], dtype=float)  # distal (shank)

            ba = a - b
            bc = c - b
            nba = np.linalg.norm(ba)
            nbc = np.linalg.norm(bc)
            if nba == 0.0 or nbc == 0.0:
                raw_angle = 180.0
            else:
                cosv = float(np.dot(ba, bc) / (nba * nbc))
                cosv = np.clip(cosv, -1.0, 1.0)
                raw_angle = float(np.degrees(np.arccos(cosv)))  # ~180 when straight

            # Internal knee flexion: 0 when straight, increases with bend
            internal = max(0.0, 180.0 - raw_angle)

            # Track extremes
            min_int = min(min_int, internal)
            max_int = max(max_int, internal)

            # Draw and label
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_connections)
            cv2.putText(
                frame,
                f"Knee Flex ({side}): {int(internal)}\xb0",
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

    rom = max_int - min_int

    # Build summary
    summary_data = {
        "movement":    "knee_flexion",
        "side":        side,
        "min_angle":   round(min_int, 2),
        "max_angle":   round(max_int, 2),
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
        "min_angle":       round(min_int, 2),
        "max_angle":       round(max_int, 2),
        "rom":             round(rom, 2),
        "metrics_file":    json_path
    }
