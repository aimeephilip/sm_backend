# app/services/hip_abduction.py
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

def process_hip_abduction(
    filepath: str,
    side: str = "left",
    client_id: str = None,
    save_output: bool = True
):
    """
    Measures hip abduction (face-on) as the angle between:
      - Torso vector: hip → shoulder (approx vertical in frontal plane)
      - Thigh vector: hip → knee (moves laterally with abduction)

    Conventions:
      - Neutral standing ≈ small angle (~0–10° depending on posture).
      - Angle increases as the leg abducts laterally.
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

    # Track min/max hip abduction
    min_abd = float("inf")
    max_abd = float("-inf")

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
                sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                hp = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                kn = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            else:
                sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                hp = lm[mp_pose.PoseLandmark.LEFT_HIP]
                kn = lm[mp_pose.PoseLandmark.LEFT_KNEE]

            # Build 3D vectors from hip origin
            v_torso = np.array([sh.x - hp.x, sh.y - hp.y, sh.z - hp.z])   # hip → shoulder
            v_thigh = np.array([kn.x - hp.x, kn.y - hp.y, kn.z - hp.z])   # hip → knee

            # Compute angle between thigh and torso, clamp for numerical stability
            dp    = np.dot(v_thigh, v_torso)
            norms = np.linalg.norm(v_thigh) * np.linalg.norm(v_torso)
            cosθ  = (dp / norms) if norms else 1.0
            cosθ  = np.clip(cosθ, -1.0, 1.0)
            abd_angle = float(np.degrees(np.arccos(cosθ)))  # 0..180

            # Track extremes
            min_abd = min(min_abd, abd_angle)
            max_abd = max(max_abd, abd_angle)

            # Draw and label
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_connections)
            cv2.putText(
                frame,
                f"Hip Abduction ({side}): {int(abd_angle)}\xb0",
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

    rom = max_abd - min_abd

    # Build summary
    summary_data = {
        "movement":    "hip_abduction",
        "side":        side,
        "min_angle":   round(min_abd, 2),
        "max_angle":   round(max_abd, 2),
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
        "min_angle":       round(min_abd, 2),
        "max_angle":       round(max_abd, 2),
        "rom":             round(rom, 2),
        "metrics_file":    json_path
    }
