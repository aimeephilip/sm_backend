# app/services/shoulder_flexion.py
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

def process_shoulder_flexion(
    filepath: str,
    side: str = "left",
    client_id: str = None,
    save_output: bool = True
):
    """
    Measures shoulder flexion as the angle between the torso (hip→shoulder)
    and the arm (shoulder→elbow). Side-on to camera recommended.

    Conventions:
      - ~0° with arm resting by the side (vectors nearly colinear).
      - ~90° at shoulder height in front.
      - Approaches ~180° when arm is fully overhead.
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

    # Track min/max flexion
    min_flex = float("inf")
    max_flex = float("-inf")

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
                el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                hp = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            else:
                sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                el = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
                hp = lm[mp_pose.PoseLandmark.LEFT_HIP]

            # Build 3D vectors from shoulder origin
            # Torso vector: shoulder ← hip (hip->shoulder)
            v_torso = np.array([hp.x - sh.x, hp.y - sh.y, hp.z - sh.z]) * -1.0  # equivalently (sh - hp)
            # Arm vector: shoulder → elbow
            v_arm   = np.array([el.x - sh.x, el.y - sh.y, el.z - sh.z])

            # Compute angle between arm and torso, clamp for numerical stability
            dp    = np.dot(v_arm, v_torso)
            norms = np.linalg.norm(v_arm) * np.linalg.norm(v_torso)
            cosθ  = (dp / norms) if norms else 1.0
            cosθ  = np.clip(cosθ, -1.0, 1.0)
            flex_angle = float(np.degrees(np.arccos(cosθ)))  # 0..180

            # Track extremes
            min_flex = min(min_flex, flex_angle)
            max_flex = max(max_flex, flex_angle)

            # Draw and label
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_connections)
            cv2.putText(
                frame,
                f"{side.capitalize()} Shoulder Flex: {int(flex_angle)}\xb0",
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

    rom = max_flex - min_flex

    # Build summary
    summary_data = {
        "movement":    "shoulder_flexion",
        "side":        side,
        "min_angle":   round(min_flex, 2),
        "max_angle":   round(max_flex, 2),
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
        "min_angle":       round(min_flex, 2),
        "max_angle":       round(max_flex, 2),
        "rom":             round(rom, 2),
        "metrics_file":    json_path
    }
