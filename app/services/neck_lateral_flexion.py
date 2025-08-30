# app/services/neck_lateral_flexion.py
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

def process_neck_lateral_flexion(
    filepath: str,
    side: str = "left",
    client_id: str = None,
    save_output: bool = True
):
    """
    Estimates neck LATERAL FLEXION (face-on) as the angle between:
      - Torso vector:  hip-midpoint → shoulder-midpoint (approx vertical)
      - Head vector:  ear (preferred) → shoulder-midpoint (fallback to nose)

    Conventions:
      - Neutral standing ≈ small angle
      - Angle increases as the ear approaches the ipsilateral shoulder
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

    # Track min/max lateral flexion
    min_lat = float("inf")
    max_lat = float("-inf")

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

            # Midpoints for shoulders and hips (3D)
            shL = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            shR = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            hpL = lm[mp_pose.PoseLandmark.LEFT_HIP]
            hpR = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            shoulder_mid = np.array([
                (shL.x + shR.x)/2.0,
                (shL.y + shR.y)/2.0,
                (shL.z + shR.z)/2.0
            ], dtype=float)
            hip_mid = np.array([
                (hpL.x + hpR.x)/2.0,
                (hpL.y + hpR.y)/2.0,
                (hpL.z + hpR.z)/2.0
            ], dtype=float)

            # Ear landmark on measured side; fallback to nose if ear is unreliable
            if side.lower() == "right":
                ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]
            else:
                ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
            head_pt = ear
            if getattr(ear, "visibility", 0.0) < 0.2:
                head_pt = lm[mp_pose.PoseLandmark.NOSE]

            head_xyz = np.array([head_pt.x, head_pt.y, head_pt.z], dtype=float)

            # Vectors
            v_torso = shoulder_mid - hip_mid          # approx vertical axis
            v_head  = head_xyz - shoulder_mid         # ear/nose relative to shoulder line

            # Angle between vectors (clamped, 0..180)
            n_t = np.linalg.norm(v_torso)
            n_h = np.linalg.norm(v_head)
            if n_t == 0.0 or n_h == 0.0:
                angle_deg = 0.0
            else:
                cosv = float(np.dot(v_torso, v_head) / (n_t * n_h))
                cosv = np.clip(cosv, -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cosv)))

            # Track extremes
            min_lat = min(min_lat, angle_deg)
            max_lat = max(max_lat, angle_deg)

            # Draw and label
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_connections)
            cv2.putText(
                frame,
                f"Neck LatFlex ({side}): {int(angle_deg)}\xb0",
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

    rom = max_lat - min_lat

    # Build summary
    summary_data = {
        "movement":    "neck_lateral_flexion",
        "side":        side,
        "min_angle":   round(min_lat, 2),
        "max_angle":   round(max_lat, 2),
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
        "min_angle":       round(min_lat, 2),
        "max_angle":       round(max_lat, 2),
        "rom":             round(rom, 2),
        "metrics_file":    json_path
    }
