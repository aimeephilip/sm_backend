# app/services/hip_flexion.py
import cv2
import mediapipe as mp
import os
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

from app.utils.history import save_to_history, get_history
from app.services.symmetry import find_pair_for, compute_symmetry

mp_pose        = mp.solutions.pose
mp_drawing     = mp.solutions.drawing_utils
mp_connections = mp.solutions.pose.POSE_CONNECTIONS

def process_hip_flexion(
    filepath: str,
    side: str = "left",
    client_id: Optional[str] = None,
    save_output: bool = True,
    session_id: Optional[str] = None,
    compute_sym: bool = True,
) -> Dict[str, Any]:
    # Open video file
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    # Prepare output paths
    folder      = os.path.dirname(filepath)
    output_path = os.path.join(folder, "pose.mp4")

    # Video writer setup
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    else:
        out = None

    # Initialize min/max flexion
    min_internal = float("inf")
    max_internal = float("-inf")

    # Run MediaPipe Pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Detect landmarks
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            if not results.pose_landmarks:
                if out:
                    out.write(frame)
                continue

            lm = results.pose_landmarks.landmark
            # Choose left or right side
            if (side or "").lower() == "right":
                sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                hp = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                kn = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            else:
                sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                hp = lm[mp_pose.PoseLandmark.LEFT_HIP]
                kn = lm[mp_pose.PoseLandmark.LEFT_KNEE]

            # Build vectors from hip origin
            v_torso = np.array([sh.x - hp.x, sh.y - hp.y, sh.z - hp.z])
            v_thigh = np.array([kn.x - hp.x, kn.y - hp.y, kn.z - hp.z])

            # Compute external raw angle
            dp    = np.dot(v_torso, v_thigh)
            norms = np.linalg.norm(v_torso) * np.linalg.norm(v_thigh)
            cos_theta  = dp / norms if norms else 1.0
            cos_theta  = np.clip(cos_theta, -1.0, 1.0)
            raw_angle = np.degrees(np.arccos(cos_theta))

            # Internal hip flexion angle (0° = standing)
            internal_angle = max(0.0, 180.0 - raw_angle)

            # Track extremes
            min_internal = min(min_internal, internal_angle)
            max_internal = max(max_internal, internal_angle)

            # Draw landmarks and label
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_connections)
            cv2.putText(
                frame,
                f"Hip Flex ({side}): {int(internal_angle)}°",
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

    # Compute ROM
    rom = max_internal - min_internal

    # Build summary entry
    summary_data = {
        "movement":    "hip_flexion",
        "side":        side,
        "min_angle":   round(min_internal, 2),
        "max_angle":   round(max_internal, 2),
        "rom":         round(rom, 2),
        "timestamp":   datetime.utcnow().isoformat() + "Z"
    }
    if session_id:
        summary_data["session_id"] = session_id
    if client_id:
        summary_data["client_id"] = client_id

    # Save to client history if provided
    if client_id:
        save_to_history(client_id, summary_data)

    # Prepare symmetry (optional)
    symmetry_block = None
    if compute_sym and client_id:
        history = get_history(client_id)
        pair, source = find_pair_for(summary_data, history, window_minutes=30)
        if pair:
            # identify left/right max for SI
            this_side = (side or "").lower()
            if this_side == "left":
                L = summary_data["max_angle"]
                R = pair.get("max_angle", 0.0)
            else:
                R = summary_data["max_angle"]
                L = pair.get("max_angle", 0.0)
            si = compute_symmetry(L, R)
            symmetry_block = {
                "index": si,
                "movement": "hip_flexion",
                "reference": {
                    "side": pair.get("side"),
                    "max_angle": round(float(pair.get("max_angle", 0.0)), 2),
                    "source": source
                }
            }
        else:
            needed = "right" if (side or "").lower() == "left" else "left"
            symmetry_block = {"needed_side": needed}

    # Save summary JSON in session folder
    json_payload = dict(summary_data)
    if symmetry_block is not None:
        json_payload["symmetry"] = symmetry_block

    json_path = os.path.join(folder, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    # Return results
    result = {
        "processed_video": output_path,
        "min_angle":       round(min_internal, 2),
        "max_angle":       round(max_internal, 2),
        "rom":             round(rom, 2),
        "metrics_file":    json_path
    }
    if symmetry_block is not None:
        result["symmetry"] = symmetry_block

    return result
