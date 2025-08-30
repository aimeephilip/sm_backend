# app/services/shoulder_internal_rotation.py
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

def _angle_between_2d(u, v):
    """Angle between 2D vectors (degrees, 0..180)."""
    u = np.array([u[0], u[1]], dtype=float)
    v = np.array([v[0], v[1]], dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    cosv = np.dot(u, v) / (nu * nv)
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def _perp2d(u):
    """+90° rotation in 2D (perpendicular)."""
    return np.array([-u[1], u[0]], dtype=float)

def process_shoulder_internal_rotation(
    filepath: str,
    side: str = "left",
    client_id: str = None,
    save_output: bool = True
):
    """
    Estimates shoulder INTERNAL ROTATION magnitude in the frontal plane.
    Face-on camera. Best with elbow ≈ 90° and upper arm near horizontal (abducted).

    Proxy definition:
      - Compute upper arm vector (shoulder→elbow) in 2D image (x,y).
      - Take a perpendicular to that vector (represents ideal forearm direction at 90° elbow).
      - Measure the angle between the actual forearm vector (elbow→wrist) and that perpendicular.
      - Treat this deviation (0..180°) as internal-rotation magnitude in the image plane.

    Notes:
      - This is a planar proxy (no depth). Keep torso square to camera, minimize trunk/scapular motion.
      - Side selection determines which landmarks (L/R) are used.
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

    # Track min/max rotation magnitude
    min_rot = float("inf")
    max_rot = float("-inf")

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
                wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            else:
                sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                el = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
                wr = lm[mp_pose.PoseLandmark.LEFT_WRIST]

            # 2D vectors in image plane
            v_arm   = np.array([el.x - sh.x, el.y - sh.y])   # humerus direction (frontal)
            v_fore  = np.array([wr.x - el.x, wr.y - el.y])   # forearm
            v_perp  = _perp2d(v_arm)                         # ideal forearm dir when elbow ≈ 90°

            # Rotation magnitude as deviation from perpendicular
            rot_mag = _angle_between_2d(v_fore, v_perp)      # 0..180 (planar)

            # Track extremes
            min_rot = min(min_rot, rot_mag)
            max_rot = max(max_rot, rot_mag)

            # Draw and label
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_connections)
            cv2.putText(
                frame,
                f"Shoulder IR ({side}): {int(rot_mag)}\xb0",
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

    rom = max_rot - min_rot

    # Build summary
    summary_data = {
        "movement":    "shoulder_internal_rotation",
        "side":        side,
        "min_angle":   round(min_rot, 2),
        "max_angle":   round(max_rot, 2),
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
        "min_angle":       round(min_rot, 2),
        "max_angle":       round(max_rot, 2),
        "rom":             round(rom, 2),
        "metrics_file":    json_path
    }
