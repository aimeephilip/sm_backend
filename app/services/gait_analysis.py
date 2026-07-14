import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_connections = mp.solutions.pose.POSE_CONNECTIONS

GAIT_HISTORY_DIR = os.path.join("app", "static", "gait_history")


def _norm_client_id(client_id: str) -> str:
    return (client_id or "").strip().lower()


def _safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _as_np(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def _angle(a, b, c) -> Optional[float]:
    """
    Returns the internal angle ABC in degrees.
    """
    try:
        ba = _as_np(a) - _as_np(b)
        bc = _as_np(c) - _as_np(b)
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return None
        cosang = float(np.dot(ba, bc) / denom)
        cosang = max(-1.0, min(1.0, cosang))
        return float(np.degrees(np.arccos(cosang)))
    except Exception:
        return None


def _fill_nans(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr

    valid = np.isfinite(arr)
    if not valid.any():
        return arr

    x = np.arange(arr.size)
    arr[~valid] = np.interp(x[~valid], x[valid], arr[valid])
    return arr


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(1, int(window))
    if window == 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _detect_local_minima(values: np.ndarray, fps: float) -> List[int]:
    """
    Detect heel-strike-like events as local minima in ankle Y.
    """
    if values.size < 3:
        return []

    y = _fill_nans(values.tolist())
    y = _moving_average(y, window=5)

    if y.size < 3:
        return []

    minima = []
    for i in range(1, len(y) - 1):
        if y[i] <= y[i - 1] and y[i] < y[i + 1]:
            minima.append(i)

    if not minima:
        return []

    # Keep only reasonably low points (filter out tiny wiggles)
    try:
        q40 = float(np.nanpercentile(y, 40))
        span = float(np.nanmax(y) - np.nanmin(y))
        threshold = q40 + (0.05 * span)
    except Exception:
        threshold = float(np.nanmedian(y))

    candidates = [i for i in minima if y[i] <= threshold]

    # Enforce a minimum separation between events.
    min_gap = max(1, int(round(fps * 0.35)))
    selected: List[int] = []
    for idx in candidates:
        if not selected:
            selected.append(idx)
            continue

        if idx - selected[-1] >= min_gap:
            selected.append(idx)
        elif y[idx] < y[selected[-1]]:
            selected[-1] = idx

    return selected


def _mean_interval(event_indices: List[int], fps: float) -> Optional[float]:
    if len(event_indices) < 2:
        return None
    times = np.array(event_indices, dtype=np.float32) / float(fps)
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    return float(np.mean(diffs))


def _confidence_score(
    pose_ratio: float,
    event_quality: float,
    visibility_quality: float,
) -> float:
    raw = (0.55 * pose_ratio) + (0.25 * event_quality) + (0.20 * visibility_quality)
    raw = max(0.0, min(1.0, raw))
    return round(raw * 100.0, 1)


def save_gait_history(client_id: str, entry: Dict[str, Any]) -> None:
    cid = _norm_client_id(client_id)
    os.makedirs(GAIT_HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(GAIT_HISTORY_DIR, f"{cid}.json")

    history: List[Dict[str, Any]] = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    history = loaded
        except Exception:
            history = []

    history.append(entry)

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_gait_history(client_id: str) -> List[Dict[str, Any]]:
    cid = _norm_client_id(client_id)
    history_file = os.path.join(GAIT_HISTORY_DIR, f"{cid}.json")
    if not os.path.exists(history_file):
        return []

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                return loaded
    except Exception:
        pass

    return []


def process_gait_video(
    filepath: str,
    client_id: Optional[str] = None,
    view: str = "side",
    save_output: bool = True,
) -> Dict[str, Any]:
    """
    Side-view gait analysis using MediaPipe Pose.

    Outputs approximate gait metrics from one walking trial:
    - cadence
    - speed (approximate, normalized by body height in frame)
    - step_length
    - stride_length
    - stance_time_left
    - stance_time_right
    - symmetry
    - trunk_lean
    - peak_knee_flexion
    - confidence

    This is a prototype heuristic, not a clinically validated gait lab.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    folder = os.path.dirname(filepath)
    output_path = os.path.join(folder, "gait_annotated.mp4")

    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    out = None
    if save_output and w > 0 and h > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Per-frame signals
    left_ankle_y: List[float] = []
    right_ankle_y: List[float] = []
    left_vis_series: List[float] = []
    right_vis_series: List[float] = []
    left_knee_flex_series: List[float] = []
    right_knee_flex_series: List[float] = []
    trunk_lean_series: List[float] = []
    body_height_norm_series: List[float] = []
    center_x_series: List[float] = []

    total_frames = 0
    valid_pose_frames = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            frame_t = total_frames / float(fps)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if not results.pose_landmarks:
                left_ankle_y.append(np.nan)
                right_ankle_y.append(np.nan)
                left_vis_series.append(0.0)
                right_vis_series.append(0.0)
                left_knee_flex_series.append(np.nan)
                right_knee_flex_series.append(np.nan)
                trunk_lean_series.append(np.nan)
                body_height_norm_series.append(np.nan)
                center_x_series.append(np.nan)

                if out is not None:
                    out.write(frame)
                continue

            valid_pose_frames += 1
            lm = results.pose_landmarks.landmark

            # Landmarks
            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            lk = lm[mp_pose.PoseLandmark.LEFT_KNEE]
            rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Visibility
            left_vis = float(np.mean([ls.visibility, lh.visibility, lk.visibility, la.visibility]))
            right_vis = float(np.mean([rs.visibility, rh.visibility, rk.visibility, ra.visibility]))
            left_vis_series.append(left_vis)
            right_vis_series.append(right_vis)

            # Core midpoints
            shoulder_mid = np.array([(ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0, (ls.z + rs.z) / 2.0], dtype=np.float32)
            hip_mid = np.array([(lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0, (lh.z + rh.z) / 2.0], dtype=np.float32)
            ankle_mid = np.array([(la.x + ra.x) / 2.0, (la.y + ra.y) / 2.0, (la.z + ra.z) / 2.0], dtype=np.float32)

            # Trunk lean relative to vertical (0° = upright)
            dx = float(shoulder_mid[0] - hip_mid[0])
            dy = float(hip_mid[1] - shoulder_mid[1])
            trunk_lean = float(np.degrees(np.arctan2(abs(dx), abs(dy) if abs(dy) > 1e-6 else 1e-6)))
            trunk_lean_series.append(trunk_lean)

            # Approximate body height in normalized coordinates
            body_height_norm = float(np.linalg.norm(shoulder_mid[:2] - ankle_mid[:2]))
            body_height_norm_series.append(body_height_norm)

            # Approximate translation signal
            center_x_series.append(float(hip_mid[0]))

            # Knee flexion estimates (internal flexion = 180 - joint angle)
            left_knee_angle = _angle(lh, lk, la)
            right_knee_angle = _angle(rh, rk, ra)

            if left_knee_angle is not None:
                left_knee_flex = max(0.0, 180.0 - left_knee_angle)
            else:
                left_knee_flex = np.nan

            if right_knee_angle is not None:
                right_knee_flex = max(0.0, 180.0 - right_knee_angle)
            else:
                right_knee_flex = np.nan

            left_knee_flex_series.append(left_knee_flex)
            right_knee_flex_series.append(right_knee_flex)

            left_ankle_y.append(float(la.y))
            right_ankle_y.append(float(ra.y))

            if out is not None:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_connections)

                # Minimal on-frame debug text
                cv2.putText(
                    frame,
                    f"t={frame_t:.1f}s",
                    (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Lean={trunk_lean:.1f} deg",
                    (12, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                out.write(frame)

    cap.release()
    if out is not None:
        out.release()

    if total_frames == 0:
        raise Exception("No frames were read from the video")

    duration = total_frames / float(fps)

    left_events = _detect_local_minima(np.asarray(left_ankle_y, dtype=np.float32), fps)
    right_events = _detect_local_minima(np.asarray(right_ankle_y, dtype=np.float32), fps)

    all_events = sorted(set(left_events + right_events))
    cadence = (len(all_events) / duration) * 60.0 if duration > 0 and all_events else None

    left_step_interval = _mean_interval(left_events, fps)
    right_step_interval = _mean_interval(right_events, fps)

    # Average step time from merged events
    step_time = None
    if len(all_events) >= 2:
        all_times = np.array(all_events, dtype=np.float32) / float(fps)
        step_diffs = np.diff(all_times)
        if step_diffs.size > 0:
            step_time = float(np.mean(step_diffs))

    stride_time = (step_time * 2.0) if step_time is not None else None

    # Approximate spatial metrics using hip translation normalized by body height.
    center_x_filled = _fill_nans(center_x_series)
    body_height_filled = _fill_nans(body_height_norm_series)

    speed = None
    if center_x_filled.size >= 2:
        x_path = float(np.nansum(np.abs(np.diff(center_x_filled))))
        mean_height = float(np.nanmean(body_height_filled)) if np.isfinite(body_height_filled).any() else np.nan
        if np.isfinite(mean_height) and mean_height > 1e-6:
            # Heuristic conversion: normalized frame movement -> approximate meters using body height as scale.
            # 1.7 m is a rough adult height proxy for a prototype.
            scale_m_per_norm = 1.70 / mean_height
            speed = (x_path * scale_m_per_norm) / duration if duration > 0 else None

    stride_length = (speed * stride_time) if (speed is not None and stride_time is not None) else None
    step_length = (stride_length / 2.0) if stride_length is not None else None

    # Symmetry: compare left/right cadence regularity and event counts.
    symmetry = None
    if left_step_interval is not None and right_step_interval is not None:
        base = max(left_step_interval, right_step_interval, 1e-6)
        diff_pct = abs(left_step_interval - right_step_interval) / base
        symmetry = max(0.0, 100.0 - (diff_pct * 100.0))
    elif left_events or right_events:
        base = max(len(left_events), len(right_events), 1)
        diff_pct = abs(len(left_events) - len(right_events)) / float(base)
        symmetry = max(0.0, 100.0 - (diff_pct * 100.0))

    trunk_lean_mean = None
    if len(trunk_lean_series) > 0 and np.isfinite(np.asarray(trunk_lean_series, dtype=np.float32)).any():
        trunk_lean_mean = float(np.nanmean(np.asarray(trunk_lean_series, dtype=np.float32)))

    peak_knee_flex = None
    knee_pool = np.concatenate([
        _fill_nans(left_knee_flex_series),
        _fill_nans(right_knee_flex_series),
    ]) if (left_knee_flex_series or right_knee_flex_series) else np.array([], dtype=np.float32)
    if knee_pool.size > 0 and np.isfinite(knee_pool).any():
        peak_knee_flex = float(np.nanmax(knee_pool))

    pose_ratio = valid_pose_frames / float(total_frames) if total_frames > 0 else 0.0
    event_quality = min(1.0, len(all_events) / 6.0)
    visibility_series = np.concatenate([
        np.asarray(left_vis_series, dtype=np.float32),
        np.asarray(right_vis_series, dtype=np.float32),
    ])
    visibility_quality = float(np.nanmean(visibility_series)) if visibility_series.size else 0.0
    confidence = _confidence_score(pose_ratio, event_quality, visibility_quality)

    summary_parts: List[str] = []

    if cadence is None:
        summary_parts.append("Not enough gait cycles were detected for a confident summary.")
    else:
        summary_parts.append(f"Cadence estimated at {cadence:.0f} steps/min.")

    if speed is not None:
        summary_parts.append(f"Approximate walking speed: {speed:.2f} m/s.")

    if symmetry is not None:
        summary_parts.append(f"Step symmetry estimated at {symmetry:.1f}%.")

    if trunk_lean_mean is not None:
        summary_parts.append(f"Average trunk lean: {trunk_lean_mean:.1f}°.")

    if peak_knee_flex is not None:
        summary_parts.append(f"Peak knee flexion: {peak_knee_flex:.1f}°.")

    if not summary_parts:
        summary_parts.append("Side-view gait captured successfully.")

    result: Dict[str, Any] = {
        "view": view,
        "processed_video": output_path if save_output and os.path.exists(output_path) else None,
        "cadence": round(cadence, 1) if cadence is not None else None,
        "speed": round(speed, 2) if speed is not None else None,
        "step_length": round(step_length, 2) if step_length is not None else None,
        "stride_length": round(stride_length, 2) if stride_length is not None else None,
        "stance_time_left": round(left_step_interval, 2) if left_step_interval is not None else None,
        "stance_time_right": round(right_step_interval, 2) if right_step_interval is not None else None,
        "stride_time": round(stride_time, 2) if stride_time is not None else None,
        "symmetry": round(symmetry, 1) if symmetry is not None else None,
        "trunk_lean": round(trunk_lean_mean, 1) if trunk_lean_mean is not None else None,
        "peak_knee_flexion": round(peak_knee_flex, 1) if peak_knee_flex is not None else None,
        "confidence": confidence,
        "events_detected": len(all_events),
        "left_events": len(left_events),
        "right_events": len(right_events),
        "summary": " ".join(summary_parts),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if client_id:
        save_gait_history(
            client_id,
            {
                "client_id": _norm_client_id(client_id),
                "view": view,
                "results": result,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "source_video": filepath,
            },
        )

    return result
