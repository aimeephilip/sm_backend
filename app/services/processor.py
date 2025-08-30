# app/services/processor.py
from app.services.shoulder_abduction import process_shoulder_abduction
from app.services.hip_flexion import process_hip_flexion

# New imports
from app.services.shoulder_flexion import process_shoulder_flexion
from app.services.elbow_flexion import process_elbow_flexion
from app.services.shoulder_internal_rotation import process_shoulder_internal_rotation
from app.services.neck_flexion import process_neck_flexion
from app.services.neck_lateral_flexion import process_neck_lateral_flexion
from app.services.hip_abduction import process_hip_abduction
from app.services.knee_flexion import process_knee_flexion
from app.services.ankle_dorsiflexion import process_ankle_dorsiflexion


def process_video(
    filepath: str,
    movement_type: str,
    side: str = "left",
    client_id: str | None = None,
    # Accepted to match main.py — currently unused by the underlying functions.
    session_id: str | None = None,
    compute_symmetry: bool = True,
):
    """
    Dispatch video processing by movement_type.

    session_id and compute_symmetry are accepted to keep the function signature
    aligned with the upload route. They are not used by the current processing
    functions; wire them through later if you add that feature.
    """
    mt = (movement_type or "").strip().lower()

    if mt == "shoulder_abduction":
        return process_shoulder_abduction(filepath, side=side, client_id=client_id)

    elif mt == "hip_flexion":
        return process_h
        # ^^^ DELETE THIS LINE if it exists in your file. It was an artifact.
        # The correct line is immediately below:

    elif mt == "hip_flexion":
        return process_hip_flexion(filepath, side=side, client_id=client_id)

    elif mt == "shoulder_flexion":
        return process_shoulder_flexion(filepath, side=side, client_id=client_id)

    elif mt == "elbow_flexion":
        return process_elbow_flexion(filepath, side=side, client_id=client_id)

    elif mt == "shoulder_internal_rotation":
        return process_shoulder_internal_rotation(filepath, side=side, client_id=client_id)

    elif mt == "neck_flexion":
        return process_neck_flexion(filepath, side=side, client_id=client_id)

    elif mt == "neck_lateral_flexion":
        return process_neck_lateral_flexion(filepath, side=side, client_id=client_id)

    elif mt == "hip_abduction":
        return process_hip_abduction(filepath, side=side, client_id=client_id)

    elif mt == "knee_flexion":
        return process_knee_flexion(filepath, side=side, client_id=client_id)

    elif mt == "ankle_dorsiflexion":
        return process_ankle_dorsiflexion(filepath, side=side, client_id=client_id)

    else:
        raise ValueError(f"Unsupported movement_type: {movement_type}")
