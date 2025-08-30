from app.services.shoulder_abduction import process_shoulder_abduction
from app.services.hip_flexion import process_hip_flexion

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
    if movement_type == "shoulder_abduction":
        return process_shoulder_abduction(
            filepath,
            side=side,
            client_id=client_id,
        )
    elif movement_type == "hip_flexion":
        return process_hip_flexion(
            filepath,
            side=side,
            client_id=client_id,
        )
    else:
        raise ValueError(f"Unsupported movement_type: {movement_type}")
