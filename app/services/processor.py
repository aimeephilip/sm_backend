from app.services.shoulder_abduction import process_shoulder_abduction
from app.services.hip_flexion import process_hip_flexion

def process_video(
    filepath: str,
    movement_type: str,
    side: str = "left",
    client_id: str | None = None,
    # Accept extra kwargs used by main.py (ignored here unless you wire them through)
    session_id: str | None = None,
    compute_symmetry: bool = True,
):
    """
    Dispatch video processing by movement_type.
    session_id and compute_symmetry are accepted to match main.py but are not used here.
    """
    if movement_type == "shoulder_abduction":
        # NOTE: do NOT pass session_id/compute_symmetry because the function doesn't accept them
        return process_shoulder_abduction(filepath, side=side, client_id=client_id)
    elif movement_type == "hip_flexion":
        return process_hip_flexion(filepath, side=side, client_id=client_id)
    else:
        raise ValueError(f"Unsupported movement_type: {movement_type}")
