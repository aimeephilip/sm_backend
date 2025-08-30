# app/services/processor.py
from app.services.shoulder_abduction import process_shoulder_abduction
from app.services.hip_flexion import process_hip_flexion

def process_video(
    filepath: str,
    movement_type: str,
    side: str = "left",
    client_id: str = None,
    session_id: str = None,
    compute_symmetry: bool = True,
):
    if movement_type == "shoulder_abduction":
        return process_shoulder_abduction(
            filepath,
            side=side,
            client_id=client_id,
            session_id=session_id,
            compute_sym=compute_symmetry,
        )
    elif movement_type == "hip_flexion":
        return process_hip_flexion(
            filepath,
            side=side,
            client_id=client_id,
            session_id=session_id,
            compute_sym=compute_symmetry,
        )
    else:
        raise ValueError(f"Unsupported movement_type: {movement_type}")
