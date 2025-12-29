# app/utils/results.py
from sqlmodel import Session
from app.models_movement import MovementResult
from app.models import User

def save_result(
    session: Session,
    user: User,
    summary: dict,
):
    result = MovementResult(
        user_id=user.id,
        movement=summary["movement"],
        side=summary["side"],
        min_angle=summary["min_angle"],
        max_angle=summary["max_angle"],
        rom=summary["rom"],
    )
    session.add(result)
    session.commit()
