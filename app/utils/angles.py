import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle at point b given three 3D points a, b, and c.
    Points must be in [x, y, z] format.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
    angle = np.degrees(np.arccos(cosine_angle))

    return angle
