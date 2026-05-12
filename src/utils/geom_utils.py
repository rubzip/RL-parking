from copy import copy

import numpy as np

from .collisions import Rectangle
from .models import CarState


def _ray_segment_intersection(ray_origin, ray_dir, p1, p2) -> Optional[float]:
    """
    Ray-segment intersection.
    Returns distance t if hit, else None.
    """
    v1 = ray_origin - p1
    v2 = p2 - p1
    v3 = np.array([-ray_dir[1], ray_dir[0]])

    denom = np.dot(v2, v3)
    if abs(denom) < 1e-8:
        return None  # Parallel

    t = np.cross(v2, v1) / denom
    u = np.dot(v1, v3) / denom

    if t >= 0 and 0.0 <= u <= 1.0:
        return t

    return None
