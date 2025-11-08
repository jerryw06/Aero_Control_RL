# frames_px4_isaac.py
"""
Minimal global-frame translation between IsaacSim (treat as ENU-ish: x=E, y=N, z=Up)
and PX4 (NED: x=North, y=East, z=Down).
We only really need Z for vertical RL, but include full 3D for completeness.
"""

import numpy as np

def enu_to_ned(v_enu: np.ndarray) -> np.ndarray:
    # ENU -> NED mapping: [x_ned, y_ned, z_ned] = [y_enu, x_enu, -z_enu]
    x_ned = v_enu[1]
    y_ned = v_enu[0]
    z_ned = -v_enu[2]
    return np.array([x_ned, y_ned, z_ned], dtype=float)

def ned_to_enu(v_ned: np.ndarray) -> np.ndarray:
    # inverse: [x_enu, y_enu, z_enu] = [y_ned, x_ned, -z_ned]
    x_enu = v_ned[1]
    y_enu = v_ned[0]
    z_enu = -v_ned[2]
    return np.array([x_enu, y_enu, z_enu], dtype=float)

def isaac_up_accel_to_px4_ned(a_up: float) -> np.ndarray:
    """
    Map a scalar 'upward' acceleration (Isaac Z-up) to PX4 NED acceleration vector.
    Positive 'a_up' means accelerate upward in Isaac; in NED, that's negative Z.
    """
    return np.array([0.0, 0.0, -a_up], dtype=float)
