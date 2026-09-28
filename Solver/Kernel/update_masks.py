import numpy as np
from typing import Any

IDENTITY_4 = np.eye(4)


def prepare_matrix_data(mesh_object: Any) -> Any:
    """
    Normalize transform-animation samples and precompute per-interval rates.

    Missing timelines and matrices fall back to a single identity transform.
    Samples are truncated to their shared length; positive-duration intervals
    receive element-wise matrix rates for later interpolation.
    """
    animation = mesh_object.get("transform_animation") or {}

    times = animation.get("times")

    if times is None:
        timeline = mesh_object.get("animation_timeline") or {}
        times = timeline.get("times")

    if times is None:
        times = np.array([0.0], dtype=np.float64)
    else:
        times = np.asarray(times)

    matrices = animation.get("matrices_world")

    if matrices is None:
        matrices = IDENTITY_4[None, ...]
    else:
        matrices = np.asarray(matrices).reshape(-1, 4, 4)

    n = min(times.shape[0], matrices.shape[0])

    times = times[:n]
    matrices = matrices[:n]

    if n > 1:
        dt = np.diff(times)
        delta = np.diff(matrices, axis=0)

        rates = np.zeros_like(delta)

        valid = dt > 0
        rates[valid] = delta[valid] / dt[valid, None, None]

    else:
        rates = None

    return times, matrices, rates
