import numpy as np
import pyopencl as cl
from pathlib import Path
from typing import Any
from numpy.typing import NDArray

import Solver.Kernel.kernel_config as kernel_config

mf = cl.mem_flags

FIELD_DTYPE = kernel_config.FIELD_DTYPE
IDENTITY_4 = np.eye(4)
ZERO_4 = np.zeros((4, 4))


def load_program(
    context: cl.Context,
    path: Path,
) -> tuple[cl.Program, dict[str, cl.Kernel]]:
    with path.open("r", encoding="utf-8") as f:
        program = cl.Program(
            context,
            f.read(),
        ).build(
            options=[
                f"-DTILE_SIZE={kernel_config.TILE_SIZE}",
            ]
        )

    kernels = {kernel.function_name: kernel for kernel in program.all_kernels()}

    return program, kernels


def to_device(context, array):
    """NumPy-Array copy to OpenCL-Buffer."""
    return cl.Buffer(
        context,
        mf.READ_WRITE | mf.COPY_HOST_PTR,
        hostbuf=array,
    )


def device_array(context, shape, dtype):
    """Allocate uninitialized OpenCL buffer."""
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize

    return cl.Buffer(
        context,
        mf.READ_WRITE,
        size=size,
    )


def zeros_device(context, shape, dtype=FIELD_DTYPE):
    return to_device(context, np.zeros(shape, dtype=dtype))


def full_device(context, shape, value, dtype=FIELD_DTYPE):
    return to_device(context, np.full(shape, value, dtype=dtype))


def transform_source_velocities(
    source_values: dict[str, NDArray], inverse_linear: NDArray
) -> None:
    """Convert only World Space source velocities into the solver frame."""
    if source_values["velocity_x"].size == 0:
        return
    vectors = np.column_stack(
        (
            source_values["velocity_x"],
            source_values["velocity_y"],
            source_values["velocity_z"],
        )
    )
    world_mask = ~source_values["velocity_local"]
    vectors[world_mask] = vectors[world_mask] @ inverse_linear.T
    source_values["velocity_x"][:] = vectors[:, 0]
    source_values["velocity_y"][:] = vectors[:, 1]
    source_values["velocity_z"][:] = vectors[:, 2]


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


def get_matrix_data(times: Any, matrices: Any, rates: Any, time_value: float) -> Any:
    """
    Interpolate a world matrix and return its current element-wise rate.

    Within an animation interval, the transform is evaluated linearly between
    its two surrounding samples. Times outside the sampled range clamp to the
    first or last interval.
    """
    n = matrices.shape[0]

    if n == 0:
        return IDENTITY_4, ZERO_4

    if n == 1:
        return matrices[0], ZERO_4

    if time_value <= times[0]:
        idx = 0
        alpha = 0.0

    elif time_value >= times[-1]:
        idx = n - 2
        alpha = 1.0

    else:
        idx = (
            np.searchsorted(
                times,
                time_value,
                side="right",
            )
            - 1
        )

        dt = times[idx + 1] - times[idx]

        if dt <= 0:
            return matrices[idx], ZERO_4

        alpha = (time_value - times[idx]) / dt

    rate = rates[idx]

    matrix = matrices[idx] + rate * ((times[idx + 1] - times[idx]) * alpha)

    return matrix, rate
