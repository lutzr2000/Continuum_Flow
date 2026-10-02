import pyopencl as cl
import numpy as np

from typing import Any

import Solver.Kernel.kernel_config as kernel_config

# Boundary mode encoding:
# 0 = outflow, 1 = inflow, 2 = no-slip wall, 3 = slip wall

SIDE_TO_AXIS_AND_INDEX = {
    "x_low": (0, 0),
    "x_high": (0, 1),
    "y_low": (1, 0),
    "y_high": (1, 1),
    "z_low": (2, 0),
    "z_high": (2, 1),
}


def convert_bc_config_format(bc_config: dict[str, Any]) -> Any:
    """
    Normalize user-facing domain boundary settings for CUDA kernel dispatch.

    Text boundary modes are converted to compact integer codes, velocity
    components become floats, and optional temperature aliases are preserved.
    """
    converted = {}
    type_map = {
        "OUTFLOW": 0,
        "INFLOW": 1,
        "WALL": 2,
        "SLIP": 3,
        "SLIP_WALL": 3,
    }

    for side in SIDE_TO_AXIS_AND_INDEX:
        face_cfg = (bc_config or {}).get(side, {})
        bc_type = face_cfg.get("type", 0)
        if isinstance(bc_type, str):
            bc_type = type_map.get(bc_type.strip().upper(), 0)
        else:
            bc_type = int(bc_type)

        velocity = face_cfg.get("velocity") or (0.0, 0.0, 0.0)
        converted_face = {
            "type": bc_type,
            "u": float(velocity[0]) if len(velocity) > 0 else 0.0,
            "v": float(velocity[1]) if len(velocity) > 1 else 0.0,
            "w": float(velocity[2]) if len(velocity) > 2 else 0.0,
        }

        if "temperature" in face_cfg:
            converted_face["temperature"] = float(face_cfg["temperature"])
        if "T" in face_cfg:
            converted_face["T"] = float(face_cfg["T"])

        converted[side] = converted_face

    return converted


def domain_bc(
    queue: cl.CommandQueue,
    boundary_kernels: dict[str, cl.Kernel],
    u: Any,
    v: Any,
    w: Any,
    p: Any,
    T: Any,
    smoke: Any,
    fuel: Any,
    bc_config: dict[str, Any],
    index_tile_map: Any,
    tile_shape: tuple[int, int, int],
    ref_temp: Any,
    u_initial: float,
    v_initial: float,
    w_initial: float,
    nx: int,
    ny: int,
    nz: int,
) -> Any:
    """
    Apply all configured domain boundary conditions to the GPU field state.

    All six domain faces are packed into scalar launch arguments and applied in
    one GPU kernel to reduce launch overhead.
    """
    bc_config = convert_bc_config_format(bc_config)

    face_args = {}

    for side in SIDE_TO_AXIS_AND_INDEX:
        bc = bc_config[side]
        bc_mode = int(bc["type"])
        temperature = bc.get("T", bc.get("temperature"))

        face_args[side] = (
            np.int32(bc_mode),
            np.float32(bc.get("u", 0.0)),
            np.float32(bc.get("v", 0.0)),
            np.float32(bc.get("w", 0.0)),
            np.float32(0.0 if temperature is None else temperature),
            np.int32(temperature is not None),
        )

    local_work_size = (
        kernel_config.THREADS_PER_BLOCK_2D[0],
        kernel_config.THREADS_PER_BLOCK_2D[0],
        kernel_config.THREADS_PER_BLOCK_2D[1],
    )

    global_work_size = (
        ((nx + local_work_size[0] - 1) // local_work_size[0]) * local_work_size[0],
        ((ny + local_work_size[1] - 1) // local_work_size[1]) * local_work_size[1],
        ((nz + local_work_size[2] - 1) // local_work_size[2]) * local_work_size[2],
    )

    boundary_kernels["domain_bc"](
        queue,
        global_work_size,
        local_work_size,
        u,
        v,
        w,
        p,
        T,
        smoke,
        fuel,
        index_tile_map,
        np.float32(ref_temp),
        np.float32(u_initial),
        np.float32(v_initial),
        np.float32(w_initial),
        *face_args["x_low"],
        *face_args["x_high"],
        *face_args["y_low"],
        *face_args["y_high"],
        *face_args["z_low"],
        *face_args["z_high"],
        np.int32(nx),
        np.int32(ny),
        np.int32(nz),
        np.int32(tile_shape[1]),
        np.int32(tile_shape[2]),
    )

    return u, v, w, p, T, smoke, fuel
