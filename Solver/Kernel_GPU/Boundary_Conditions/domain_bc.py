from numba import cuda

from Solver.Kernel_GPU.kernel_config import THREADS_PER_BLOCK_2D

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

def convert_bc_config_format(bc_config):
    converted = {}
    type_map = {
        "OUTFLOW": 0,
        "INFLOW": 1,
        "WALL": 2,
        "SLIP": 3,
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


@cuda.jit(cache=True)
def _pressure_poisson_apply_neumann_bcs(p):
    """
    applies the hard-coded zero-gradient pressure boundary conditions on all
    six domain faces on the GPU.

    The pressure Poisson solve uses homogeneous Neumann boundary conditions,
    meaning the pressure at the boundary is copied from the adjacent interior
    cell. This kernel writes the boundary values after each iteration so
    the next iteration starts from a pressure field with valid boundary values.

    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = p.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if i == 0:
        p[i, j, k] = p[1, j, k]
    elif i == nx - 1:
        p[i, j, k] = p[nx - 2, j, k]

    if j == 0:
        p[i, j, k] = p[i, 1, k]
    elif j == ny - 1:
        p[i, j, k] = p[i, ny - 2, k]

    if k == 0:
        p[i, j, k] = p[i, j, 1]
    elif k == nz - 1:
        p[i, j, k] = p[i, j, nz - 2]


@cuda.jit(device=True, cache=True)
def _apply_face_state(
    u,
    v,
    w,
    p,
    T,
    smoke,
    fuel,
    i,
    j,
    k,
    src_i,
    src_j,
    src_k,
    axis,
    side_index,
    bc_mode,
    u_value,
    v_value,
    w_value,
    temp_value,
    use_temp,
):
    """
    applies one configured domain-face boundary condition to a single GPU cell.

    The device helper updates velocity based on the selected boundary mode,
    copies pressure from the adjacent interior cell, and propagates scalar
    fields from the neighbor unless an explicit face temperature is configured.
    It is called by the domain boundary kernel once per matching face, so edge
    and corner cells are processed sequentially in a fixed order.
    """
    neighbor_u = u[src_i, src_j, src_k]
    neighbor_v = v[src_i, src_j, src_k]
    neighbor_w = w[src_i, src_j, src_k]
    neighbor_smoke = smoke[src_i, src_j, src_k]
    neighbor_fuel = fuel[src_i, src_j, src_k]

    if bc_mode == 0:
        u[i, j, k] = neighbor_u
        v[i, j, k] = neighbor_v
        w[i, j, k] = neighbor_w
        if axis == 0:
            u[i, j, k] = (
                min(neighbor_u, 0.0) if side_index == 0 else max(neighbor_u, 0.0)
            )
        elif axis == 1:
            v[i, j, k] = (
                min(neighbor_v, 0.0) if side_index == 0 else max(neighbor_v, 0.0)
            )
        else:
            w[i, j, k] = (
                min(neighbor_w, 0.0) if side_index == 0 else max(neighbor_w, 0.0)
            )
    elif bc_mode == 1:
        u[i, j, k] = u_value
        v[i, j, k] = v_value
        w[i, j, k] = w_value
    elif bc_mode == 2:
        u[i, j, k] = 0.0
        v[i, j, k] = 0.0
        w[i, j, k] = 0.0
    else:
        u[i, j, k] = 0.0 if axis == 0 else neighbor_u
        v[i, j, k] = 0.0 if axis == 1 else neighbor_v
        w[i, j, k] = 0.0 if axis == 2 else neighbor_w

    p[i, j, k] = p[src_i, src_j, src_k]
    T[i, j, k] = temp_value if use_temp else T[src_i, src_j, src_k]
    smoke[i, j, k] = neighbor_smoke
    fuel[i, j, k] = neighbor_fuel


@cuda.jit(cache=True)
def _domain_bc_kernel(
    u,
    v,
    w,
    p,
    T,
    smoke,
    fuel,
    x_low_mode,
    x_low_u,
    x_low_v,
    x_low_w,
    x_low_temp,
    x_low_use_temp,
    x_high_mode,
    x_high_u,
    x_high_v,
    x_high_w,
    x_high_temp,
    x_high_use_temp,
    y_low_mode,
    y_low_u,
    y_low_v,
    y_low_w,
    y_low_temp,
    y_low_use_temp,
    y_high_mode,
    y_high_u,
    y_high_v,
    y_high_w,
    y_high_temp,
    y_high_use_temp,
    z_low_mode,
    z_low_u,
    z_low_v,
    z_low_w,
    z_low_temp,
    z_low_use_temp,
    z_high_mode,
    z_high_u,
    z_high_v,
    z_high_w,
    z_high_temp,
    z_high_use_temp,
):
    """
    Apply all configured domain face boundary conditions in one 3D launch.

    One thread owns one boundary cell and sequentially applies every matching
    face condition for corners and edges in the same fixed side order.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if 0 < i < nx - 1 and 0 < j < ny - 1 and 0 < k < nz - 1:
        return

    if i == 0:
        _apply_face_state(
            u,
            v,
            w,
            p,
            T,
            smoke,
            fuel,
            i,
            j,
            k,
            1,
            j,
            k,
            0,
            0,
            x_low_mode,
            x_low_u,
            x_low_v,
            x_low_w,
            x_low_temp,
            x_low_use_temp,
        )
    elif i == nx - 1:
        _apply_face_state(
            u,
            v,
            w,
            p,
            T,
            smoke,
            fuel,
            i,
            j,
            k,
            nx - 2,
            j,
            k,
            0,
            1,
            x_high_mode,
            x_high_u,
            x_high_v,
            x_high_w,
            x_high_temp,
            x_high_use_temp,
        )

    if j == 0:
        _apply_face_state(
            u,
            v,
            w,
            p,
            T,
            smoke,
            fuel,
            i,
            j,
            k,
            i,
            1,
            k,
            1,
            0,
            y_low_mode,
            y_low_u,
            y_low_v,
            y_low_w,
            y_low_temp,
            y_low_use_temp,
        )
    elif j == ny - 1:
        _apply_face_state(
            u,
            v,
            w,
            p,
            T,
            smoke,
            fuel,
            i,
            j,
            k,
            i,
            ny - 2,
            k,
            1,
            1,
            y_high_mode,
            y_high_u,
            y_high_v,
            y_high_w,
            y_high_temp,
            y_high_use_temp,
        )

    if k == 0:
        _apply_face_state(
            u,
            v,
            w,
            p,
            T,
            smoke,
            fuel,
            i,
            j,
            k,
            i,
            j,
            1,
            2,
            0,
            z_low_mode,
            z_low_u,
            z_low_v,
            z_low_w,
            z_low_temp,
            z_low_use_temp,
        )
    elif k == nz - 1:
        _apply_face_state(
            u,
            v,
            w,
            p,
            T,
            smoke,
            fuel,
            i,
            j,
            k,
            i,
            j,
            nz - 2,
            2,
            1,
            z_high_mode,
            z_high_u,
            z_high_v,
            z_high_w,
            z_high_temp,
            z_high_use_temp,
        )


def domain_bc(u, v, w, p, T, smoke, fuel, bc_config):
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
            bc_mode,
            float(bc.get("u", 0.0)),
            float(bc.get("v", 0.0)),
            float(bc.get("w", 0.0)),
            0.0 if temperature is None else float(temperature),
            temperature is not None,
        )

    threadsperblock = (
        THREADS_PER_BLOCK_2D[0],
        THREADS_PER_BLOCK_2D[0],
        THREADS_PER_BLOCK_2D[1],
    )
    blockspergrid = (
        (u.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (u.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (u.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )

    _domain_bc_kernel[blockspergrid, threadsperblock](
        u,
        v,
        w,
        p,
        T,
        smoke,
        fuel,
        *face_args["x_low"],
        *face_args["x_high"],
        *face_args["y_low"],
        *face_args["y_high"],
        *face_args["z_low"],
        *face_args["z_high"],
    )

    return u, v, w, p, T, smoke, fuel
