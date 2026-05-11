from numba import njit, prange


BC_OUTFLOW = 0
BC_INFLOW = 1
BC_NO_SLIP_WALL = 2
BC_SLIP_WALL = 3

SIDE_TO_AXIS_AND_INDEX = {
    "x_low": (0, 0),
    "x_high": (0, 1),
    "y_low": (1, 0),
    "y_high": (1, 1),
    "z_low": (2, 0),
    "z_high": (2, 1),
}

BC_TYPE_TO_MODE = {
    "OUTFLOW": BC_OUTFLOW,
    "INFLOW": BC_INFLOW,
    "WALL": BC_NO_SLIP_WALL,
    "SLIP_WALL": BC_SLIP_WALL,
}


@njit(cache=True, parallel=True)
def _apply_face_bc_numba(
    u, v, w, p, T, smoke, fuel,
    axis, side_index, bc_mode,
    u_value, v_value, w_value,
    temp_value, use_temp,
):
    """
    Apply one boundary condition to one domain face on the CPU.

    The face sweep is flattened to one parallel loop so Numba can distribute
    the work across threads while keeping the inner logic branch-light.
    """
    nx, ny, nz = u.shape

    if axis == 0:
        i = 0 if side_index == 0 else nx - 1
        src_i = 1 if side_index == 0 else nx - 2
        face_size = ny * nz

        for idx in prange(face_size):
            j = idx // nz
            k = idx - j * nz

            neighbor_u = u[src_i, j, k]
            neighbor_v = v[src_i, j, k]
            neighbor_w = w[src_i, j, k]
            neighbor_smoke = smoke[src_i, j, k]
            neighbor_fuel = fuel[src_i, j, k]

            if bc_mode == BC_OUTFLOW:
                u[i, j, k] = min(neighbor_u, 0.0) if side_index == 0 else max(neighbor_u, 0.0)
                v[i, j, k] = neighbor_v
                w[i, j, k] = neighbor_w
            elif bc_mode == BC_INFLOW:
                u[i, j, k] = u_value
                v[i, j, k] = v_value
                w[i, j, k] = w_value
            elif bc_mode == BC_NO_SLIP_WALL:
                u[i, j, k] = 0.0
                v[i, j, k] = 0.0
                w[i, j, k] = 0.0
            else:
                u[i, j, k] = 0.0
                v[i, j, k] = neighbor_v
                w[i, j, k] = neighbor_w

            p[i, j, k] = p[src_i, j, k]
            T[i, j, k] = temp_value if use_temp else T[src_i, j, k]
            smoke[i, j, k] = neighbor_smoke
            fuel[i, j, k] = neighbor_fuel
        return

    if axis == 1:
        j = 0 if side_index == 0 else ny - 1
        src_j = 1 if side_index == 0 else ny - 2
        face_size = nx * nz

        for idx in prange(face_size):
            i = idx // nz
            k = idx - i * nz

            neighbor_u = u[i, src_j, k]
            neighbor_v = v[i, src_j, k]
            neighbor_w = w[i, src_j, k]
            neighbor_smoke = smoke[i, src_j, k]
            neighbor_fuel = fuel[i, src_j, k]

            if bc_mode == BC_OUTFLOW:
                u[i, j, k] = neighbor_u
                v[i, j, k] = min(neighbor_v, 0.0) if side_index == 0 else max(neighbor_v, 0.0)
                w[i, j, k] = neighbor_w
            elif bc_mode == BC_INFLOW:
                u[i, j, k] = u_value
                v[i, j, k] = v_value
                w[i, j, k] = w_value
            elif bc_mode == BC_NO_SLIP_WALL:
                u[i, j, k] = 0.0
                v[i, j, k] = 0.0
                w[i, j, k] = 0.0
            else:
                u[i, j, k] = neighbor_u
                v[i, j, k] = 0.0
                w[i, j, k] = neighbor_w

            p[i, j, k] = p[i, src_j, k]
            T[i, j, k] = temp_value if use_temp else T[i, src_j, k]
            smoke[i, j, k] = neighbor_smoke
            fuel[i, j, k] = neighbor_fuel
        return

    k = 0 if side_index == 0 else nz - 1
    src_k = 1 if side_index == 0 else nz - 2
    face_size = nx * ny

    for idx in prange(face_size):
        i = idx // ny
        j = idx - i * ny

        neighbor_u = u[i, j, src_k]
        neighbor_v = v[i, j, src_k]
        neighbor_w = w[i, j, src_k]
        neighbor_smoke = smoke[i, j, src_k]
        neighbor_fuel = fuel[i, j, src_k]

        if bc_mode == BC_OUTFLOW:
            u[i, j, k] = neighbor_u
            v[i, j, k] = neighbor_v
            w[i, j, k] = min(neighbor_w, 0.0) if side_index == 0 else max(neighbor_w, 0.0)
        elif bc_mode == BC_INFLOW:
            u[i, j, k] = u_value
            v[i, j, k] = v_value
            w[i, j, k] = w_value
        elif bc_mode == BC_NO_SLIP_WALL:
            u[i, j, k] = 0.0
            v[i, j, k] = 0.0
            w[i, j, k] = 0.0
        else:
            u[i, j, k] = neighbor_u
            v[i, j, k] = neighbor_v
            w[i, j, k] = 0.0

        p[i, j, k] = p[i, j, src_k]
        T[i, j, k] = temp_value if use_temp else T[i, j, src_k]
        smoke[i, j, k] = neighbor_smoke
        fuel[i, j, k] = neighbor_fuel


def _apply_face_bc(
    u, v, w, p, T, smoke, fuel,
    side, bc_mode,
    u_value=0.0, v_value=0.0, w_value=0.0,
    temp_value=None,
):
    """Dispatch one CPU boundary update to the compiled Numba face kernel."""
    axis, side_index = SIDE_TO_AXIS_AND_INDEX[side]
    _apply_face_bc_numba(
        u, v, w, p, T, smoke, fuel,
        axis, side_index, bc_mode,
        u_value, v_value, w_value,
        0.0 if temp_value is None else temp_value,
        temp_value is not None,
    )
    return u, v, w, p, T, smoke, fuel


def inflow_bc(u, v, w, p, T, smoke, fuel, side, u_inflow, v_inflow, w_inflow, t_inflow=None):
    """
    Apply inflow boundary conditions to one side of the domain on the CPU.

    Velocity is fixed to the prescribed inflow values, pressure receives a
    homogeneous Neumann condition and temperature uses either a fixed inflow
    value or a Neumann condition when no temperature is specified.
    """
    return _apply_face_bc(
        u, v, w, p, T, smoke, fuel,
        side,
        BC_INFLOW,
        u_value=u_inflow,
        v_value=v_inflow,
        w_value=w_inflow,
        temp_value=t_inflow,
    )


def outflow_bc(u, v, w, p, T, smoke, fuel, side):
    """
    Apply outflow boundary conditions to one side of the domain on the CPU.

    All fields use homogeneous Neumann conditions so their boundary values are
    copied from the adjacent interior cells.
    """
    return _apply_face_bc(
        u, v, w, p, T, smoke, fuel,
        side,
        BC_OUTFLOW,
    )


def slip_wall_bc(u, v, w, p, T, smoke, fuel, side, t_wall=None):
    """
    Apply slip-wall boundary conditions to one side of the domain on the CPU.

    The velocity component normal to the wall is forced to zero, tangential
    components receive homogeneous Neumann conditions and pressure always uses
    a homogeneous Neumann condition.
    """
    return _apply_face_bc(
        u, v, w, p, T, smoke, fuel,
        side,
        BC_SLIP_WALL,
        temp_value=t_wall,
    )


def no_slip_wall_bc(u, v, w, p, T, smoke, fuel, side, t_wall=None):
    """
    Apply no-slip wall boundary conditions to one side of the domain on the CPU.

    All three velocity components are forced to zero at the wall, pressure uses
    a homogeneous Neumann condition and temperature is either copied from the
    interior or fixed to a prescribed wall value.
    """
    return _apply_face_bc(
        u, v, w, p, T, smoke, fuel,
        side,
        BC_NO_SLIP_WALL,
        temp_value=t_wall,
    )


def apply_all_BC(u, v, w, p, T, smoke, fuel, bc_config):
    """
    Apply all configured domain boundary conditions to the CPU field state.

    The function iterates over the configured faces and dispatches the matching
    boundary mode for INFLOW, OUTFLOW, SLIP_WALL and WALL.
    """
    for side, bc in bc_config.items():
        bc_type = bc["type"]
        bc_mode = BC_TYPE_TO_MODE.get(bc_type)
        if bc_mode is None:
            raise ValueError(f"Unknown boundary condition '{bc_type}' for side '{side}'")

        temperature = bc.get("T", bc.get("temperature"))
        if bc_mode == BC_INFLOW:
            u, v, w, p, T, smoke, fuel = inflow_bc(
                u, v, w, p, T, smoke, fuel,
                side,
                bc.get("u", 0.0),
                bc.get("v", 0.0),
                bc.get("w", 0.0),
                temperature,
            )
        elif bc_mode == BC_OUTFLOW:
            u, v, w, p, T, smoke, fuel = outflow_bc(u, v, w, p, T, smoke, fuel, side)
        elif bc_mode == BC_NO_SLIP_WALL:
            u, v, w, p, T, smoke, fuel = no_slip_wall_bc(
                u, v, w, p, T, smoke, fuel, side, temperature
            )
        else:
            u, v, w, p, T, smoke, fuel = slip_wall_bc(
                u, v, w, p, T, smoke, fuel, side, temperature
            )

    return u, v, w, p, T, smoke, fuel
