from numba import cuda

from Kernel.Kernel_Config import THREADS_PER_BLOCK_2D, boundary_face_blocks_per_grid


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


@cuda.jit(device=True)
def _apply_face_state(
    u, v, w, p, T,
    i, j, k,
    src_i, src_j, src_k,
    axis, bc_mode,
    u_value, v_value, w_value,
    temp_value, use_temp,
):
    neighbor_u = u[src_i, src_j, src_k]
    neighbor_v = v[src_i, src_j, src_k]
    neighbor_w = w[src_i, src_j, src_k]

    if bc_mode == BC_OUTFLOW:
        u[i, j, k] = neighbor_u
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
        u[i, j, k] = 0.0 if axis == 0 else neighbor_u
        v[i, j, k] = 0.0 if axis == 1 else neighbor_v
        w[i, j, k] = 0.0 if axis == 2 else neighbor_w

    p[i, j, k] = p[src_i, src_j, src_k]
    T[i, j, k] = temp_value if use_temp else T[src_i, src_j, src_k]


@cuda.jit
def _domain_bc_kernel(
    u, v, w, p, T,
    axis, side_index, bc_mode,
    u_value, v_value, w_value,
    temp_value, use_temp,
):
    """
    Apply one domain boundary condition on one selected face of the simulation.

    The kernel handles all supported domain BC types. Pressure always uses a
    homogeneous Neumann condition and temperature uses either a fixed value or
    a homogeneous Neumann condition depending on ``use_temp``.
    """
    a, b = cuda.grid(2)
    nx, ny, nz = u.shape

    if axis == 0:
        if a >= ny or b >= nz:
            return
        i = 0 if side_index == 0 else nx - 1
        src_i = 1 if side_index == 0 else nx - 2
        _apply_face_state(
            u, v, w, p, T,
            i, a, b,
            src_i, a, b,
            axis,
            bc_mode,
            u_value, v_value, w_value,
            temp_value, use_temp,
        )
        return

    if axis == 1:
        if a >= nx or b >= nz:
            return
        j = 0 if side_index == 0 else ny - 1
        src_j = 1 if side_index == 0 else ny - 2
        _apply_face_state(
            u, v, w, p, T,
            a, j, b,
            a, src_j, b,
            axis,
            bc_mode,
            u_value, v_value, w_value,
            temp_value, use_temp,
        )
        return

    if axis == 2:
        if a >= nx or b >= ny:
            return
        k = 0 if side_index == 0 else nz - 1
        src_k = 1 if side_index == 0 else nz - 2
        _apply_face_state(
            u, v, w, p, T,
            a, b, k,
            a, b, src_k,
            axis,
            bc_mode,
            u_value, v_value, w_value,
            temp_value, use_temp,
        )


def _launch_face_bc(
    u, v, w, p, T,
    side, bc_mode,
    u_value=0.0, v_value=0.0, w_value=0.0,
    temp_value=None,
    threadsperblock=None,
):
    """Launch the shared domain-face BC kernel with readable host-side defaults."""
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_2D

    axis, side_index = SIDE_TO_AXIS_AND_INDEX[side]
    blockspergrid = boundary_face_blocks_per_grid(u.shape, axis, threadsperblock)
    _domain_bc_kernel[blockspergrid, threadsperblock](
        u, v, w, p, T,
        axis, side_index, bc_mode,
        u_value, v_value, w_value,
        0.0 if temp_value is None else temp_value,
        temp_value is not None,
    )
    return u, v, w, p, T


def inflow_bc(u, v, w, p, T, side, u_inflow, v_inflow, w_inflow, t_inflow=None, threadsperblock=None):
    """
    Apply inflow boundary conditions to one side of the domain on the GPU.

    Velocity is fixed to the prescribed inflow values, pressure receives a
    homogeneous Neumann condition and temperature uses either a fixed inflow
    value or a Neumann condition when no temperature is specified.
    """
    return _launch_face_bc(
        u, v, w, p, T,
        side,
        BC_INFLOW,
        u_value=u_inflow,
        v_value=v_inflow,
        w_value=w_inflow,
        temp_value=t_inflow,
        threadsperblock=threadsperblock,
    )


def outflow_bc(u, v, w, p, T, side, threadsperblock=None):
    """
    Apply outflow boundary conditions to one side of the domain on the GPU.

    All fields use homogeneous Neumann conditions so their boundary values are
    copied from the adjacent interior cells.
    """
    return _launch_face_bc(
        u, v, w, p, T,
        side,
        BC_OUTFLOW,
        threadsperblock=threadsperblock,
    )


def slip_wall_bc(u, v, w, p, T, side, t_wall=None, threadsperblock=None):
    """
    Apply slip-wall boundary conditions to one side of the domain on the GPU.

    The velocity component normal to the wall is forced to zero, tangential
    components receive homogeneous Neumann conditions and pressure always uses
    a homogeneous Neumann condition.
    """
    return _launch_face_bc(
        u, v, w, p, T,
        side,
        BC_SLIP_WALL,
        temp_value=t_wall,
        threadsperblock=threadsperblock,
    )


def no_slip_wall_bc(u, v, w, p, T, side, t_wall=None, threadsperblock=None):
    """
    Apply no-slip wall boundary conditions to one side of the domain on the GPU.

    All three velocity components are forced to zero at the wall, pressure uses
    a homogeneous Neumann condition and temperature is either copied from the
    interior or fixed to a prescribed wall value.
    """
    return _launch_face_bc(
        u, v, w, p, T,
        side,
        BC_NO_SLIP_WALL,
        temp_value=t_wall,
        threadsperblock=threadsperblock,
    )


def apply_all_BC(u, v, w, p, T, bc_config):
    """
    Apply all configured domain boundary conditions to the GPU field state.

    The function iterates over the configured faces and dispatches the matching
    boundary mode for INFLOW, OUTFLOW, SLIP_WALL and WALL.
    """
    for side, bc in bc_config.items():
        bc_type = bc["type"]
        temperature = bc.get("T", bc.get("temperature"))

        if bc_type == "OUTFLOW":
            u, v, w, p, T = outflow_bc(u, v, w, p, T, side)
        elif bc_type == "INFLOW":
            u, v, w, p, T = inflow_bc(
                u, v, w, p, T,
                side,
                bc.get("u", 0.0),
                bc.get("v", 0.0),
                bc.get("w", 0.0),
                temperature,
            )
        elif bc_type == "WALL":
            u, v, w, p, T = no_slip_wall_bc(u, v, w, p, T, side, temperature)
        elif bc_type == "SLIP_WALL":
            u, v, w, p, T = slip_wall_bc(u, v, w, p, T, side, temperature)
        else:
            raise ValueError(f"Unknown boundary condition '{bc_type}' for side '{side}'")

    return u, v, w, p, T
