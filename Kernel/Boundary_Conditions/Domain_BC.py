from numba import cuda

THREADS_PER_BLOCK_3D = (8, 8, 8)
THREADS_PER_BLOCK_2D = (16, 16)


@cuda.jit
def _outflow_bc_kernel(u, v, w, p, T, axis, side_index):
    """
    applies outflow boundary conditions on one domain face on the GPU.

    Each thread updates one cell on the selected boundary face. All fields use a
    homogeneous Neumann condition, meaning their boundary value is copied from
    the adjacent interior cell.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        axis (int): face orientation encoded as 0 for x, 1 for y and 2 for z
        side_index (int): lower face when 0, upper face when 1
    """
    a, b = cuda.grid(2)
    nx, ny, nz = u.shape

    if axis == 0:
        if a >= ny or b >= nz:
            return
        i = 0 if side_index == 0 else nx - 1
        src_i = 1 if side_index == 0 else nx - 2
        u[i, a, b] = u[src_i, a, b]
        v[i, a, b] = v[src_i, a, b]
        w[i, a, b] = w[src_i, a, b]
        p[i, a, b] = p[src_i, a, b]
        T[i, a, b] = T[src_i, a, b]
    elif axis == 1:
        if a >= nx or b >= nz:
            return
        j = 0 if side_index == 0 else ny - 1
        src_j = 1 if side_index == 0 else ny - 2
        u[a, j, b] = u[a, src_j, b]
        v[a, j, b] = v[a, src_j, b]
        w[a, j, b] = w[a, src_j, b]
        p[a, j, b] = p[a, src_j, b]
        T[a, j, b] = T[a, src_j, b]
    else:
        if a >= nx or b >= ny:
            return
        k = 0 if side_index == 0 else nz - 1
        src_k = 1 if side_index == 0 else nz - 2
        u[a, b, k] = u[a, b, src_k]
        v[a, b, k] = v[a, b, src_k]
        w[a, b, k] = w[a, b, src_k]
        p[a, b, k] = p[a, b, src_k]
        T[a, b, k] = T[a, b, src_k]


@cuda.jit
def _inflow_bc_kernel(u, v, w, p, T, axis, side_index, u_inflow, v_inflow, w_inflow, t_inflow, use_t_inflow):
    """
    applies inflow boundary conditions on one domain face on the GPU.

    Each thread updates one cell on the selected boundary face. Velocity uses
    fixed inflow values, pressure uses a homogeneous Neumann condition and
    temperature uses either a fixed inflow temperature or a homogeneous Neumann
    condition.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        axis (int): face orientation encoded as 0 for x, 1 for y and 2 for z
        side_index (int): lower face when 0, upper face when 1
        u_inflow (float): prescribed inflow value for u
        v_inflow (float): prescribed inflow value for v
        w_inflow (float): prescribed inflow value for w
        t_inflow (float): prescribed inflow temperature
        use_t_inflow (bool): whether temperature should use a fixed Dirichlet value
    """
    a, b = cuda.grid(2)
    nx, ny, nz = u.shape

    if axis == 0:
        if a >= ny or b >= nz:
            return
        i = 0 if side_index == 0 else nx - 1
        src_i = 1 if side_index == 0 else nx - 2
        u[i, a, b] = u_inflow
        v[i, a, b] = v_inflow
        w[i, a, b] = w_inflow
        p[i, a, b] = p[src_i, a, b]
        T[i, a, b] = t_inflow if use_t_inflow else T[src_i, a, b]
    elif axis == 1:
        if a >= nx or b >= nz:
            return
        j = 0 if side_index == 0 else ny - 1
        src_j = 1 if side_index == 0 else ny - 2
        u[a, j, b] = u_inflow
        v[a, j, b] = v_inflow
        w[a, j, b] = w_inflow
        p[a, j, b] = p[a, src_j, b]
        T[a, j, b] = t_inflow if use_t_inflow else T[a, src_j, b]
    else:
        if a >= nx or b >= ny:
            return
        k = 0 if side_index == 0 else nz - 1
        src_k = 1 if side_index == 0 else nz - 2
        u[a, b, k] = u_inflow
        v[a, b, k] = v_inflow
        w[a, b, k] = w_inflow
        p[a, b, k] = p[a, b, src_k]
        T[a, b, k] = t_inflow if use_t_inflow else T[a, b, src_k]


@cuda.jit
def _no_slip_wall_bc_kernel(u, v, w, p, T, axis, side_index, t_wall, use_t_wall):
    """
    applies no-slip wall boundary conditions on one domain face on the GPU.

    Each thread updates one cell on the selected boundary face. All three
    velocity components are forced to zero, pressure uses a homogeneous Neumann
    condition and temperature uses either a fixed wall value or a homogeneous
    Neumann condition.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        axis (int): face orientation encoded as 0 for x, 1 for y and 2 for z
        side_index (int): lower face when 0, upper face when 1
        t_wall (float): prescribed wall temperature
        use_t_wall (bool): whether temperature should use a fixed Dirichlet value
    """
    a, b = cuda.grid(2)
    nx, ny, nz = u.shape

    if axis == 0:
        if a >= ny or b >= nz:
            return
        i = 0 if side_index == 0 else nx - 1
        src_i = 1 if side_index == 0 else nx - 2
        u[i, a, b] = 0.0
        v[i, a, b] = 0.0
        w[i, a, b] = 0.0
        p[i, a, b] = p[src_i, a, b]
        T[i, a, b] = t_wall if use_t_wall else T[src_i, a, b]
    elif axis == 1:
        if a >= nx or b >= nz:
            return
        j = 0 if side_index == 0 else ny - 1
        src_j = 1 if side_index == 0 else ny - 2
        u[a, j, b] = 0.0
        v[a, j, b] = 0.0
        w[a, j, b] = 0.0
        p[a, j, b] = p[a, src_j, b]
        T[a, j, b] = t_wall if use_t_wall else T[a, src_j, b]
    else:
        if a >= nx or b >= ny:
            return
        k = 0 if side_index == 0 else nz - 1
        src_k = 1 if side_index == 0 else nz - 2
        u[a, b, k] = 0.0
        v[a, b, k] = 0.0
        w[a, b, k] = 0.0
        p[a, b, k] = p[a, b, src_k]
        T[a, b, k] = t_wall if use_t_wall else T[a, b, src_k]


@cuda.jit
def _slip_wall_bc_kernel(u, v, w, p, T, axis, side_index, t_wall, use_t_wall):
    """
    applies slip-wall boundary conditions on one domain face on the GPU.

    Each thread updates one cell on the selected boundary face. The velocity
    component normal to the wall is forced to zero, tangential components use a
    homogeneous Neumann condition, pressure uses a homogeneous Neumann condition
    and temperature uses either a fixed wall value or a homogeneous Neumann
    condition.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        axis (int): face orientation encoded as 0 for x, 1 for y and 2 for z
        side_index (int): lower face when 0, upper face when 1
        t_wall (float): prescribed wall temperature
        use_t_wall (bool): whether temperature should use a fixed Dirichlet value
    """
    a, b = cuda.grid(2)
    nx, ny, nz = u.shape

    if axis == 0:
        if a >= ny or b >= nz:
            return
        i = 0 if side_index == 0 else nx - 1
        src_i = 1 if side_index == 0 else nx - 2
        u[i, a, b] = 0.0
        v[i, a, b] = v[src_i, a, b]
        w[i, a, b] = w[src_i, a, b]
        p[i, a, b] = p[src_i, a, b]
        T[i, a, b] = t_wall if use_t_wall else T[src_i, a, b]
    elif axis == 1:
        if a >= nx or b >= nz:
            return
        j = 0 if side_index == 0 else ny - 1
        src_j = 1 if side_index == 0 else ny - 2
        u[a, j, b] = u[a, src_j, b]
        v[a, j, b] = 0.0
        w[a, j, b] = w[a, src_j, b]
        p[a, j, b] = p[a, src_j, b]
        T[a, j, b] = t_wall if use_t_wall else T[a, src_j, b]
    else:
        if a >= nx or b >= ny:
            return
        k = 0 if side_index == 0 else nz - 1
        src_k = 1 if side_index == 0 else nz - 2
        u[a, b, k] = u[a, b, src_k]
        v[a, b, k] = v[a, b, src_k]
        w[a, b, k] = 0.0
        p[a, b, k] = p[a, b, src_k]
        T[a, b, k] = t_wall if use_t_wall else T[a, b, src_k]


def _side_to_axis_and_index(side):
    """
    translates a textual side identifier into axis and lower/upper face indices.

    Args:
        side (str): one of x_low, x_high, y_low, y_high, z_low or z_high
    Returns:
        tuple[int, int]: axis index and side index used by the CUDA kernels
    """
    if side == "x_low":
        return 0, 0
    if side == "x_high":
        return 0, 1
    if side == "y_low":
        return 1, 0
    if side == "y_high":
        return 1, 1
    if side == "z_low":
        return 2, 0
    if side == "z_high":
        return 2, 1
    raise ValueError(f"Unknown boundary side '{side}'")


def _boundary_blockspergrid(field_shape, axis, threadsperblock):
    """
    computes the 2D grid dimensions needed to launch one boundary-face kernel.

    Args:
        field_shape (tuple[int, int, int]): full shape of the affected 3D field
        axis (int): face orientation encoded as 0 for x, 1 for y and 2 for z
        threadsperblock (tuple[int, int]): CUDA thread block shape for face kernels
    Returns:
        tuple[int, int]: CUDA grid shape for the selected face
    """
    if axis == 0:
        return (
            (field_shape[1] + threadsperblock[0] - 1) // threadsperblock[0],
            (field_shape[2] + threadsperblock[1] - 1) // threadsperblock[1],
        )
    if axis == 1:
        return (
            (field_shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
            (field_shape[2] + threadsperblock[1] - 1) // threadsperblock[1],
        )
    return (
        (field_shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (field_shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
    )


def inflow_bc(u, v, w, p, T, side, u_inflow, v_inflow, w_inflow, t_inflow=None, threadsperblock=None):
    """
    applies inflow boundary conditions to one side of the domain on the GPU.

    The velocity components are fixed to the prescribed inflow values, pressure
    receives a homogeneous Neumann condition and temperature uses either a fixed
    inflow value or a Neumann condition when no temperature is specified.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        side (str): boundary side identifier
        u_inflow (float): inflow velocity in x-direction
        v_inflow (float): inflow velocity in y-direction
        w_inflow (float): inflow velocity in z-direction
        t_inflow (float, optional): prescribed inflow temperature
        threadsperblock (tuple[int, int], optional): CUDA block shape for face kernels
    Returns:
        tuple: updated velocity, pressure and temperature fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_2D
    axis, side_index = _side_to_axis_and_index(side)
    blockspergrid = _boundary_blockspergrid(u.shape, axis, threadsperblock)
    _inflow_bc_kernel[blockspergrid, threadsperblock](
        u, v, w, p, T,
        axis, side_index,
        u_inflow, v_inflow, w_inflow,
        0.0 if t_inflow is None else t_inflow,
        t_inflow is not None,
    )

    return u, v, w, p, T


def outflow_bc(u, v, w, p, T, side, threadsperblock=None):
    """
    applies outflow boundary conditions to one side of the domain on the GPU.

    All fields use homogeneous Neumann conditions so their boundary values are
    copied from the adjacent interior cells.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        side (str): boundary side identifier
        threadsperblock (tuple[int, int], optional): CUDA block shape for face kernels
    Returns:
        tuple: updated velocity, pressure and temperature fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_2D
    axis, side_index = _side_to_axis_and_index(side)
    blockspergrid = _boundary_blockspergrid(u.shape, axis, threadsperblock)
    _outflow_bc_kernel[blockspergrid, threadsperblock](u, v, w, p, T, axis, side_index)
    return u, v, w, p, T


def slip_wall_bc(u, v, w, p, T, side, t_wall=None, threadsperblock=None):
    """
    applies slip-wall boundary conditions to one side of the domain on the GPU.

    The velocity component normal to the wall is forced to zero, the tangential
    components receive homogeneous Neumann conditions and pressure always uses a
    homogeneous Neumann condition. Temperature is either copied from the
    interior or fixed to a prescribed wall value.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        side (str): boundary side identifier
        t_wall (float, optional): prescribed wall temperature
        threadsperblock (tuple[int, int], optional): CUDA block shape for face kernels
    Returns:
        tuple: updated velocity, pressure and temperature fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_2D
    axis, side_index = _side_to_axis_and_index(side)
    blockspergrid = _boundary_blockspergrid(u.shape, axis, threadsperblock)
    _slip_wall_bc_kernel[blockspergrid, threadsperblock](
        u, v, w, p, T,
        axis, side_index,
        0.0 if t_wall is None else t_wall,
        t_wall is not None,
    )

    return u, v, w, p, T


def no_slip_wall_bc(u, v, w, p, T, side, t_wall=None, threadsperblock=None):
    """
    applies no-slip wall boundary conditions to one side of the domain on the GPU.

    All three velocity components are forced to zero at the wall, pressure uses
    a homogeneous Neumann condition and temperature is either copied from the
    interior or fixed to a prescribed wall value.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        side (str): boundary side identifier
        t_wall (float, optional): prescribed wall temperature
        threadsperblock (tuple[int, int], optional): CUDA block shape for face kernels
    Returns:
        tuple: updated velocity, pressure and temperature fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_2D
    axis, side_index = _side_to_axis_and_index(side)
    blockspergrid = _boundary_blockspergrid(u.shape, axis, threadsperblock)
    _no_slip_wall_bc_kernel[blockspergrid, threadsperblock](
        u, v, w, p, T,
        axis, side_index,
        0.0 if t_wall is None else t_wall,
        t_wall is not None,
    )

    return u, v, w, p, T


def apply_all_BC(u, v, w, p, T, bc_config):
    """
    applies all configured domain boundary conditions to the GPU field state.

    The function iterates over the configured faces and dispatches the matching
    CUDA boundary kernels for INFLOW, OUTFLOW, SLIP_WALL and WALL conditions.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        bc_config (dict): boundary-condition configuration indexed by domain side
    Returns:
        tuple: updated velocity, pressure and temperature fields
    """
    for side, bc in bc_config.items():
        bc_type = bc["type"]

        if bc_type == "OUTFLOW":
            u, v, w, p, T = outflow_bc(u, v, w, p, T, side)
        elif bc_type == "INFLOW":
            u, v, w, p, T = inflow_bc(
                u, v, w, p, T,
                side,
                bc.get("u", 0.0),
                bc.get("v", 0.0),
                bc.get("w", 0.0),
                bc.get("T", bc.get("temperature")),
            )
        elif bc_type == "WALL":
            u, v, w, p, T = no_slip_wall_bc(
                u, v, w, p, T, side, bc.get("T", bc.get("temperature"))
            )
        elif bc_type == "SLIP_WALL":
            u, v, w, p, T = slip_wall_bc(
                u, v, w, p, T, side, bc.get("T", bc.get("temperature"))
            )
        else:
            raise ValueError(f"Unknown boundary condition '{bc_type}' for side '{side}'")

    return u, v, w, p, T
