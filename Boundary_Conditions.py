from numba import cuda

THREADS_PER_BLOCK_3D = (8, 8, 8)
THREADS_PER_BLOCK_2D = (16, 16)


@cuda.jit
def apply_neumann_boundary_condition(field, axis, side_index):
    """
    applies a zero-gradient boundary condition to one side of a 3D field on the GPU.

    Each thread updates one boundary cell on the selected domain face by copying
    the value from the adjacent interior cell. This implements a homogeneous
    Neumann condition for scalar or velocity fields.

    Args:
        field (device array): field whose selected boundary face will be updated
        axis (int): face orientation encoded as 0 for x, 1 for y and 2 for z
        side_index (int): lower face when 0, upper face when 1
    """
    a, b = cuda.grid(2)
    nx, ny, nz = field.shape

    if axis == 0:
        if a >= ny or b >= nz:
            return
        i = 0 if side_index == 0 else nx - 1
        src_i = 1 if side_index == 0 else nx - 2
        field[i, a, b] = field[src_i, a, b]
    elif axis == 1:
        if a >= nx or b >= nz:
            return
        j = 0 if side_index == 0 else ny - 1
        src_j = 1 if side_index == 0 else ny - 2
        field[a, j, b] = field[a, src_j, b]
    else:
        if a >= nx or b >= ny:
            return
        k = 0 if side_index == 0 else nz - 1
        src_k = 1 if side_index == 0 else nz - 2
        field[a, b, k] = field[a, b, src_k]


@cuda.jit
def apply_dirichlet_boundary_condition(field, axis, side_index, value):
    """
    applies a fixed-value boundary condition to one side of a 3D field on the GPU.

    Each thread writes one boundary cell on the selected face with the prescribed
    constant value. This is used for inflow velocities, no-slip walls and fixed
    scalar wall temperatures.

    Args:
        field (device array): field whose selected boundary face will be updated
        axis (int): face orientation encoded as 0 for x, 1 for y and 2 for z
        side_index (int): lower face when 0, upper face when 1
        value (float): prescribed boundary value
    """
    a, b = cuda.grid(2)
    nx, ny, nz = field.shape

    if axis == 0:
        if a >= ny or b >= nz:
            return
        i = 0 if side_index == 0 else nx - 1
        field[i, a, b] = value
    elif axis == 1:
        if a >= nx or b >= nz:
            return
        j = 0 if side_index == 0 else ny - 1
        field[a, j, b] = value
    else:
        if a >= nx or b >= ny:
            return
        k = 0 if side_index == 0 else nz - 1
        field[a, b, k] = value


@cuda.jit
def obstacle_boundary_conditions_velocity(u, v, w, mask):
    """
    applies no-slip velocity conditions inside a 3D obstacle mask on the GPU.

    Each thread checks one obstacle cell and sets all three velocity components
    to zero when the mask marks that cell as solid.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        mask (device array): boolean obstacle mask
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if mask[i, j, k]:
        u[i, j, k] = 0.0
        v[i, j, k] = 0.0
        w[i, j, k] = 0.0


@cuda.jit
def obstacle_boundary_conditions_scalar(phi, mask, value):
    """
    applies a fixed scalar value inside a 3D obstacle mask on the GPU.

    Each thread checks one obstacle cell and writes the prescribed scalar value
    when the cell belongs to the obstacle region.

    Args:
        phi (device array): scalar field that will be clamped inside the obstacle
        mask (device array): boolean obstacle mask
        value (float): scalar value prescribed inside the obstacle
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if mask[i, j, k]:
        phi[i, j, k] = value


@cuda.jit
def obstacle_boundary_conditions_pressure(p, mask):
    """
    applies zero pressure inside a 3D obstacle mask on the GPU.

    Each thread checks one obstacle cell and sets the pressure to zero when the
    mask marks that cell as part of the obstacle region.

    Args:
        p (device array): pressure field
        mask (device array): boolean obstacle mask
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if mask[i, j, k]:
        p[i, j, k] = 0.0


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


def launch_neumann_boundary_condition(field, side, threadsperblock=None):
    """
    launches the zero-gradient boundary-condition kernel for one domain face.

    Args:
        field (device array): field whose selected boundary face will be updated
        side (str): boundary side identifier
        threadsperblock (tuple[int, int], optional): CUDA block shape for face kernels
    Returns:
        device array: the updated field
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_2D
    axis, side_index = _side_to_axis_and_index(side)
    blockspergrid = _boundary_blockspergrid(field.shape, axis, threadsperblock)
    apply_neumann_boundary_condition[blockspergrid, threadsperblock](field, axis, side_index)
    return field


def launch_dirichlet_boundary_condition(field, side, value, threadsperblock=None):
    """
    launches the fixed-value boundary-condition kernel for one domain face.

    Args:
        field (device array): field whose selected boundary face will be updated
        side (str): boundary side identifier
        value (float): prescribed boundary value
        threadsperblock (tuple[int, int], optional): CUDA block shape for face kernels
    Returns:
        device array: the updated field
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_2D
    axis, side_index = _side_to_axis_and_index(side)
    blockspergrid = _boundary_blockspergrid(field.shape, axis, threadsperblock)
    apply_dirichlet_boundary_condition[blockspergrid, threadsperblock](field, axis, side_index, value)
    return field


def launch_inflow_bc(u, v, w, p, T, side, u_inflow, v_inflow, w_inflow, t_inflow=None, threadsperblock=None):
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
    launch_dirichlet_boundary_condition(u, side, u_inflow, threadsperblock)
    launch_dirichlet_boundary_condition(v, side, v_inflow, threadsperblock)
    launch_dirichlet_boundary_condition(w, side, w_inflow, threadsperblock)
    launch_neumann_boundary_condition(p, side, threadsperblock)

    if t_inflow is None:
        launch_neumann_boundary_condition(T, side, threadsperblock)
    else:
        launch_dirichlet_boundary_condition(T, side, t_inflow, threadsperblock)

    return u, v, w, p, T


def launch_outflow_bc(u, v, w, p, T, side, threadsperblock=None):
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
    launch_neumann_boundary_condition(u, side, threadsperblock)
    launch_neumann_boundary_condition(v, side, threadsperblock)
    launch_neumann_boundary_condition(w, side, threadsperblock)
    launch_neumann_boundary_condition(p, side, threadsperblock)
    launch_neumann_boundary_condition(T, side, threadsperblock)
    return u, v, w, p, T


def launch_slip_wall_bc(u, v, w, p, T, side, t_wall=None, threadsperblock=None):
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
    if side == "x_low" or side == "x_high":
        launch_dirichlet_boundary_condition(u, side, 0.0, threadsperblock)
        launch_neumann_boundary_condition(v, side, threadsperblock)
        launch_neumann_boundary_condition(w, side, threadsperblock)
    elif side == "y_low" or side == "y_high":
        launch_neumann_boundary_condition(u, side, threadsperblock)
        launch_dirichlet_boundary_condition(v, side, 0.0, threadsperblock)
        launch_neumann_boundary_condition(w, side, threadsperblock)
    else:
        launch_neumann_boundary_condition(u, side, threadsperblock)
        launch_neumann_boundary_condition(v, side, threadsperblock)
        launch_dirichlet_boundary_condition(w, side, 0.0, threadsperblock)

    launch_neumann_boundary_condition(p, side, threadsperblock)
    if t_wall is None:
        launch_neumann_boundary_condition(T, side, threadsperblock)
    else:
        launch_dirichlet_boundary_condition(T, side, t_wall, threadsperblock)

    return u, v, w, p, T


def launch_no_slip_wall_bc(u, v, w, p, T, side, t_wall=None, threadsperblock=None):
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
    launch_dirichlet_boundary_condition(u, side, 0.0, threadsperblock)
    launch_dirichlet_boundary_condition(v, side, 0.0, threadsperblock)
    launch_dirichlet_boundary_condition(w, side, 0.0, threadsperblock)
    launch_neumann_boundary_condition(p, side, threadsperblock)

    if t_wall is None:
        launch_neumann_boundary_condition(T, side, threadsperblock)
    else:
        launch_dirichlet_boundary_condition(T, side, t_wall, threadsperblock)

    return u, v, w, p, T


def launch_obstacle_boundary_conditions_velocity(u, v, w, mask, threadsperblock=None):
    """
    launches the obstacle no-slip velocity kernel on the GPU.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        mask (device array): boolean obstacle mask
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        tuple: updated velocity fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D
    blockspergrid = (
        (mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    obstacle_boundary_conditions_velocity[blockspergrid, threadsperblock](u, v, w, mask)
    return u, v, w


def launch_obstacle_boundary_conditions_scalar(phi, mask, value, threadsperblock=None):
    """
    launches the obstacle scalar boundary kernel on the GPU.

    Args:
        phi (device array): scalar field that will be clamped inside the obstacle
        mask (device array): boolean obstacle mask
        value (float): scalar value prescribed inside the obstacle
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        device array: the updated scalar field
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D
    blockspergrid = (
        (mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    obstacle_boundary_conditions_scalar[blockspergrid, threadsperblock](phi, mask, value)
    return phi


def launch_obstacle_boundary_conditions_pressure(p, mask, threadsperblock=None):
    """
    launches the obstacle pressure boundary kernel on the GPU.

    Args:
        p (device array): pressure field
        mask (device array): boolean obstacle mask
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        device array: the updated pressure field
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D
    blockspergrid = (
        (mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    obstacle_boundary_conditions_pressure[blockspergrid, threadsperblock](p, mask)
    return p


def apply_all_BC(u, v, w, p, T, bc_config, u_inflow, v_inflow, w_inflow):
    """
    applies all configured domain boundary conditions to the GPU field state.

    The function iterates over the configured faces and dispatches the matching
    CUDA boundary kernels for inflow, outflow, slip-wall and no-slip-wall
    conditions.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        bc_config (dict): boundary-condition configuration indexed by domain side
        u_inflow (float): default inflow velocity in x-direction
        v_inflow (float): default inflow velocity in y-direction
        w_inflow (float): default inflow velocity in z-direction
    Returns:
        tuple: updated velocity, pressure and temperature fields
    """
    for side, bc in bc_config.items():
        bc_type = bc["type"]

        if bc_type == "outflow":
            u, v, w, p, T = launch_outflow_bc(u, v, w, p, T, side)
        elif bc_type == "inflow":
            u, v, w, p, T = launch_inflow_bc(
                u, v, w, p, T,
                side,
                bc.get("u", u_inflow),
                bc.get("v", v_inflow),
                bc.get("w", w_inflow),
                bc.get("T", bc.get("temperature")),
            )
        elif bc_type == "no_slip_wall":
            u, v, w, p, T = launch_no_slip_wall_bc(
                u, v, w, p, T, side, bc.get("T", bc.get("temperature"))
            )
        elif bc_type == "slip_wall":
            u, v, w, p, T = launch_slip_wall_bc(
                u, v, w, p, T, side, bc.get("T", bc.get("temperature"))
            )
        else:
            raise ValueError(f"Unknown boundary condition '{bc_type}' for side '{side}'")

    return u, v, w, p, T


def apply_all_obstacle_BCs(u, v, w, p, T, smoke, fuel, obstacle_mask, obstacle_solid,
                           obstacle_initial_temperature, obstacle_initial_smoke,
                           obstacle_initial_fuel):
    """
    applies all obstacle boundary conditions to the GPU field state.

    Velocity components are forced to zero inside solid obstacles, pressure is
    set to zero there as well and scalar fields are clamped to their configured
    obstacle values.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        smoke (device array): smoke field
        fuel (device array): fuel field
        obstacle_mask (device array): boolean obstacle mask
        obstacle_solid (bool): whether obstacle velocity conditions should be applied
        obstacle_initial_temperature (float): temperature prescribed inside the obstacle
        obstacle_initial_smoke (float): smoke value prescribed inside the obstacle
        obstacle_initial_fuel (float): fuel value prescribed inside the obstacle
    Returns:
        tuple: updated velocity, pressure and scalar fields
    """
    if obstacle_solid:
        u, v, w = launch_obstacle_boundary_conditions_velocity(u, v, w, obstacle_mask)
        p = launch_obstacle_boundary_conditions_pressure(p, obstacle_mask)

    T = launch_obstacle_boundary_conditions_scalar(T, obstacle_mask, obstacle_initial_temperature)
    smoke = launch_obstacle_boundary_conditions_scalar(smoke, obstacle_mask, obstacle_initial_smoke)
    fuel = launch_obstacle_boundary_conditions_scalar(fuel, obstacle_mask, obstacle_initial_fuel)

    return u, v, w, p, T, smoke, fuel
