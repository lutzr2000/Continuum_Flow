import numpy as np
import sys
import os

from numba import cuda, njit, prange, set_num_threads, get_num_threads

import Obstacles
import Output_Functions

# ===============================
# Parameters
# ===============================

BLENDER_PYTHON_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe"
VDB_WRITER_SCRIPT = os.path.join(os.path.dirname(__file__), 'VDB_Writer.py')
CPU_PARALLEL = True

def _build_obstacle_mask(config):
    obstacle_cfg = config["obstacle"]
    resolution_cfg = config["resolution"]

    if obstacle_cfg["shape"] != "sphere":
        raise ValueError(f"Unsupported obstacle shape '{obstacle_cfg['shape']}'")

    sphere_cfg = obstacle_cfg["sphere"]
    return Obstacles.sphere(
        resolution_cfg["NX"],
        resolution_cfg["NY"],
        resolution_cfg["NZ"],
        resolution_cfg["DELTA"],
        resolution_cfg["NX"] * sphere_cfg["x_factor"] * resolution_cfg["DELTA"],
        resolution_cfg["NY"] * sphere_cfg["y_factor"] * resolution_cfg["DELTA"],
        resolution_cfg["NZ"] * sphere_cfg["z_factor"] * resolution_cfg["DELTA"],
        sphere_cfg["radius"],
    )


def apply_config(config):
    global RHO, NU, NU_TEMPERATURE, NU_SMOKE, NU_FUEL
    global TEMPERATURE_DISSIPATION_RATE, TEMPERATURE_PRODUCTION_RATE
    global SMOKE_DISSIPATION_RATE, SMOKE_PRODUCTION_RATE
    global FUEL_BURN_RATE, FUEL_IGNITION_TEMPERATURE, T_REFERENCE
    global BUOANCY_FACTOR, EXPANSION_RATE
    global T_MAX, CFL_MAX
    global MAX_ITER, PRECISION, CPU_PARALLEL, CPU_COUNT
    global DELTA, NX, NY, NZ
    global OUTPUT_FPS, PRINT_FREQUENCY, OUTPUT_TIME_STEP, OUTPUT_STATUS
    global WRITE_QUEUE_SIZE, OUTPATH, OUTPUT_VARIABLES
    global BLENDER_PYTHON_EXE, VDB_WRITER_SCRIPT
    global BC_CONFIG, U_INFLOW, V_INFLOW, W_INFLOW
    global obstacle_mask, OBSTACLE_SOLID
    global OBSTACLE_INTIAL_TEMPERATURE, OBSTACLE_INTIAL_SMOKE, OBSTACLE_INTIAL_FUEL
    global TURBULENCE_AMPLITUDE, TURBULENCE_FREQUENCY, TURBULENCE_SCALE

    fluid = config["fluid"]
    time_cfg = config["time"]
    solver = config["solver"]
    resolution = config["resolution"]
    output = config["output"]
    boundary = config["boundary_conditions"]
    obstacle = config["obstacle"]

    RHO = fluid["RHO"]
    NU = fluid["NU"]
    NU_TEMPERATURE = fluid["NU_TEMPERATURE"]
    NU_SMOKE = fluid["NU_SMOKE"]
    NU_FUEL = fluid["NU_FUEL"]
    TEMPERATURE_DISSIPATION_RATE = fluid["TEMPERATURE_DISSIPATION_RATE"]
    TEMPERATURE_PRODUCTION_RATE = fluid["TEMPERATURE_PRODUCTION_RATE"]
    SMOKE_DISSIPATION_RATE = fluid["SMOKE_DISSIPATION_RATE"]
    SMOKE_PRODUCTION_RATE = fluid["SMOKE_PRODUCTION_RATE"]
    FUEL_BURN_RATE = fluid["FUEL_BURN_RATE"]
    FUEL_IGNITION_TEMPERATURE = fluid["FUEL_IGNITION_TEMPERATURE"]
    T_REFERENCE = fluid["T_REFERENCE"]
    BUOANCY_FACTOR = fluid["BUOANCY_FACTOR"]
    EXPANSION_RATE = fluid["EXPANSION_RATE"]

    T_MAX = time_cfg["T_MAX"]
    CFL_MAX = time_cfg["CFL_MAX"]

    MAX_ITER = solver["MAX_ITER"]
    PRECISION = solver["PRECISION"]
    CPU_COUNT = solver["CPU_COUNT"]

    DELTA = resolution["DELTA"]
    NX = resolution["NX"]
    NY = resolution["NY"]
    NZ = resolution["NZ"]

    OUTPUT_FPS = output["OUTPUT_FPS"]
    PRINT_FREQUENCY = output["PRINT_FREQUENCY"]
    OUTPUT_TIME_STEP = 1.0 / OUTPUT_FPS
    OUTPUT_STATUS = output["OUTPUT_STATUS"]
    WRITE_QUEUE_SIZE = output["WRITE_QUEUE_SIZE"]
    OUTPATH = output["OUTPATH"]
    OUTPUT_VARIABLES = output["OUTPUT_VARIABLES"]

    BC_CONFIG = boundary["BC_CONFIG"]
    U_INFLOW = boundary["U_INFLOW"]
    V_INFLOW = boundary["V_INFLOW"]
    W_INFLOW = boundary["W_INFLOW"]

    OBSTACLE_SOLID = obstacle["solid"]
    OBSTACLE_INTIAL_TEMPERATURE = obstacle["initial_temperature"]
    OBSTACLE_INTIAL_SMOKE = obstacle["initial_smoke"]
    OBSTACLE_INTIAL_FUEL = obstacle["initial_fuel"]
    obstacle_mask = _build_obstacle_mask(config)


def upload_simulation_state_to_gpu():
    """
    Allocates the simulation fields on the host, uploads persistent arrays to the GPU
    and returns both device arrays and scalar kernel constants.

    Returns:
        tuple[dict, dict]:
            - device_state: device arrays used by CUDA kernels
            - gpu_constants: scalar constants that should be passed to CUDA kernels
    """
    precision_dtype = np.dtype(PRECISION)

    #------------Host field allocation-------------------
    u = np.full((NX, NY, NZ), U_INFLOW, dtype=precision_dtype)
    v = np.full((NX, NY, NZ), V_INFLOW, dtype=precision_dtype)
    w = np.full((NX, NY, NZ), W_INFLOW, dtype=precision_dtype)

    p = np.zeros((NX, NY, NZ), dtype=precision_dtype)
    T = np.full((NX, NY, NZ), T_REFERENCE, dtype=precision_dtype)
    smoke = np.zeros((NX, NY, NZ), dtype=precision_dtype)
    fuel = np.zeros((NX, NY, NZ), dtype=precision_dtype)
    flame = np.zeros((NX, NY, NZ), dtype=precision_dtype)

    Fx = np.zeros((NX, NY, NZ), dtype=precision_dtype)
    Fy = np.zeros((NX, NY, NZ), dtype=precision_dtype)
    Fz = np.zeros((NX, NY, NZ), dtype=precision_dtype)

    obstacle_mask_host = np.asarray(obstacle_mask)

    #------------Device upload-------------------
    device_state = {
        "u": cuda.to_device(u),
        "v": cuda.to_device(v),
        "w": cuda.to_device(w),
        "un": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "vn": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "wn": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "u_work": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "v_work": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "w_work": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "p": cuda.to_device(p),
        "pressure_work": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "pressure_rhs": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "T": cuda.to_device(T),
        "temperature_work": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "smoke": cuda.to_device(smoke),
        "smoke_work": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "fuel": cuda.to_device(fuel),
        "fuel_work": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "flame": cuda.to_device(flame),
        "flame_work": cuda.device_array((NX, NY, NZ), dtype=precision_dtype),
        "Fx": cuda.to_device(Fx),
        "Fy": cuda.to_device(Fy),
        "Fz": cuda.to_device(Fz),
        "obstacle_mask": cuda.to_device(obstacle_mask_host),
    }

    gpu_constants = {
        "RHO": precision_dtype.type(RHO),
        "NU": precision_dtype.type(NU),
        "NU_TEMPERATURE": precision_dtype.type(NU_TEMPERATURE),
        "NU_SMOKE": precision_dtype.type(NU_SMOKE),
        "NU_FUEL": precision_dtype.type(NU_FUEL),
        "TEMPERATURE_DISSIPATION_RATE": precision_dtype.type(TEMPERATURE_DISSIPATION_RATE),
        "TEMPERATURE_PRODUCTION_RATE": precision_dtype.type(TEMPERATURE_PRODUCTION_RATE),
        "SMOKE_DISSIPATION_RATE": precision_dtype.type(SMOKE_DISSIPATION_RATE),
        "SMOKE_PRODUCTION_RATE": precision_dtype.type(SMOKE_PRODUCTION_RATE),
        "FUEL_BURN_RATE": precision_dtype.type(FUEL_BURN_RATE),
        "FUEL_IGNITION_TEMPERATURE": precision_dtype.type(FUEL_IGNITION_TEMPERATURE),
        "T_REFERENCE": precision_dtype.type(T_REFERENCE),
        "BUOANCY_FACTOR": precision_dtype.type(BUOANCY_FACTOR),
        "EXPANSION_RATE": precision_dtype.type(EXPANSION_RATE),
        "DELTA": precision_dtype.type(DELTA),
        "U_INFLOW": precision_dtype.type(U_INFLOW),
        "V_INFLOW": precision_dtype.type(V_INFLOW),
        "W_INFLOW": precision_dtype.type(W_INFLOW),
        "OBSTACLE_SOLID": OBSTACLE_SOLID,
        "OBSTACLE_INTIAL_TEMPERATURE": precision_dtype.type(OBSTACLE_INTIAL_TEMPERATURE),
        "OBSTACLE_INTIAL_SMOKE": precision_dtype.type(OBSTACLE_INTIAL_SMOKE),
        "OBSTACLE_INTIAL_FUEL": precision_dtype.type(OBSTACLE_INTIAL_FUEL),
        "NX": NX,
        "NY": NY,
        "NZ": NZ,
    }

    return device_state, gpu_constants

# ===============================
# Methods
# ===============================

@cuda.jit
def update_x_velocity(u, v, w, p, dt, Fx, un, delta, rho, nu):
    """
    CUDA kernel that updates the velocity field in x-direction based on the
    momentum equation. Convection is done by first order upwind, diffusion with
    central differences.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        dt (float): timestep size
        Fx (device array): x-direction body force field
        un (device array): output array for updated x-velocity
        delta (float): grid spacing
        rho (float): density
        nu (float): kinematic viscosity
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta
    pressure_coeff = dt / (2.0 * rho * delta)
    diffusion_coeff = nu * dt / (delta * delta)
    force_coeff = dt / rho

    u_center = u[i, j, k]
    v_center = v[i, j, k]
    w_center = w[i, j, k]

    #------------Upwinding-------------------
    if u_center >= 0.0:
        u_x_high = u_center
        u_x_low = u[i - 1, j, k]
    else:
        u_x_high = u[i + 1, j, k]
        u_x_low = u_center

    if v_center >= 0.0:
        u_y_high = u_center
        u_y_low = u[i, j - 1, k]
    else:
        u_y_high = u[i, j + 1, k]
        u_y_low = u_center

    if w_center >= 0.0:
        u_z_high = u_center
        u_z_low = u[i, j, k - 1]
    else:
        u_z_high = u[i, j, k + 1]
        u_z_low = u_center

    #------------Convection-------------------
    convection = dt_over_delta * (
        u_center * (u_x_high - u_x_low) +
        v_center * (u_y_high - u_y_low) +
        w_center * (u_z_high - u_z_low)
    )

    #------------Diffusion-------------------
    diffusion = diffusion_coeff * (
        (u[i + 1, j, k] - 2.0 * u_center + u[i - 1, j, k]) +
        (u[i, j + 1, k] - 2.0 * u_center + u[i, j - 1, k]) +
        (u[i, j, k + 1] - 2.0 * u_center + u[i, j, k - 1])
    )

    #------------Pressure-------------------
    pressure_gradient = pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])

    #------------Update-------------------
    un[i, j, k] = u_center - convection - pressure_gradient + diffusion + force_coeff * Fx[i, j, k]

@cuda.jit
def update_y_velocity(u, v, w, p, dt, Fy, vn, delta, rho, nu):
    """
    CUDA kernel that updates the velocity field in y-direction based on the
    momentum equation. Convection is done by first order upwind, diffusion with
    central differences.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        dt (float): timestep size
        Fy (device array): y-direction body force field
        vn (device array): output array for updated y-velocity
        delta (float): grid spacing
        rho (float): density
        nu (float): kinematic viscosity
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = v.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta
    pressure_coeff = dt / (2.0 * rho * delta)
    diffusion_coeff = nu * dt / (delta * delta)
    force_coeff = dt / rho

    u_center = u[i, j, k]
    v_center = v[i, j, k]
    w_center = w[i, j, k]

    #------------Upwinding-------------------
    if u_center >= 0.0:
        v_x_high = v_center
        v_x_low = v[i - 1, j, k]
    else:
        v_x_high = v[i + 1, j, k]
        v_x_low = v_center

    if v_center >= 0.0:
        v_y_high = v_center
        v_y_low = v[i, j - 1, k]
    else:
        v_y_high = v[i, j + 1, k]
        v_y_low = v_center

    if w_center >= 0.0:
        v_z_high = v_center
        v_z_low = v[i, j, k - 1]
    else:
        v_z_high = v[i, j, k + 1]
        v_z_low = v_center

    #------------Convection-------------------
    convection = dt_over_delta * (
        u_center * (v_x_high - v_x_low) +
        v_center * (v_y_high - v_y_low) +
        w_center * (v_z_high - v_z_low)
    )

    #------------Diffusion-------------------
    diffusion = diffusion_coeff * (
        (v[i + 1, j, k] - 2.0 * v_center + v[i - 1, j, k]) +
        (v[i, j + 1, k] - 2.0 * v_center + v[i, j - 1, k]) +
        (v[i, j, k + 1] - 2.0 * v_center + v[i, j, k - 1])
    )

    #------------Pressure-------------------
    pressure_gradient = pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])

    #------------Update-------------------
    vn[i, j, k] = v_center - convection - pressure_gradient + diffusion + force_coeff * Fy[i, j, k]

@cuda.jit
def update_z_velocity(u, v, w, p, dt, Fz, wn, delta, rho, nu):
    """
    CUDA kernel that updates the velocity field in z-direction based on the
    momentum equation. Convection is done by first order upwind, diffusion with
    central differences.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        dt (float): timestep size
        Fz (device array): z-direction body force field
        wn (device array): output array for updated z-velocity
        delta (float): grid spacing
        rho (float): density
        nu (float): kinematic viscosity
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = w.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta
    pressure_coeff = dt / (2.0 * rho * delta)
    diffusion_coeff = nu * dt / (delta * delta)
    force_coeff = dt / rho

    u_center = u[i, j, k]
    v_center = v[i, j, k]
    w_center = w[i, j, k]

    #------------Upwinding-------------------
    if u_center >= 0.0:
        w_x_high = w_center
        w_x_low = w[i - 1, j, k]
    else:
        w_x_high = w[i + 1, j, k]
        w_x_low = w_center

    if v_center >= 0.0:
        w_y_high = w_center
        w_y_low = w[i, j - 1, k]
    else:
        w_y_high = w[i, j + 1, k]
        w_y_low = w_center

    if w_center >= 0.0:
        w_z_high = w_center
        w_z_low = w[i, j, k - 1]
    else:
        w_z_high = w[i, j, k + 1]
        w_z_low = w_center

    #------------Convection-------------------
    convection = dt_over_delta * (
        u_center * (w_x_high - w_x_low) +
        v_center * (w_y_high - w_y_low) +
        w_center * (w_z_high - w_z_low)
    )

    #------------Diffusion-------------------
    diffusion = diffusion_coeff * (
        (w[i + 1, j, k] - 2.0 * w_center + w[i - 1, j, k]) +
        (w[i, j + 1, k] - 2.0 * w_center + w[i, j - 1, k]) +
        (w[i, j, k + 1] - 2.0 * w_center + w[i, j, k - 1])
    )

    #------------Pressure-------------------
    pressure_gradient = pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])

    #------------Update-------------------
    wn[i, j, k] = w_center - convection - pressure_gradient + diffusion + force_coeff * Fz[i, j, k]

def launch_update_x_velocity(d_u, d_v, d_w, d_p, dt, d_Fx, d_un, delta, rho, nu, threadsperblock=(8, 8, 8)):
    """
    Launch helper for the CUDA x-velocity update kernel.
    """
    blockspergrid = (
        (d_u.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (d_u.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (d_u.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    update_x_velocity[blockspergrid, threadsperblock](d_u, d_v, d_w, d_p, dt, d_Fx, d_un, delta, rho, nu)
    return d_un

def launch_update_y_velocity(d_u, d_v, d_w, d_p, dt, d_Fy, d_vn, delta, rho, nu, threadsperblock=(8, 8, 8)):
    """
    Launch helper for the CUDA y-velocity update kernel.
    """
    blockspergrid = (
        (d_v.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (d_v.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (d_v.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    update_y_velocity[blockspergrid, threadsperblock](d_u, d_v, d_w, d_p, dt, d_Fy, d_vn, delta, rho, nu)
    return d_vn

def launch_update_z_velocity(d_u, d_v, d_w, d_p, dt, d_Fz, d_wn, delta, rho, nu, threadsperblock=(8, 8, 8)):
    """
    Launch helper for the CUDA z-velocity update kernel.
    """
    blockspergrid = (
        (d_w.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (d_w.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (d_w.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    update_z_velocity[blockspergrid, threadsperblock](d_u, d_v, d_w, d_p, dt, d_Fz, d_wn, delta, rho, nu)
    return d_wn

@cuda.jit
def pressure_equation_right_side(u, v, w, T, b, dt, Fx, Fy, Fz, delta, rho, expansion_rate, t_reference):
    """
    CUDA kernel that computes the right hand side of the pressure Poisson equation.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        T (device array): temperature field
        b (device array): output array for the pressure equation right hand side
        dt (float): timestep size
        Fx (device array): x-direction body force field
        Fy (device array): y-direction body force field
        Fz (device array): z-direction body force field
        delta (float): grid spacing
        rho (float): density
        expansion_rate (float): thermal expansion coupling
        t_reference (float): reference temperature
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    half_inv_delta = 0.5 / delta
    rho_over_dt = rho / dt

    #------------Derivatives on main diagonal-------------------
    du_dx = (u[i + 1, j, k] - u[i - 1, j, k]) * half_inv_delta
    dv_dy = (v[i, j + 1, k] - v[i, j - 1, k]) * half_inv_delta
    dw_dz = (w[i, j, k + 1] - w[i, j, k - 1]) * half_inv_delta

    #------------Of-diagonal derivatives-------------------
    du_dy = (u[i, j + 1, k] - u[i, j - 1, k]) * half_inv_delta
    du_dz = (u[i, j, k + 1] - u[i, j, k - 1]) * half_inv_delta
    dv_dx = (v[i + 1, j, k] - v[i - 1, j, k]) * half_inv_delta
    dv_dz = (v[i, j, k + 1] - v[i, j, k - 1]) * half_inv_delta
    dw_dx = (w[i + 1, j, k] - w[i - 1, j, k]) * half_inv_delta
    dw_dy = (w[i, j + 1, k] - w[i, j - 1, k]) * half_inv_delta

    # This comes from the divergence of the time derivative
    divergence = du_dx + dv_dy + dw_dz

    # This comes from the divergence of the convection term
    nonlinear = (
        du_dx * du_dx +
        dv_dy * dv_dy +
        dw_dz * dw_dz +
        2.0 * (du_dy * dv_dx + du_dz * dw_dx + dv_dz * dw_dy)
    )

    #------------Artifical thermal divergence-------------------
    dFx_dx = (Fx[i + 1, j, k] - Fx[i - 1, j, k]) * half_inv_delta
    dFy_dy = (Fy[i, j + 1, k] - Fy[i, j - 1, k]) * half_inv_delta
    dFz_dz = (Fz[i, j, k + 1] - Fz[i, j, k - 1]) * half_inv_delta
    thermal_divergence = expansion_rate * (T[i, j, k] - t_reference)

    #------------Right hand side-------------------
    b[i, j, k] = rho_over_dt * (divergence - thermal_divergence) - rho * (
        nonlinear + dFx_dx + dFy_dy + dFz_dz
    )


def launch_pressure_equation_right_side(d_u, d_v, d_w, d_T, d_b, dt, d_Fx, d_Fy, d_Fz, delta, rho,
                                        expansion_rate, t_reference, threadsperblock=(8, 8, 8)):
    """
    Launch helper for the CUDA pressure RHS kernel.
    """
    blockspergrid = (
        (d_u.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (d_u.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (d_u.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    pressure_equation_right_side[blockspergrid, threadsperblock](
        d_u, d_v, d_w, d_T, d_b, dt, d_Fx, d_Fy, d_Fz, delta, rho, expansion_rate, t_reference
    )
    return d_b


@cuda.jit
def pressure_poisson_jacobi_step(p_old, p_new, b, delta):
    """
    CUDA kernel for one Jacobi iteration of the pressure Poisson solve.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = p_old.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    delta2 = delta * delta
    p_new[i, j, k] = (
        p_old[i + 1, j, k] + p_old[i - 1, j, k] +
        p_old[i, j + 1, k] + p_old[i, j - 1, k] +
        p_old[i, j, k + 1] + p_old[i, j, k - 1] -
        delta2 * b[i, j, k]
    ) / 6.0


@cuda.jit
def pressure_poisson_apply_neumann_x(p):
    j, k = cuda.grid(2)
    ny = p.shape[1]
    nz = p.shape[2]

    if j >= ny or k >= nz:
        return

    nx = p.shape[0]
    p[0, j, k] = p[1, j, k]
    p[nx - 1, j, k] = p[nx - 2, j, k]


@cuda.jit
def pressure_poisson_apply_neumann_y(p):
    i, k = cuda.grid(2)
    nx = p.shape[0]
    nz = p.shape[2]

    if i >= nx or k >= nz:
        return

    ny = p.shape[1]
    p[i, 0, k] = p[i, 1, k]
    p[i, ny - 1, k] = p[i, ny - 2, k]


@cuda.jit
def pressure_poisson_apply_neumann_z(p):
    i, j = cuda.grid(2)
    nx = p.shape[0]
    ny = p.shape[1]

    if i >= nx or j >= ny:
        return

    nz = p.shape[2]
    p[i, j, 0] = p[i, j, 1]
    p[i, j, nz - 1] = p[i, j, nz - 2]


def launch_pressure_poisson_jacobi_step(d_p_old, d_p_new, d_b, delta, threadsperblock=(8, 8, 8)):
    """
    Launch helper for one Jacobi iteration of the pressure Poisson solve.
    """
    blockspergrid = (
        (d_p_old.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (d_p_old.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (d_p_old.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    pressure_poisson_jacobi_step[blockspergrid, threadsperblock](d_p_old, d_p_new, d_b, delta)
    return d_p_new


def launch_pressure_poisson_neumann_bcs(d_p, threadsperblock=(16, 16)):
    """
    Launch helpers for the three Neumann pressure boundary kernels.
    """
    blockspergrid_x = (
        (d_p.shape[1] + threadsperblock[0] - 1) // threadsperblock[0],
        (d_p.shape[2] + threadsperblock[1] - 1) // threadsperblock[1],
    )
    blockspergrid_y = (
        (d_p.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (d_p.shape[2] + threadsperblock[1] - 1) // threadsperblock[1],
    )
    blockspergrid_z = (
        (d_p.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (d_p.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
    )

    pressure_poisson_apply_neumann_x[blockspergrid_x, threadsperblock](d_p)
    pressure_poisson_apply_neumann_y[blockspergrid_y, threadsperblock](d_p)
    pressure_poisson_apply_neumann_z[blockspergrid_z, threadsperblock](d_p)
    return d_p


def pressure_poisson(u, v, w, p, T, p_work, b, dt, Fx, Fy, Fz, delta, rho, expansion_rate, t_reference,
                     max_iter=10, threadsperblock_3d=(8, 8, 8), threadsperblock_2d=(16, 16)):
    """
    Host-side pressure Poisson solve that launches CUDA kernels for the RHS,
    Jacobi iterations and Neumann boundary conditions.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        p_work (device array): work array for the pressure iteration
        b (device array): work array for the pressure equation right hand side
        dt (float): timestep size
        Fx (device array): x-direction body force field
        Fy (device array): y-direction body force field
        Fz (device array): z-direction body force field
        delta (float): grid spacing
        rho (float): density
        expansion_rate (float): thermal expansion coupling
        t_reference (float): reference temperature
        max_iter (int): number of Jacobi iterations
    Returns:
        device array: updated pressure field
    """
    launch_pressure_equation_right_side(
        u, v, w, T, b, dt, Fx, Fy, Fz, delta, rho, expansion_rate, t_reference, threadsperblock_3d
    )

    p_old = p
    p_new = p_work

    for _ in range(max_iter):
        launch_pressure_poisson_jacobi_step(p_old, p_new, b, delta, threadsperblock_3d)
        launch_pressure_poisson_neumann_bcs(p_new, threadsperblock_2d)

        #------------Swap-------------------
        p_old, p_new = p_new, p_old

    return p_old


@cuda.jit
def apply_neumann_boundary_condition(field, axis, side_index):
    """
    CUDA kernel that applies a zero-gradient boundary condition to one side of a 3D field.
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
    CUDA kernel that applies a fixed-value boundary condition to one side of a 3D field.
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
    CUDA kernel that applies no-slip velocity conditions inside a 3D obstacle mask.
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
def obstacle_boundary_conditions_pressure(p, mask):
    """
    CUDA kernel that applies zero pressure inside a 3D obstacle mask.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if mask[i, j, k]:
        p[i, j, k] = 0.0


@cuda.jit
def obstacle_boundary_conditions_scalar(phi, mask, value):
    """
    CUDA kernel that applies a fixed scalar value inside a 3D obstacle mask.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if mask[i, j, k]:
        phi[i, j, k] = value


def _side_to_axis_and_index(side):
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


def launch_neumann_boundary_condition(field, side, threadsperblock=(16, 16)):
    """
    Launch helper for a zero-gradient boundary condition on one domain side.
    """
    axis, side_index = _side_to_axis_and_index(side)
    blockspergrid = _boundary_blockspergrid(field.shape, axis, threadsperblock)
    apply_neumann_boundary_condition[blockspergrid, threadsperblock](field, axis, side_index)
    return field


def launch_dirichlet_boundary_condition(field, side, value, threadsperblock=(16, 16)):
    """
    Launch helper for a fixed-value boundary condition on one domain side.
    """
    axis, side_index = _side_to_axis_and_index(side)
    blockspergrid = _boundary_blockspergrid(field.shape, axis, threadsperblock)
    apply_dirichlet_boundary_condition[blockspergrid, threadsperblock](field, axis, side_index, value)
    return field


def launch_inflow_bc(u, v, w, p, T, side, u_inflow, v_inflow, w_inflow, t_inflow=None, threadsperblock=(16, 16)):
    """
    Launch helper for inflow boundary conditions on GPU arrays.
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


def launch_outflow_bc(u, v, w, p, T, side, threadsperblock=(16, 16)):
    """
    Launch helper for outflow boundary conditions on GPU arrays.
    """
    launch_neumann_boundary_condition(u, side, threadsperblock)
    launch_neumann_boundary_condition(v, side, threadsperblock)
    launch_neumann_boundary_condition(w, side, threadsperblock)
    launch_neumann_boundary_condition(p, side, threadsperblock)
    launch_neumann_boundary_condition(T, side, threadsperblock)
    return u, v, w, p, T


def launch_slip_wall_bc(u, v, w, p, T, side, t_wall=None, threadsperblock=(16, 16)):
    """
    Launch helper for slip-wall boundary conditions on GPU arrays.
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


def launch_no_slip_wall_bc(u, v, w, p, T, side, t_wall=None, threadsperblock=(16, 16)):
    """
    Launch helper for no-slip-wall boundary conditions on GPU arrays.
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


def launch_obstacle_boundary_conditions_velocity(u, v, w, mask, threadsperblock=(8, 8, 8)):
    """
    Launch helper for obstacle no-slip velocity conditions on GPU arrays.
    """
    blockspergrid = (
        (mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    obstacle_boundary_conditions_velocity[blockspergrid, threadsperblock](u, v, w, mask)
    return u, v, w


def launch_obstacle_boundary_conditions_pressure(p, mask, threadsperblock=(8, 8, 8)):
    """
    Launch helper for obstacle pressure conditions on GPU arrays.
    """
    blockspergrid = (
        (mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    obstacle_boundary_conditions_pressure[blockspergrid, threadsperblock](p, mask)
    return p


def launch_obstacle_boundary_conditions_scalar(phi, mask, value, threadsperblock=(8, 8, 8)):
    """
    Launch helper for obstacle scalar conditions on GPU arrays.
    """
    blockspergrid = (
        (mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    obstacle_boundary_conditions_scalar[blockspergrid, threadsperblock](phi, mask, value)
    return phi


@cuda.jit
def copy_field_3d(src, dst):
    """
    CUDA kernel that copies one 3D field into another equally shaped 3D field.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = src.shape

    if i >= nx or j >= ny or k >= nz:
        return

    dst[i, j, k] = src[i, j, k]


def launch_copy_field_3d(src, dst, threadsperblock=(8, 8, 8)):
    """
    Launch helper for copying a 3D field on the GPU.
    """
    blockspergrid = (
        (src.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (src.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (src.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    copy_field_3d[blockspergrid, threadsperblock](src, dst)
    return dst


def copy_device_fields_to_host(field_map):
    """
    Copies a dictionary of device arrays to host NumPy arrays.
    """
    return {name: device_array.copy_to_host() for name, device_array in field_map.items()}


@cuda.reduce
def max_abs_reduce(a, b):
    """
    Reduction that returns the maximum absolute value over an array.
    """
    abs_a = abs(a)
    abs_b = abs(b)
    return abs_a if abs_a > abs_b else abs_b


def max_abs_device_array(device_array):
    """
    Applies the CUDA max-abs reduction to a flattened 1D view of a device array.
    """
    return max_abs_reduce(device_array.reshape(device_array.size))


def compute_cfl_gpu(d_u, d_v, d_w, dt, delta):
    """
    Computes the CFL number on the GPU and returns a host scalar.
    """
    max_u = max_abs_device_array(d_u)
    max_v = max_abs_device_array(d_v)
    max_w = max_abs_device_array(d_w)

    cfl_x = max_u * dt / delta
    cfl_y = max_v * dt / delta
    cfl_z = max_w * dt / delta
    return max(cfl_x, cfl_y, cfl_z)


def compute_new_timestep_gpu(d_u, d_v, d_w, d_Fx, d_Fy, d_Fz, rho, delta, nu, cfl_max):
    """
    Computes a stable timestep using GPU reductions and returns a host scalar.
    """
    eps = 1e-12

    abs_u_max = max_abs_device_array(d_u)
    abs_v_max = max_abs_device_array(d_v)
    abs_w_max = max_abs_device_array(d_w)
    abs_Fx_max = max_abs_device_array(d_Fx)
    abs_Fy_max = max_abs_device_array(d_Fy)
    abs_Fz_max = max_abs_device_array(d_Fz)

    cfl_delta = cfl_max * delta
    dt_conv = min(
        cfl_delta / max(abs_u_max, eps),
        cfl_delta / max(abs_v_max, eps),
        cfl_delta / max(abs_w_max, eps),
    )
    dt_diff = delta * delta / (6.0 * nu)
    dt_forcing = min(
        cfl_delta * rho / max(abs_Fx_max, eps),
        cfl_delta * rho / max(abs_Fy_max, eps),
        cfl_delta * rho / max(abs_Fz_max, eps),
    )

    return min(dt_conv, dt_diff, dt_forcing)

def apply_all_obstacle_BCs(u, v, w, p, T, smoke, fuel, obstacle_mask):
    """
    Applies obstacle boundary conditions to all fields. If OBSTACLE_SOLID is True, the velocity and pressure BCs will be applied.
    """
    if OBSTACLE_SOLID:
        u, v, w = launch_obstacle_boundary_conditions_velocity(u, v, w, obstacle_mask)
        p = launch_obstacle_boundary_conditions_pressure(p, obstacle_mask)

    T = launch_obstacle_boundary_conditions_scalar(T, obstacle_mask, OBSTACLE_INTIAL_TEMPERATURE)
    smoke = launch_obstacle_boundary_conditions_scalar(smoke, obstacle_mask, OBSTACLE_INTIAL_SMOKE)
    fuel = launch_obstacle_boundary_conditions_scalar(fuel, obstacle_mask, OBSTACLE_INTIAL_FUEL)

    return u, v, w, p, T, smoke, fuel

def apply_all_BC(u, v, w, p, T):
    """
    Applies all boundary conditions to the velocity, pressure and temperature fields.
    """
    for side, bc in BC_CONFIG.items():
        bc_type = bc["type"]

        if bc_type == "outflow":
            u, v, w, p, T = launch_outflow_bc(u, v, w, p, T, side)

        elif bc_type == "inflow":
            u, v, w, p, T = launch_inflow_bc(
                u, v, w, p, T,
                side,
                bc.get("u", U_INFLOW),
                bc.get("v", V_INFLOW),
                bc.get("w", W_INFLOW),
                bc.get("T", bc.get("temperature")),
            )

        elif bc_type == "no_slip_wall":
            u, v, w, p, T = launch_no_slip_wall_bc(u, v, w, p, T, side, bc.get("T", bc.get("temperature")))

        elif bc_type == "slip_wall":
            u, v, w, p, T = launch_slip_wall_bc(u, v, w, p, T, side, bc.get("T", bc.get("temperature")))

        else:
            raise ValueError(f"Unknown boundary condition '{bc_type}' for side '{side}'")

    return u, v, w, p, T

# ===============================
# Main
# ===============================

def main(config=None):
    apply_config(config)

    #------------Initialise-------------------
    print('Initialise')
    print('Cell count: ', int(NX * NY * NZ))

    if CPU_PARALLEL:
        available_threads = os.cpu_count() or 1
        set_num_threads(CPU_COUNT)
        print(f'Numba solver threads: {get_num_threads()} / {available_threads} Cores')

    #------------Fields-------------------
    device_state, gpu_constants = upload_simulation_state_to_gpu()

    d_u = device_state["u"]
    d_v = device_state["v"]
    d_w = device_state["w"]
    d_un = device_state["un"]
    d_vn = device_state["vn"]
    d_wn = device_state["wn"]
    d_u_work = device_state["u_work"]
    d_v_work = device_state["v_work"]
    d_w_work = device_state["w_work"]
    d_p = device_state["p"]
    d_pressure_work = device_state["pressure_work"]
    d_pressure_rhs = device_state["pressure_rhs"]
    d_T = device_state["T"]
    d_smoke = device_state["smoke"]
    d_fuel = device_state["fuel"]
    d_flame = device_state["flame"]
    d_Fx = device_state["Fx"]
    d_Fy = device_state["Fy"]
    d_Fz = device_state["Fz"]
    d_obstacle_mask = device_state["obstacle_mask"]

    #------------BCs-------------------
    d_u, d_v, d_w, d_p, d_T = apply_all_BC(d_u, d_v, d_w, d_p, d_T)

    #------------Obstacle-------------------
    d_u, d_v, d_w, d_p, d_T, d_smoke, d_fuel = apply_all_obstacle_BCs(
        d_u, d_v, d_w, d_p, d_T, d_smoke, d_fuel, d_obstacle_mask
    )

    host_fields = copy_device_fields_to_host({
        "u": d_u,
        "v": d_v,
        "w": d_w,
        "p": d_p,
        "T": d_T,
        "smoke": d_smoke,
        "fuel": d_fuel,
        "flame": d_flame,
    })

    #------------Dynamic time step-------------------
    t = 0.0
    dt = compute_new_timestep_gpu(
        d_u, d_v, d_w, d_Fx, d_Fy, d_Fz,
        gpu_constants["RHO"], gpu_constants["DELTA"], gpu_constants["NU"], CFL_MAX
    )
    if dt > 1.0 / OUTPUT_FPS:
        dt = 1.0 / OUTPUT_FPS

    #------------Prepare Output-------------------
    next_output_time = 0.0
    output_index = 0
    
    template_fields = Output_Functions.create_output_field_map(
        host_fields["u"],
        host_fields["v"],
        host_fields["w"],
        host_fields["p"],
        host_fields["T"],
        host_fields["smoke"],
        host_fields["fuel"],
        host_fields["flame"],
    )
    write_queue, buffer_pool, writer_thread, shared_memory_blocks = Output_Functions.setup_output(
        OUTPATH, OUTPUT_VARIABLES, template_fields, WRITE_QUEUE_SIZE, BLENDER_PYTHON_EXE, VDB_WRITER_SCRIPT, DELTA
    )

    #------------Main time loop-------------------
    print('Start time iteration')
    sys.stdout.write('\rProgress: [0%]')

    while t < T_MAX:
        #------------Start-------------------
        launch_copy_field_3d(d_u, d_un)
        launch_copy_field_3d(d_v, d_vn)
        launch_copy_field_3d(d_w, d_wn)

        #------------Pressure-------------------
        d_p = pressure_poisson(
            d_un, d_vn, d_wn, d_p, d_T, d_pressure_work, d_pressure_rhs, dt, d_Fx, d_Fy, d_Fz,
            gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["EXPANSION_RATE"],
            gpu_constants["T_REFERENCE"], MAX_ITER
        )

        #------------Velocity-------------------
        d_u = launch_update_x_velocity(
            d_un, d_vn, d_wn, d_p, dt, d_Fx, d_u_work,
            gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["NU"]
        )
        d_v = launch_update_y_velocity(
            d_un, d_vn, d_wn, d_p, dt, d_Fy, d_v_work,
            gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["NU"]
        )
        d_w = launch_update_z_velocity(
            d_un, d_vn, d_wn, d_p, dt, d_Fz, d_w_work,
            gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["NU"]
        )

        d_u_work, d_u = d_u, d_u_work
        d_v_work, d_v = d_v, d_v_work
        d_w_work, d_w = d_w, d_w_work

        #------------BCs-------------------
        d_u, d_v, d_w, d_p, d_T = apply_all_BC(d_u, d_v, d_w, d_p, d_T)

        #------------Obstacle-------------------
        d_u, d_v, d_w, d_p, d_T, d_smoke, d_fuel = apply_all_obstacle_BCs(
            d_u, d_v, d_w, d_p, d_T, d_smoke, d_fuel, d_obstacle_mask
        )

        #------------Output-------------------
        while t >= next_output_time:
            host_fields = copy_device_fields_to_host({
                "u": d_u,
                "v": d_v,
                "w": d_w,
                "p": d_p,
                "T": d_T,
                "smoke": d_smoke,
                "fuel": d_fuel,
                "flame": d_flame,
            })
            current_fields = Output_Functions.create_output_field_map(
                host_fields["u"],
                host_fields["v"],
                host_fields["w"],
                host_fields["p"],
                host_fields["T"],
                host_fields["smoke"],
                host_fields["fuel"],
                host_fields["flame"],
            )
            Output_Functions.enqueue_output(write_queue, buffer_pool, OUTPUT_VARIABLES, current_fields, output_index, t)

            output_index += 1
            next_output_time += OUTPUT_TIME_STEP
            sys.stdout.write(f'\rProgress: [{(t / T_MAX * 100):.3f}%]')
            sys.stdout.flush()

            if OUTPUT_STATUS:
                print('#################################################')
                print(f'Simulation time {t} sec')
                CFL = compute_cfl_gpu(d_u, d_v, d_w, dt, gpu_constants["DELTA"])
                print(f'Current dt: {np.round(dt, 5)}')
                print(f'CFL-Condition: {np.round(CFL, 5)}')

        #------------Dynamic time step-------------------
        t += dt

        dt_new = compute_new_timestep_gpu(
            d_u, d_v, d_w, d_Fx, d_Fy, d_Fz,
            gpu_constants["RHO"], gpu_constants["DELTA"], gpu_constants["NU"], CFL_MAX
        )

        dt_max_increase = dt * 1.5
        dt_max_decrease = dt * 0.5

        if dt_new > dt_max_increase:
            dt = dt_max_increase
        elif dt_new < dt_max_decrease:
            dt = dt_max_decrease
        else:
            dt = dt_new

        if dt > 1.0 / OUTPUT_FPS:
            dt = 1.0 / OUTPUT_FPS

    #------------Empty write queue-------------------
    Output_Functions.shutdown_output(write_queue, writer_thread, shared_memory_blocks)

    #------------Conclusion-------------------
    print('Simulation finished!')
