import sys
from time import perf_counter

import numpy as np
from numba import cuda

import Kernel.Boundary_Conditions.Domain_BC as BC
import Kernel.Helper_Functions as Helper_Functions
import Kernel.Boundary_Conditions.Obstacle_BC as Obstacle_BC
import Kernel.Output.Output_Functions as Output_Functions
import Kernel.Boundary_Conditions.Source_BC as Source_BC
import Kernel.Time_Step as Time_Step

# ===============================
# Parameters
# ===============================

THREADS_PER_BLOCK_3D = (8, 8, 8)
THREADS_PER_BLOCK_2D = (4, 4)
REDUCTION_THREADS_PER_BLOCK = 512
BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D
BC.THREADS_PER_BLOCK_2D = THREADS_PER_BLOCK_2D
Obstacle_BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D
Source_BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D
Helper_Functions.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D
Helper_Functions.THREADS_PER_BLOCK_2D = THREADS_PER_BLOCK_2D
Helper_Functions.REDUCTION_THREADS_PER_BLOCK = REDUCTION_THREADS_PER_BLOCK

# ===============================
# Methods
# ===============================

@cuda.jit
def update_velocity(u, v, w, p, dt, Fx, Fy, Fz, un, vn, wn, delta, rho, nu):
    """
    CUDA kernel that updates all three velocity components based on the
    momentum equation. Convection is done by first order upwind, diffusion with
    central differences.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        dt (float): timestep size
        Fx (device array): x-direction body force field
        Fy (device array): y-direction body force field
        Fz (device array): z-direction body force field
        un (device array): output array for updated x-velocity
        vn (device array): output array for updated y-velocity
        wn (device array): output array for updated z-velocity
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
        v_x_high = v_center
        v_x_low = v[i - 1, j, k]
        w_x_high = w_center
        w_x_low = w[i - 1, j, k]
    else:
        u_x_high = u[i + 1, j, k]
        u_x_low = u_center
        v_x_high = v[i + 1, j, k]
        v_x_low = v_center
        w_x_high = w[i + 1, j, k]
        w_x_low = w_center

    if v_center >= 0.0:
        u_y_high = u_center
        u_y_low = u[i, j - 1, k]
        v_y_high = v_center
        v_y_low = v[i, j - 1, k]
        w_y_high = w_center
        w_y_low = w[i, j - 1, k]

    else:
        u_y_high = u[i, j + 1, k]
        u_y_low = u_center
        v_y_high = v[i, j + 1, k]
        v_y_low = v_center
        w_y_high = w[i, j + 1, k]
        w_y_low = w_center

    if w_center >= 0.0:
        u_z_high = u_center
        u_z_low = u[i, j, k - 1]
        v_z_high = v_center
        v_z_low = v[i, j, k - 1]
        w_z_high = w_center
        w_z_low = w[i, j, k - 1]

    else:
        u_z_high = u[i, j, k + 1]
        u_z_low = u_center
        v_z_high = v[i, j, k + 1]
        v_z_low = v_center
        w_z_high = w[i, j, k + 1]
        w_z_low = w_center

    #------------Convection-------------------
    convection_x = dt_over_delta * (
        u_center * (u_x_high - u_x_low) +
        v_center * (u_y_high - u_y_low) +
        w_center * (u_z_high - u_z_low)
    )

    convection_y = dt_over_delta * (
        u_center * (v_x_high - v_x_low) +
        v_center * (v_y_high - v_y_low) +
        w_center * (v_z_high - v_z_low)
    )

    convection_z = dt_over_delta * (
        u_center * (w_x_high - w_x_low) +
        v_center * (w_y_high - w_y_low) +
        w_center * (w_z_high - w_z_low)
    )

    #------------Diffusion-------------------
    diffusion_x = diffusion_coeff * (
        (u[i + 1, j, k] - 2.0 * u_center + u[i - 1, j, k]) +
        (u[i, j + 1, k] - 2.0 * u_center + u[i, j - 1, k]) +
        (u[i, j, k + 1] - 2.0 * u_center + u[i, j, k - 1])
    )
    diffusion_y = diffusion_coeff * (
        (v[i + 1, j, k] - 2.0 * v_center + v[i - 1, j, k]) +
        (v[i, j + 1, k] - 2.0 * v_center + v[i, j - 1, k]) +
        (v[i, j, k + 1] - 2.0 * v_center + v[i, j, k - 1])
    )
    diffusion_z = diffusion_coeff * (
        (w[i + 1, j, k] - 2.0 * w_center + w[i - 1, j, k]) +
        (w[i, j + 1, k] - 2.0 * w_center + w[i, j - 1, k]) +
        (w[i, j, k + 1] - 2.0 * w_center + w[i, j, k - 1])
    )

    #------------Pressure-------------------
    pressure_gradient_x = pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
    pressure_gradient_y = pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
    pressure_gradient_z = pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])

    #------------Update-------------------
    un[i, j, k] = u_center - convection_x - pressure_gradient_x + diffusion_x + force_coeff * Fx[i, j, k]
    vn[i, j, k] = v_center - convection_y - pressure_gradient_y + diffusion_y + force_coeff * Fy[i, j, k]
    wn[i, j, k] = w_center - convection_z - pressure_gradient_z + diffusion_z + force_coeff * Fz[i, j, k]

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


@cuda.jit
def _pressure_poisson_jacobi_step(p_old, p_new, b, delta):
    """
    performs one Jacobi iteration of the 3D pressure Poisson equation on the GPU.

    Each interior cell is updated from the previous pressure iterate `p_old`
    using the 7-point stencil of the Laplace operator and the already computed
    right hand side `b`. Boundary cells are skipped here because their values
    are imposed separately by `_pressure_poisson_apply_neumann_bcs`.

    Args:
        p_old (device array): pressure field from the previous Jacobi iteration
        p_new (device array): output array for the updated pressure field
        b (device array): right hand side of the pressure Poisson equation
        delta (float): grid spacing
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = p_old.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if 0 < i < nx - 1 and 0 < j < ny - 1 and 0 < k < nz - 1:
        delta2 = delta * delta
        p_new[i, j, k] = (
            p_old[i + 1, j, k] + p_old[i - 1, j, k] +
            p_old[i, j + 1, k] + p_old[i, j - 1, k] +
            p_old[i, j, k + 1] + p_old[i, j, k - 1] -
            delta2 * b[i, j, k]
        ) / 6.0


@cuda.jit
def _pressure_poisson_apply_neumann_bcs(p):
    """
    applies the hard-coded zero-gradient pressure boundary conditions on all
    six domain faces on the GPU.

    The pressure Poisson solve uses homogeneous Neumann boundary conditions,
    meaning the pressure at the boundary is copied from the adjacent interior
    cell. This kernel writes the boundary values after each Jacobi iteration so
    the next iteration starts from a pressure field with valid boundary values.

    Args:
        p (device array): pressure field whose domain boundaries will be updated
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


def pressure_poisson(u, v, w, p, T, p_work, b, dt, Fx, Fy, Fz, delta, rho, expansion_rate, t_reference,
                     max_iter=10, threadsperblock_3d=None):
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
    if threadsperblock_3d is None:
        threadsperblock_3d = THREADS_PER_BLOCK_3D
    blockspergrid_3d = (
        (u.shape[0] + threadsperblock_3d[0] - 1) // threadsperblock_3d[0],
        (u.shape[1] + threadsperblock_3d[1] - 1) // threadsperblock_3d[1],
        (u.shape[2] + threadsperblock_3d[2] - 1) // threadsperblock_3d[2],
    )

    pressure_equation_right_side[blockspergrid_3d, threadsperblock_3d](
        u, v, w, T, b, dt, Fx, Fy, Fz, delta, rho, expansion_rate, t_reference
    )

    p_old = p
    p_new = p_work

    for _ in range(max_iter):
        _pressure_poisson_jacobi_step[blockspergrid_3d, threadsperblock_3d](p_old, p_new, b, delta)
        _pressure_poisson_apply_neumann_bcs[blockspergrid_3d, threadsperblock_3d](p_new)

        #------------Swap-------------------
        p_old, p_new = p_new, p_old

    return p_old


@cuda.jit
def buoyancy_approximation(T, Fz, buoyancy_factor, t_reference):
    """
    computes the buoyancy force in z-direction with the Boussinesq approximation on the GPU.

    Each thread updates one interior cell of the z-direction body-force field.
    The force is derived from the local temperature difference to the reference
    temperature and scaled by gravity and the configured buoyancy factor.

    Args:
        T (device array): temperature field
        Fz (device array): z-direction force field that will be updated in-place
        buoyancy_factor (float): thermal expansion coefficient used by the model
        t_reference (float): reference temperature
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = T.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    g = 9.81
    Fz[i, j, k] = g * buoyancy_factor * (T[i, j, k] - t_reference)


@cuda.jit
def update_scalar_fields(T, smoke, fuel, u, v, w, dt, T_out, smoke_out, fuel_out, flame_out,
                         delta, nu_temperature, nu_smoke, nu_fuel,
                         temperature_dissipation_rate, temperature_production_rate,
                         smoke_dissipation_rate, smoke_production_rate,
                         fuel_burn_rate, fuel_ignition_temperature, t_reference):
    """
    updates temperature, smoke and fuel in one GPU transport sweep.

    Convection is evaluated with first-order upwinding, diffusion with central
    differences and the source terms model fuel ignition, temperature release
    and smoke production. A flame indicator is written alongside the updated
    scalar fields.

    Args:
        T (device array): temperature field
        smoke (device array): smoke density field
        fuel (device array): fuel density field
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        dt (float): timestep size
        T_out (device array): output array for updated temperature
        smoke_out (device array): output array for updated smoke density
        fuel_out (device array): output array for updated fuel density
        flame_out (device array): output array for the flame indicator
        delta (float): grid spacing
        nu_temperature (float): diffusion coefficient for temperature
        nu_smoke (float): diffusion coefficient for smoke
        nu_fuel (float): diffusion coefficient for fuel
        temperature_dissipation_rate (float): temperature dissipation coefficient
        temperature_production_rate (float): temperature production coefficient
        smoke_dissipation_rate (float): smoke dissipation coefficient
        smoke_production_rate (float): smoke production coefficient
        fuel_burn_rate (float): burning rate for ignited fuel
        fuel_ignition_temperature (float): ignition threshold for fuel burning
        t_reference (float): reference temperature
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dt_over_delta = dt / delta
    dt_over_delta2 = dt / (delta * delta)
    temp_diffusion_coeff = nu_temperature * dt_over_delta2
    smoke_diffusion_coeff = nu_smoke * dt_over_delta2
    fuel_diffusion_coeff = nu_fuel * dt_over_delta2

    uijk = u[i, j, k]
    vijk = v[i, j, k]
    wijk = w[i, j, k]

    T_center = T[i, j, k]
    T_xm = T[i - 1, j, k]
    T_xp = T[i + 1, j, k]
    T_ym = T[i, j - 1, k]
    T_yp = T[i, j + 1, k]
    T_zm = T[i, j, k - 1]
    T_zp = T[i, j, k + 1]

    smoke_center = smoke[i, j, k]
    smoke_xm = smoke[i - 1, j, k]
    smoke_xp = smoke[i + 1, j, k]
    smoke_ym = smoke[i, j - 1, k]
    smoke_yp = smoke[i, j + 1, k]
    smoke_zm = smoke[i, j, k - 1]
    smoke_zp = smoke[i, j, k + 1]

    fuel_center = fuel[i, j, k]
    fuel_xm = fuel[i - 1, j, k]
    fuel_xp = fuel[i + 1, j, k]
    fuel_ym = fuel[i, j - 1, k]
    fuel_yp = fuel[i, j + 1, k]
    fuel_zm = fuel[i, j, k - 1]
    fuel_zp = fuel[i, j, k + 1]

    if uijk >= 0.0:
        temp_dx = T_center - T_xm
        smoke_dx = smoke_center - smoke_xm
        fuel_dx = fuel_center - fuel_xm
    else:
        temp_dx = T_xp - T_center
        smoke_dx = smoke_xp - smoke_center
        fuel_dx = fuel_xp - fuel_center

    if vijk >= 0.0:
        temp_dy = T_center - T_ym
        smoke_dy = smoke_center - smoke_ym
        fuel_dy = fuel_center - fuel_ym
    else:
        temp_dy = T_yp - T_center
        smoke_dy = smoke_yp - smoke_center
        fuel_dy = fuel_yp - fuel_center

    if wijk >= 0.0:
        temp_dz = T_center - T_zm
        smoke_dz = smoke_center - smoke_zm
        fuel_dz = fuel_center - fuel_zm
    else:
        temp_dz = T_zp - T_center
        smoke_dz = smoke_zp - smoke_center
        fuel_dz = fuel_zp - fuel_center

    temp_convection = dt_over_delta * (uijk * temp_dx + vijk * temp_dy + wijk * temp_dz)
    smoke_convection = dt_over_delta * (uijk * smoke_dx + vijk * smoke_dy + wijk * smoke_dz)
    fuel_convection = dt_over_delta * (uijk * fuel_dx + vijk * fuel_dy + wijk * fuel_dz)

    temp_diffusion = temp_diffusion_coeff * (
        (T_xp - 2.0 * T_center + T_xm) +
        (T_yp - 2.0 * T_center + T_ym) +
        (T_zp - 2.0 * T_center + T_zm)
    )
    smoke_diffusion = smoke_diffusion_coeff * (
        (smoke_xp - 2.0 * smoke_center + smoke_xm) +
        (smoke_yp - 2.0 * smoke_center + smoke_ym) +
        (smoke_zp - 2.0 * smoke_center + smoke_zm)
    )
    fuel_diffusion = fuel_diffusion_coeff * (
        (fuel_xp - 2.0 * fuel_center + fuel_xm) +
        (fuel_yp - 2.0 * fuel_center + fuel_ym) +
        (fuel_zp - 2.0 * fuel_center + fuel_zm)
    )

    if T_center > fuel_ignition_temperature:
        fuel_source = -fuel_burn_rate * fuel_center
    else:
        fuel_source = 0.0

    temperature_source = (
        -temperature_dissipation_rate * (T_center - t_reference) +
        temperature_production_rate * (-fuel_source)
    )
    smoke_source = smoke_production_rate * (-fuel_source) - smoke_dissipation_rate * smoke_center

    T_out[i, j, k] = T_center - temp_convection + temp_diffusion + dt * temperature_source
    smoke_out[i, j, k] = smoke_center - smoke_convection + smoke_diffusion + dt * smoke_source
    fuel_out[i, j, k] = fuel_center - fuel_convection + fuel_diffusion + dt * fuel_source
    flame_out[i, j, k] = 1.0 if fuel_source < 0.0 else 0.0


# ===============================
# Main
# ===============================

def main(config=None):
    simulation_params = Helper_Functions.apply_config(config)

    T_MAX = simulation_params["T_MAX"]
    CFL_MAX = simulation_params["CFL_MAX"]
    MAX_ITER = simulation_params["MAX_ITER"]
    DELTA = simulation_params["DELTA"]
    NX = simulation_params["NX"]
    NY = simulation_params["NY"]
    NZ = simulation_params["NZ"]
    OUTPUT_FPS = simulation_params["OUTPUT_FPS"]
    OUTPUT_TIME_STEP = simulation_params["OUTPUT_TIME_STEP"]
    OUTPUT_STATUS = simulation_params["OUTPUT_STATUS"]
    WRITE_QUEUE_SIZE = simulation_params["WRITE_QUEUE_SIZE"]
    OUTPATH = simulation_params["OUTPATH"]
    OUTPUT_VARIABLES = simulation_params["OUTPUT_VARIABLES"]
    HOST_VDB_WRITER = simulation_params["HOST_VDB_WRITER"]
    BC_CONFIG = simulation_params["BC_CONFIG"]
    U_INFLOW = simulation_params["U_INFLOW"]
    V_INFLOW = simulation_params["V_INFLOW"]
    W_INFLOW = simulation_params["W_INFLOW"]
    obstacle_mask = simulation_params["obstacle_mask"]

    section_timings = {
        "initial_bcs": 0.0,
        "initial_obstacle": 0.0,
        "initial_dt": 0.0,
        "setup_output": 0.0,
        "buoyancy": 0.0,
        "pressure": 0.0,
        "velocity": 0.0,
        "scalars": 0.0,
        "bcs": 0.0,
        "obstacle": 0.0,
        "output": 0.0,
        "dt_update": 0.0,
    }
    total_start_time = perf_counter()

    #------------Initialise-------------------
    print('Initialise')
    print('Cell count: ', int(NX * NY * NZ))

    #------------Fields-------------------
    device_state, gpu_constants = Helper_Functions.upload_simulation_state_to_gpu(simulation_params)

    u = device_state["u"]
    v = device_state["v"]
    w = device_state["w"]
    u_work = device_state["u_work"]
    v_work = device_state["v_work"]
    w_work = device_state["w_work"]
    p = device_state["p"]
    pressure_work = device_state["pressure_work"]
    pressure_rhs = device_state["pressure_rhs"]
    T = device_state["T"]
    temperature_work = device_state["temperature_work"]
    smoke = device_state["smoke"]
    smoke_work = device_state["smoke_work"]
    fuel = device_state["fuel"]
    fuel_work = device_state["fuel_work"]
    flame = device_state["flame"]
    flame_work = device_state["flame_work"]
    Fx = device_state["Fx"]
    Fy = device_state["Fy"]
    Fz = device_state["Fz"]
    obstacle_mask = device_state["obstacle_mask"]
    source_mask = device_state["source_mask"]
    source_temperature = device_state["source_temperature"]
    source_smoke = device_state["source_smoke"]
    source_fuel = device_state["source_fuel"]
    device_fields = {
        "u": u,
        "v": v,
        "w": w,
        "p": p,
        "T": T,
        "smoke": smoke,
        "fuel": fuel,
        "flame": flame,
    }

    #------------BCs-------------------
    section_start = perf_counter()
    u, v, w, p, T = BC.apply_all_BC(u, v, w, p, T, BC_CONFIG, U_INFLOW, V_INFLOW, W_INFLOW)
    cuda.synchronize()
    section_timings["initial_bcs"] += perf_counter() - section_start

    #------------Source-------------------
    section_start = perf_counter()
    T, smoke, fuel = Source_BC.source_bc(
        T, smoke, fuel, source_mask, source_temperature, source_smoke, source_fuel
    )
    #------------Obstacle-------------------
    u, v, w, p, T, smoke, fuel, flame = Obstacle_BC.obstacle_bc(
        u, v, w, p, T, smoke, fuel, flame, obstacle_mask
    )
    cuda.synchronize()
    section_timings["initial_obstacle"] += perf_counter() - section_start

    host_output_fields = Helper_Functions.copy_device_fields_to_host(
        Helper_Functions.select_fields(device_fields, OUTPUT_VARIABLES)
    )

    #------------Estimate force maxima-------------------
    g = 9.81
    fx_max = 0.0
    fy_max = 0.0
    source_temperature_delta = max(
        0.0,
        float(gpu_constants["SOURCE_TEMPERATURE_MAX"] - gpu_constants["T_REFERENCE"]),
    )
    fz_max = g * gpu_constants["BUOANCY_FACTOR"] * source_temperature_delta * 1.5

    #------------Dynamic time step-------------------
    t = 0.0
    step_index = 0

    section_start = perf_counter()
    dt = Time_Step.compute_new_timestep_gpu(
        u, v, w, fx_max, fy_max, fz_max,
        gpu_constants["RHO"], gpu_constants["DELTA"], gpu_constants["NU"], CFL_MAX
    )
    section_timings["initial_dt"] += perf_counter() - section_start
    if dt > 1.0 / OUTPUT_FPS:
        dt = 1.0 / OUTPUT_FPS

    #------------Prepare Output-------------------
    next_output_time = 0.0
    output_index = 0
    
    section_start = perf_counter()
    write_queue, buffer_pool, writer_threads, shared_memory_blocks = Output_Functions.setup_output(
        OUTPATH,
        OUTPUT_VARIABLES,
        host_output_fields,
        WRITE_QUEUE_SIZE,
        DELTA,
        HOST_VDB_WRITER,
    )
    section_timings["setup_output"] += perf_counter() - section_start

    #------------Main time loop-------------------
    print('Start time iteration')
    sys.stdout.write('\rProgress: [0%]')

    while t < T_MAX:
        blockspergrid_3d = (
            (u.shape[0] + THREADS_PER_BLOCK_3D[0] - 1) // THREADS_PER_BLOCK_3D[0],
            (u.shape[1] + THREADS_PER_BLOCK_3D[1] - 1) // THREADS_PER_BLOCK_3D[1],
            (u.shape[2] + THREADS_PER_BLOCK_3D[2] - 1) // THREADS_PER_BLOCK_3D[2],
        )

        #------------Buoyancy-------------------
        section_start = perf_counter()
        buoyancy_approximation[blockspergrid_3d, THREADS_PER_BLOCK_3D](
            T, Fz, gpu_constants["BUOANCY_FACTOR"], gpu_constants["T_REFERENCE"]
        )
        cuda.synchronize()
        section_timings["buoyancy"] += perf_counter() - section_start

        #------------Pressure-------------------
        section_start = perf_counter()
        p = pressure_poisson(
            u, v, w, p, T, pressure_work, pressure_rhs, dt, Fx, Fy, Fz,
            gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["EXPANSION_RATE"],
            gpu_constants["T_REFERENCE"], MAX_ITER
        )
        cuda.synchronize()
        section_timings["pressure"] += perf_counter() - section_start

        #------------Velocity-------------------
        section_start = perf_counter()
        update_velocity[blockspergrid_3d, THREADS_PER_BLOCK_3D](
            u, v, w, p, dt, Fx, Fy, Fz, u_work, v_work, w_work,
            gpu_constants["DELTA"], gpu_constants["RHO"], gpu_constants["NU"]
        )
        cuda.synchronize()
        section_timings["velocity"] += perf_counter() - section_start

        u, u_work = u_work, u
        v, v_work = v_work, v
        w, w_work = w_work, w

        #------------Scalars-------------------
        section_start = perf_counter()
        update_scalar_fields[blockspergrid_3d, THREADS_PER_BLOCK_3D](
            T, smoke, fuel, u, v, w, dt,
            temperature_work, smoke_work, fuel_work, flame_work,
            gpu_constants["DELTA"],
            gpu_constants["NU_TEMPERATURE"],
            gpu_constants["NU_SMOKE"],
            gpu_constants["NU_FUEL"],
            gpu_constants["TEMPERATURE_DISSIPATION_RATE"],
            gpu_constants["TEMPERATURE_PRODUCTION_RATE"],
            gpu_constants["SMOKE_DISSIPATION_RATE"],
            gpu_constants["SMOKE_PRODUCTION_RATE"],
            gpu_constants["FUEL_BURN_RATE"],
            gpu_constants["FUEL_IGNITION_TEMPERATURE"],
            gpu_constants["T_REFERENCE"],
        )
        cuda.synchronize()
        section_timings["scalars"] += perf_counter() - section_start

        #------------Swap-------------------
        T, temperature_work = temperature_work, T
        smoke, smoke_work = smoke_work, smoke
        fuel, fuel_work = fuel_work, fuel
        flame, flame_work = flame_work, flame

        #------------BCs-------------------
        section_start = perf_counter()
        u, v, w, p, T = BC.apply_all_BC(
            u, v, w, p, T, BC_CONFIG, U_INFLOW, V_INFLOW, W_INFLOW
        )
        cuda.synchronize()
        section_timings["bcs"] += perf_counter() - section_start

        #------------Source-------------------
        section_start = perf_counter()
        T, smoke, fuel = Source_BC.source_bc(
            T, smoke, fuel, source_mask, source_temperature, source_smoke, source_fuel
        )
        #------------Obstacle-------------------
        u, v, w, p, T, smoke, fuel, flame = Obstacle_BC.obstacle_bc(
            u, v, w, p, T, smoke, fuel, flame, obstacle_mask
        )
        cuda.synchronize()
        section_timings["obstacle"] += perf_counter() - section_start
        device_fields["u"] = u
        device_fields["v"] = v
        device_fields["w"] = w
        device_fields["p"] = p
        device_fields["T"] = T
        device_fields["smoke"] = smoke
        device_fields["fuel"] = fuel
        device_fields["flame"] = flame

        #------------Output-------------------
        while t >= next_output_time:
            section_start = perf_counter()
            host_output_fields = Helper_Functions.copy_device_fields_to_host(
                Helper_Functions.select_fields(device_fields, OUTPUT_VARIABLES)
            )
            Output_Functions.enqueue_output(write_queue, buffer_pool, OUTPUT_VARIABLES, host_output_fields, output_index, t)
            section_timings["output"] += perf_counter() - section_start

            output_index += 1
            next_output_time += OUTPUT_TIME_STEP
            sys.stdout.write(f'\rProgress: [{(t / T_MAX * 100):.3f}%]')
            sys.stdout.flush()

            if OUTPUT_STATUS:
                print('#################################################')
                print(f'Simulation time {t} sec')
                print(f'Current dt: {np.round(dt, 5)}')

        #------------Dynamic time step-------------------
        t += dt
        step_index += 1

        section_start = perf_counter()
        dt_new = Time_Step.compute_new_timestep_gpu(
            u, v, w, fx_max, fy_max, fz_max,
            gpu_constants["RHO"], gpu_constants["DELTA"], gpu_constants["NU"], CFL_MAX
        )
        section_timings["dt_update"] += perf_counter() - section_start

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
    Output_Functions.shutdown_output(write_queue, writer_threads, shared_memory_blocks)

    #------------Conclusion-------------------
    print('Simulation finished!')
    total_runtime = perf_counter() - total_start_time
    print('Timing summary:')
    for name, elapsed in section_timings.items():
        share = 100.0 * elapsed / total_runtime if total_runtime > 0.0 else 0.0
        print(f'  {name}: {elapsed:.3f} s ({share:.1f}%)')
    print(f'  total: {total_runtime:.3f} s')
