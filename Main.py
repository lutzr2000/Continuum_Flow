"""
Description:
    A finite difference based CFD solver for Blender. Convection is discretized by first order upwind,
    diffusion and pressure by central differencing. The pressure velocity coupling is done by solving
    a poisson equation.

    This version extends the original 2D solver to a 3D Cartesian grid.
"""
# TODO Export to VDB
# TODO add forces, especially a random one

import numpy as np
import time
import sys
import os
import threading
import queue

from numba import njit, prange, set_num_threads, get_num_threads

import Boundary_Conditions as BC
import Obstacle_Boundary_Conditions as Obstacle_BC
import Obstacles
import Output_Functions
import Helper_Functions

# ===============================
# Parameters
# ===============================

# fluid
RHO = 1.225
NU = 1.81e-5
NU_TEMPERATURE = 0.001
NU_SMOKE = 0.001
NU_FUEL = 0.001
TEMPERATURE_DISSIPATION_RATE = 0.1
TEMPERATURE_PRODUCTION_RATE = 1.0
SMOKE_DISSIPATION_RATE = 0.1
SMOKE_PRODUCTION_RATE = 1.0
FUEL_BURN_RATE = 0.1
FUEL_IGNITION_TEMPERATURE = 100.0
T_REFERENCE = 300.0
BUOANCY_FACTOR = 1/T_REFERENCE
EXPANSION_RATE = 0.003

# time
T_MAX = 30.0
CFL_MAX = 0.8

# solver
MAX_ITER = 4
PRECISION = np.float32
CPU_PARALLEL = True
RESERVE_CPU_CORES_FOR_IO = 1

# resolution
DELTA = 0.2 
NX = 64 
NY = 64 
NZ = 64 
x = np.linspace(0.0, (NX - 1) * DELTA, NX, dtype=PRECISION)
y = np.linspace(0.0, (NY - 1) * DELTA, NY, dtype=PRECISION)
z = np.linspace(0.0, (NZ - 1) * DELTA, NZ, dtype=PRECISION)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# output
OUTPUT_FPS = 24
PRINT_FREQUENCY = 100
OUTPUT_TIME_STEP = 1.0 / OUTPUT_FPS
OUTPATH = r"C:\Blenderzeug\BlenderCFD\Test\Test.nc"
OUTPUT_STATUS = False
WRITE_QUEUE_SIZE = 512
NETCDF_COMPRESSION_LEVEL = 0

# Boundary conditions
U_INFLOW = 0.0
V_INFLOW = 0.0
W_INFLOW = 0.0

# Geometry
obstacle_mask = Obstacles.sphere(X, Y, Z, NX * 0.5 * DELTA, NY * 0.5 * DELTA, NX * 0.05 * DELTA, 0.6)

# ===============================
# Methods
# ===============================

@njit(parallel=CPU_PARALLEL)
def buoyancy_approximation(T, Fz, expansion_coefficient, T_ref):
    """
    computes the buoyancy force in z-direction with the Boussinesq approximation.

    Args:
        T (3d-array): temperature field
        Fy (3d-array): y-direction force field
        expansion_coefficient (float): thermal expansion coefficient
        T_ref (float): reference temperature
    Returns:
        Fy (3d-array): updated y-direction force field
    """
    nx, ny, nz = T.shape
    g = 9.81
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                Fz[i, j, k] = g * expansion_coefficient * (T[i, j, k] - T_ref)

    return Fz


@njit(parallel=CPU_PARALLEL)
def update_scalar_fields(T, smoke, fuel, u, v, w, dt, T_out, smoke_out, fuel_out):
    """
    Updates temperature, smoke and fuel with convection, diffusion and source terms in one transport sweep.
    Convection is done by first order upwind, diffusion with central differences.

    Args:
        T (3d-array): temperature field
        smoke (3d-array): smoke density field
        fuel (3d-array): fuel density field
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        dt (float): timestep size
        T_out (3d-array): output array for updated temperature
        smoke_out (3d-array): output array for updated smoke
        fuel_out (3d-array): output array for updated fuel
    Returns:
        T_out (3d-array): updated temperature field
        smoke_out (3d-array): updated smoke field
        fuel_out (3d-array): updated fuel field
    """
    nx, ny, nz = u.shape

    dt_over_delta = dt / DELTA
    dt_over_delta2 = dt / (DELTA * DELTA)
    temp_diffusion_coeff = NU_TEMPERATURE * dt_over_delta2
    smoke_diffusion_coeff = NU_SMOKE * dt_over_delta2
    fuel_diffusion_coeff = NU_FUEL * dt_over_delta2

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                uijk = u[i, j, k]
                vijk = v[i, j, k]
                wijk = w[i, j, k]
                #------------Upwinding-------------------
                if uijk >= 0.0:
                    t_x_high = T[i, j, k]
                    t_x_low = T[i - 1, j, k]
                    smoke_x_high = smoke[i, j, k]
                    smoke_x_low = smoke[i - 1, j, k]
                    fuel_x_high = fuel[i, j, k]
                    fuel_x_low = fuel[i - 1, j, k]
                else:
                    t_x_high = T[i + 1, j, k]
                    t_x_low = T[i, j, k]
                    smoke_x_high = smoke[i + 1, j, k]
                    smoke_x_low = smoke[i, j, k]
                    fuel_x_high = fuel[i + 1, j, k]
                    fuel_x_low = fuel[i, j, k]

                if vijk >= 0.0:
                    t_y_high = T[i, j, k]
                    t_y_low = T[i, j - 1, k]
                    smoke_y_high = smoke[i, j, k]
                    smoke_y_low = smoke[i, j - 1, k]
                    fuel_y_high = fuel[i, j, k]
                    fuel_y_low = fuel[i, j - 1, k]
                else:
                    t_y_high = T[i, j + 1, k]
                    t_y_low = T[i, j, k]
                    smoke_y_high = smoke[i, j + 1, k]
                    smoke_y_low = smoke[i, j, k]
                    fuel_y_high = fuel[i, j + 1, k]
                    fuel_y_low = fuel[i, j, k]

                if wijk >= 0.0:
                    t_z_high = T[i, j, k]
                    t_z_low = T[i, j, k - 1]
                    smoke_z_high = smoke[i, j, k]
                    smoke_z_low = smoke[i, j, k - 1]
                    fuel_z_high = fuel[i, j, k]
                    fuel_z_low = fuel[i, j, k - 1]
                else:
                    t_z_high = T[i, j, k + 1]
                    t_z_low = T[i, j, k]
                    smoke_z_high = smoke[i, j, k + 1]
                    smoke_z_low = smoke[i, j, k]
                    fuel_z_high = fuel[i, j, k + 1]
                    fuel_z_low = fuel[i, j, k]

                #------------Convection-------------------
                T_center = T[i, j, k]
                smoke_center = smoke[i, j, k]
                fuel_center = fuel[i, j, k]

                temp_convection = dt_over_delta * (
                    uijk * (t_x_high - t_x_low) +
                    vijk * (t_y_high - t_y_low) +
                    wijk * (t_z_high - t_z_low)
                )
                smoke_convection = dt_over_delta * (
                    uijk * (smoke_x_high - smoke_x_low) +
                    vijk * (smoke_y_high - smoke_y_low) +
                    wijk * (smoke_z_high - smoke_z_low)
                )
                fuel_convection = dt_over_delta * (
                    uijk * (fuel_x_high - fuel_x_low) +
                    vijk * (fuel_y_high - fuel_y_low) +
                    wijk * (fuel_z_high - fuel_z_low)
                )

                #------------Diffusion-------------------
                temp_diffusion = temp_diffusion_coeff * (
                    (T[i + 1, j, k] - 2.0 * T_center + T[i - 1, j, k]) +
                    (T[i, j + 1, k] - 2.0 * T_center + T[i, j - 1, k]) +
                    (T[i, j, k + 1] - 2.0 * T_center + T[i, j, k - 1])
                )
                smoke_diffusion = smoke_diffusion_coeff * (
                    (smoke[i + 1, j, k] - 2.0 * smoke_center + smoke[i - 1, j, k]) +
                    (smoke[i, j + 1, k] - 2.0 * smoke_center + smoke[i, j - 1, k]) +
                    (smoke[i, j, k + 1] - 2.0 * smoke_center + smoke[i, j, k - 1])
                )
                fuel_diffusion = fuel_diffusion_coeff * (
                    (fuel[i + 1, j, k] - 2.0 * fuel_center + fuel[i - 1, j, k]) +
                    (fuel[i, j + 1, k] - 2.0 * fuel_center + fuel[i, j - 1, k]) +
                    (fuel[i, j, k + 1] - 2.0 * fuel_center + fuel[i, j, k - 1])
                )

                #------------Ignition of fuel-------------------
                if T_center > FUEL_IGNITION_TEMPERATURE:
                    fuel_source = -FUEL_BURN_RATE * fuel_center
                else:
                    fuel_source = 0.0

                #------------Burning fuel creates temperature-------------------
                temperature_source = (
                    -TEMPERATURE_DISSIPATION_RATE * (T_center - T_REFERENCE) +
                    TEMPERATURE_PRODUCTION_RATE * (-fuel_source)
                )
                #------------Burning fuel creates smoke-------------------
                smoke_source = SMOKE_PRODUCTION_RATE * (-fuel_source) - SMOKE_DISSIPATION_RATE * smoke_center

                #------------Update-------------------
                T_out[i, j, k] = T_center - temp_convection + temp_diffusion + dt * temperature_source
                smoke_out[i, j, k] = smoke_center - smoke_convection + smoke_diffusion + dt * smoke_source
                fuel_out[i, j, k] = fuel_center - fuel_convection + fuel_diffusion + dt * fuel_source

    return T_out, smoke_out, fuel_out


@njit(parallel=CPU_PARALLEL)
def update_x_velocity(u, v, w, p, dt, Fx, un):
    """
    Updates the velocity field in x-direction based on the momentum equation. 
    Convection is done by first order upwind, diffusion with central differences.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        p (3d-array): pressure field
        dt (float): timestep size
        Fx (3d-array): x-direction body force field
        un (3d-array): output array for updated x-velocity
    Returns:
        un (3d-array): new x-velocity field
    """
    nx, ny, nz = u.shape
    dt_over_delta = dt / DELTA
    pressure_coeff = dt / (2.0 * RHO * DELTA)
    diffusion_coeff = NU * dt / (DELTA * DELTA)
    force_coeff = dt / RHO

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                #------------Upwinding-------------------
                if u[i, j, k] >= 0.0:
                    u_x_high = u[i, j, k]
                    u_x_low = u[i - 1, j, k]
                else:
                    u_x_high = u[i + 1, j, k]
                    u_x_low = u[i, j, k]

                if v[i, j, k] >= 0.0:
                    u_y_high = u[i, j, k]
                    u_y_low = u[i, j - 1, k]
                else:
                    u_y_high = u[i, j + 1, k]
                    u_y_low = u[i, j, k]

                if w[i, j, k] >= 0.0:
                    u_z_high = u[i, j, k]
                    u_z_low = u[i, j, k - 1]
                else:
                    u_z_high = u[i, j, k + 1]
                    u_z_low = u[i, j, k]

                #------------Convection-------------------
                u_center = u[i, j, k]
                convection = dt_over_delta * (
                    u_center * (u_x_high - u_x_low) +
                    v[i, j, k] * (u_y_high - u_y_low) +
                    w[i, j, k] * (u_z_high - u_z_low)
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

    return un


@njit(parallel=CPU_PARALLEL)
def update_y_velocity(u, v, w, p, dt, Fy, vn):
    """
    Updates the velocity field in y-direction based on the momentum equation.
    Convection is done by first order upwind, diffusion with central differences.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        p (3d-array): pressure field
        dt (float): timestep size
        Fy (3d-array): y-direction body force field
        vn (3d-array): output array for updated y-velocity
    Returns:
        vn (3d-array): new y-velocity field
    """
    nx, ny, nz = v.shape
    dt_over_delta = dt / DELTA
    pressure_coeff = dt / (2.0 * RHO * DELTA)
    diffusion_coeff = NU * dt / (DELTA * DELTA)
    force_coeff = dt / RHO

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                #------------Upwinding-------------------
                if u[i, j, k] >= 0.0:
                    v_x_high = v[i, j, k]
                    v_x_low = v[i - 1, j, k]
                else:
                    v_x_high = v[i + 1, j, k]
                    v_x_low = v[i, j, k]

                if v[i, j, k] >= 0.0:
                    v_y_high = v[i, j, k]
                    v_y_low = v[i, j - 1, k]
                else:
                    v_y_high = v[i, j + 1, k]
                    v_y_low = v[i, j, k]

                if w[i, j, k] >= 0.0:
                    v_z_high = v[i, j, k]
                    v_z_low = v[i, j, k - 1]
                else:
                    v_z_high = v[i, j, k + 1]
                    v_z_low = v[i, j, k]

                #------------Convection-------------------
                v_center = v[i, j, k]
                convection = dt_over_delta * (
                    u[i, j, k] * (v_x_high - v_x_low) +
                    v_center * (v_y_high - v_y_low) +
                    w[i, j, k] * (v_z_high - v_z_low)
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

    return vn


@njit(parallel=CPU_PARALLEL)
def update_z_velocity(u, v, w, p, dt, Fz, wn):
    """
    Updates the velocity field in z-direction based on the momentum equation. 
    Convection is done by first order upwind, diffusion with central differences.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        p (3d-array): pressure field
        dt (float): timestep size
        Fz (3d-array): z-direction body force field
        wn (3d-array): output array for updated z-velocity
    Returns:
        wn (3d-array): new z-velocity field
    """
    nx, ny, nz = w.shape
    dt_over_delta = dt / DELTA
    pressure_coeff = dt / (2.0 * RHO * DELTA)
    diffusion_coeff = NU * dt / (DELTA * DELTA)
    force_coeff = dt / RHO

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                #------------Upwinding-------------------
                if u[i, j, k] >= 0.0:
                    w_x_high = w[i, j, k]
                    w_x_low = w[i - 1, j, k]
                else:
                    w_x_high = w[i + 1, j, k]
                    w_x_low = w[i, j, k]

                if v[i, j, k] >= 0.0:
                    w_y_high = w[i, j, k]
                    w_y_low = w[i, j - 1, k]
                else:
                    w_y_high = w[i, j + 1, k]
                    w_y_low = w[i, j, k]

                if w[i, j, k] >= 0.0:
                    w_z_high = w[i, j, k]
                    w_z_low = w[i, j, k - 1]
                else:
                    w_z_high = w[i, j, k + 1]
                    w_z_low = w[i, j, k]

                #------------Convection-------------------
                w_center = w[i, j, k]
                convection = dt_over_delta * (
                    u[i, j, k] * (w_x_high - w_x_low) +
                    v[i, j, k] * (w_y_high - w_y_low) +
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

    return wn

@njit(parallel=CPU_PARALLEL)
def pressure_equation_right_side(u, v, w, T, b, dt, Fx, Fy, Fz):
    """
    computes the right hand side of the pressure Poisson equation.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        T (3d-array): temperature field
        b (3d-array): output array for the pressure equation right hand side
        dt (float): timestep size
        Fx (3d-array): x-direction body force field
        Fy (3d-array): y-direction body force field
        Fz (3d-array): z-direction body force field
    Returns:
        b (3d-array): right hand side of the pressure Poisson equation
    """
    nx, ny, nz = u.shape
    half_inv_delta = 0.5 / DELTA
    rho_over_dt = RHO / dt

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
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

                # This comes from the divergnce of the time derivative
                divergence = du_dx + dv_dy + dw_dz
                # This comes from the divergence of the convection term
                nonlinear = (
                    du_dx * du_dx +
                    dv_dy * dv_dy +
                    dw_dz * dw_dz +
                    2.0 * (du_dy * dv_dx + du_dz * dw_dx + dv_dz * dw_dy)
                )

                #------------Artifical thermal divergence-------------------
                # This is very inaccurate to real physics, but very siple
                dFx_dx = (Fx[i + 1, j, k] - Fx[i - 1, j, k]) * half_inv_delta
                dFy_dy = (Fy[i, j + 1, k] - Fy[i, j - 1, k]) * half_inv_delta
                dFz_dz = (Fz[i, j, k + 1] - Fz[i, j, k - 1]) * half_inv_delta
                thermal_divergence = EXPANSION_RATE * (T[i, j, k] - T_REFERENCE)

                #------------Right hand side-------------------
                b[i, j, k] = rho_over_dt * (divergence - thermal_divergence) - RHO * (
                    nonlinear + dFx_dx + dFy_dy + dFz_dz
                )

    return b


@njit(parallel=CPU_PARALLEL)
def pressure_poisson(u, v, w, p, T, p_work, b, dt, Fx, Fy, Fz, max_iter=10):
    """
    solves the 3D pressure Poisson equation with Jacobi iterations.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        p (3d-array): pressure field
        T (3d-array): temperature field
        p_work (3d-array): work array for the pressure iteration
        b (3d-array): work array for the pressure equation right hand side
        dt (float): timestep size
        Fx (3d-array): x-direction body force field
        Fy (3d-array): y-direction body force field
        Fz (3d-array): z-direction body force field
        max_iter (int): number of Jacobi iterations
    Returns:
        p_old (3d-array): updated pressure field
    """
    nx, ny, nz = p.shape
    delta2 = DELTA * DELTA
    b = pressure_equation_right_side(u, v, w, T, b, dt, Fx, Fy, Fz)
    p_old = p
    p_new = p_work

    for _ in range(max_iter):
        #------------Laplace part-------------------
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    p_new[i, j, k] = (
                        p_old[i + 1, j, k] + p_old[i - 1, j, k] +
                        p_old[i, j + 1, k] + p_old[i, j - 1, k] +
                        p_old[i, j, k + 1] + p_old[i, j, k - 1] -
                        delta2 * b[i, j, k]
                    ) / 6.0

        #------------BCs-------------------
        # pressure has always Neumann BCs
        for j in range(ny):
            for k in range(nz):
                p_new[0, j, k] = p_new[1, j, k]
                p_new[nx - 1, j, k] = p_new[nx - 2, j, k]

        for i in range(nx):
            for k in range(nz):
                p_new[i, 0, k] = p_new[i, 1, k]
                p_new[i, ny - 1, k] = p_new[i, ny - 2, k]

        for i in range(nx):
            for j in range(ny):
                p_new[i, j, 0] = p_new[i, j, 1]
                p_new[i, j, nz - 1] = p_new[i, j, nz - 2]

        #------------Obstacle-------------------
        p_new = Obstacle_BC.obstacle_boundary_conditions_pressure(p_new, obstacle_mask)
        #------------Swap-------------------
        p_old, p_new = p_new, p_old

    return p_old

# ===============================
# Main
# ===============================

def main():
    #------------Initialise-------------------
    print('Initialise')
    print('Cell count: ', int(NX * NY * NZ))

    if CPU_PARALLEL:
        available_threads = os.cpu_count() or 1
        solver_threads = max(1, available_threads - RESERVE_CPU_CORES_FOR_IO)
        set_num_threads(solver_threads)
        print(f'Numba solver threads: {get_num_threads()} / {available_threads} Cores')

    #------------Fields-------------------
    u = np.full((NX, NY, NZ), U_INFLOW, dtype=PRECISION)
    v = np.full((NX, NY, NZ), V_INFLOW, dtype=PRECISION)
    w = np.full((NX, NY, NZ), W_INFLOW, dtype=PRECISION)
    un = np.empty_like(u)
    vn = np.empty_like(v)
    wn = np.empty_like(w)
    u_work = np.empty_like(u)
    v_work = np.empty_like(v)
    w_work = np.empty_like(w)
    np.copyto(un, u)
    np.copyto(vn, v)
    np.copyto(wn, w)

    p = np.zeros((NX, NY, NZ), dtype=PRECISION)
    pressure_work = np.empty_like(p)
    pressure_rhs = np.empty_like(p)

    T = np.full((NX, NY, NZ), T_REFERENCE, dtype=PRECISION)
    temperature_work = np.empty_like(T)

    smoke = np.zeros((NX, NY, NZ), dtype=PRECISION)
    smoke_work = np.empty_like(smoke)

    fuel = np.zeros((NX, NY, NZ), dtype=PRECISION)
    fuel_work = np.empty_like(fuel)

    Fx = np.zeros_like(p)
    Fy = np.zeros_like(p)
    Fz = np.zeros_like(p)

    timestep_plane_max_u = np.empty(NX, dtype=PRECISION)
    timestep_plane_max_v = np.empty(NX, dtype=PRECISION)
    timestep_plane_max_w = np.empty(NX, dtype=PRECISION)
    timestep_plane_max_F = np.empty(NX, dtype=PRECISION)

    #------------BCs-------------------
    u, v, w, p, T = BC.outflow_BC(u, v, w, p, T, 'x_low')
    u, v, w, p, T = BC.outflow_BC(u, v, w, p, T, 'x_high')
    u, v, w, p, T = BC.outflow_BC(u, v, w, p, T, 'y_low')
    u, v, w, p, T = BC.outflow_BC(u, v, w, p, T, 'y_high')
    u, v, w, p, T = BC.no_slip_wall_BC(u, v, w, p, T, 'z_low')
    u, v, w, p, T = BC.no_slip_wall_BC(u, v, w, p, T, 'z_high')

    #------------Obstacle-------------------
    u, v, w = Obstacle_BC.obstacle_boundary_conditions_velocity(u, v, w, obstacle_mask)
    p = Obstacle_BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)
    T = Obstacle_BC.obstacle_boundary_conditions_scalar(T, obstacle_mask, 600.0)
    fuel = Obstacle_BC.obstacle_boundary_conditions_scalar(fuel, obstacle_mask, 1.0)

    start_total_time = time.time()
    total_loop_time = 0.0
    time_pressure_update = 0.0
    time_velocity_update = 0.0
    time_scalar_update = 0.0
    time_BC = 0.0
    time_obstacle = 0.0
    time_start_copy = 0.0
    time_compute_step = 0.0

    #------------Dynamic time step-------------------
    t = 0.0
    dt = Helper_Functions.compute_new_timestep(
        u, v, w, Fz, RHO, DELTA, NU, CFL_MAX,
        timestep_plane_max_u, timestep_plane_max_v, timestep_plane_max_w, timestep_plane_max_F
    )
    if dt > 1.0 / OUTPUT_FPS:
        dt = 1.0 / OUTPUT_FPS

    #------------Prepare Output-------------------
    next_output_time = 0.0
    output_index = 0

    dataset, u_var, v_var, w_var, p_var, T_var, smoke_var, fuel_var, time_var = Output_Functions.initialize_netcdf(
        OUTPATH, NX, NY, NZ, x, y, z, comp_level=NETCDF_COMPRESSION_LEVEL
    )

    write_queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
    buffer_pool = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
    max_write_queue_fill = 0

    for _ in range(WRITE_QUEUE_SIZE):
        buffer_pool.put({
            'u': np.empty_like(u),
            'v': np.empty_like(v),
            'w': np.empty_like(w),
            'p': np.empty_like(p),
            'T': np.empty_like(T),
            'smoke': np.empty_like(smoke),
            'fuel': np.empty_like(fuel),
        })

    def writer_thread_func():
        while True:
            item = write_queue.get()
            if item is None:
                write_queue.task_done()
                break

            output_idx, time_value, fields = item
            Output_Functions.write_to_netcdf(
                u_var, v_var, w_var, p_var, T_var, smoke_var, fuel_var, time_var,
                output_idx, time_value,
                fields['u'], fields['v'], fields['w'], fields['p'], fields['T'], fields['smoke'], fields['fuel']
            )
            buffer_pool.put(fields)
            write_queue.task_done()

    writer_thread = threading.Thread(target=writer_thread_func, daemon=True)
    writer_thread.start()

    #------------Main time loop-------------------
    print('Start time iteration')
    sys.stdout.write('\rProgress: [0%]')

    while t < T_MAX:
        #------------Start-------------------
        loop_start_time = time.time()

        t0 = time.perf_counter()
        np.copyto(un, u)
        np.copyto(vn, v)
        np.copyto(wn, w)
        t1 = time.perf_counter()
        time_start_copy += (t1 - t0)

        #------------Buoancy-------------------
        Fz = buoyancy_approximation(T, Fz, BUOANCY_FACTOR, T_REFERENCE)

        #------------Pressure-------------------
        t0 = time.perf_counter()
        p = pressure_poisson(un, vn, wn, p, T, pressure_work, pressure_rhs, dt, Fx, Fy, Fz, MAX_ITER)
        t1 = time.perf_counter()
        time_pressure_update += (t1 - t0)

        #------------Velocity-------------------
        t0 = time.perf_counter()
        u = update_x_velocity(un, vn, wn, p, dt, Fx, u_work)
        v = update_y_velocity(un, vn, wn, p, dt, Fy, v_work)
        w = update_z_velocity(un, vn, wn, p, dt, Fz, w_work)
        t1 = time.perf_counter()
        time_velocity_update += (t1 - t0)

        #------------Smoke, Fuel and Temperature-------------------
        t0 = time.perf_counter()
        T, smoke, fuel = update_scalar_fields(T, smoke, fuel, u, v, w, dt, temperature_work, smoke_work, fuel_work)
        t1 = time.perf_counter()
        time_scalar_update += (t1 - t0)

        #------------BCs-------------------
        t0 = time.perf_counter()
        u, v, w, p, T = BC.outflow_BC(u, v, w, p, T, 'x_low')
        u, v, w, p, T = BC.outflow_BC(u, v, w, p, T, 'x_high')
        u, v, w, p, T = BC.outflow_BC(u, v, w, p, T, 'y_low')
        u, v, w, p, T = BC.outflow_BC(u, v, w, p, T, 'y_high')
        u, v, w, p, T = BC.no_slip_wall_BC(u, v, w, p, T, 'z_low')
        u, v, w, p, T = BC.no_slip_wall_BC(u, v, w, p, T, 'z_high')
        t1 = time.perf_counter()
        time_BC += (t1 - t0)

        #------------Obstacle-------------------
        t0 = time.perf_counter()
        u, v, w = Obstacle_BC.obstacle_boundary_conditions_velocity(u, v, w, obstacle_mask)
        p = Obstacle_BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)
        T = Obstacle_BC.obstacle_boundary_conditions_scalar(T, obstacle_mask, 600.0)
        fuel = Obstacle_BC.obstacle_boundary_conditions_scalar(fuel, obstacle_mask, 1.0)
        t1 = time.perf_counter()
        time_obstacle += (t1 - t0)

        #------------Output-------------------
        while t >= next_output_time:
            fields = buffer_pool.get()
            np.copyto(fields['u'], u)
            np.copyto(fields['v'], v)
            np.copyto(fields['w'], w)
            np.copyto(fields['p'], p)
            np.copyto(fields['T'], T)
            np.copyto(fields['smoke'], smoke)
            np.copyto(fields['fuel'], fuel)
            write_queue.put((output_index, t, fields))
            max_write_queue_fill = max(max_write_queue_fill, write_queue.qsize())

            output_index += 1
            next_output_time += OUTPUT_TIME_STEP
            sys.stdout.write(f'\rProgress: [{(t / T_MAX * 100):.3f}%]')
            sys.stdout.flush()

            if OUTPUT_STATUS:
                print('#################################################')
                print(f'Simulation time {t} sec')
                CFL = Helper_Functions.compute_CFL(u, v, w, dt, DELTA)
                print(f'Current dt: {np.round(dt, 5)}')
                print(f'CFL-Condition: {np.round(CFL, 5)}')

        #------------Dynamic time step-------------------
        t0 = time.perf_counter()
        t += dt

        dt_new = Helper_Functions.compute_new_timestep(
            u, v, w, Fz, RHO, DELTA, NU, CFL_MAX,
            timestep_plane_max_u, timestep_plane_max_v, timestep_plane_max_w, timestep_plane_max_F
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

        t1 = time.perf_counter()
        time_compute_step += (t1 - t0)

        #------------End-------------------
        loop_end_time = time.time()
        total_loop_time += loop_end_time - loop_start_time

    #------------Empty write queue-------------------
    write_queue.join()
    write_queue.put(None)
    write_queue.join()
    writer_thread.join()

    Output_Functions.close_netcdf(dataset)
    end_total_time = time.time()

    #------------Conclusion-------------------
    print('Simulation finished!')
    print(f'Total runtime: {end_total_time - start_total_time:.4f} seconds')
    print(f'Total time spent in main loop: {total_loop_time:.4f} seconds')
    print(f'Time spend on array copying: {time_start_copy:.4f} seconds')
    print(f'Time spend on pressure: {time_pressure_update:.4f} seconds')
    print(f'Time spend on velocity solve: {time_velocity_update:.4f} seconds')
    print(f'Time spend on scalar update: {time_scalar_update:.4f} seconds')
    print(f'Time spend on boundary conditions: {time_BC:.4f} seconds')
    print(f'Time spend on obstacles: {time_obstacle:.4f} seconds')
    print(f'Time spend on computing next time step: {time_compute_step:.4f} seconds')
    print(f'Max async write queue fill: {max_write_queue_fill}/{WRITE_QUEUE_SIZE}')


main()
