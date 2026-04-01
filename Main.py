"""
Description:
    A finite difference based CFD solver for Blender. Convetion is discretized by first order upwind,
    diffusion and pressure by central differencing. The pressure velocity coupling is done by solving
    a poisson equation.

    This code is based on the methods presented here:
    https://drzgan.github.io/Python_CFD/15.1.%20Cavity%20flow%20with%20upwind%20scheme.html
    Huge thanks to Dr. Zhengtao Gan.
"""

# ===============================
# Import
# ===============================

import numpy as np
import time
import sys

from numba import njit, prange

import Boundary_Conditions as BC
import Obstacles 
import Output_Functions
import Helper_Functions

# ===============================
# Parameters
# ===============================

# TODO Time the code
# TODO Test tiling for better parallel
# TODO In general solver only profits very little from parallel CPU why? 

# fluid
RHO = 1.225
NU = 1.81e-5
NU_TEMPERATURE = 0.001

# time
T_MAX = 20
CFL_MAX = 0.8

# solver 
TOLERANCE = 0.01
MAX_ITER = 4
PRECISION = np.float32
CPU_PARALLEL = False

# resolution
DELTA = 0.04
NX = 1024
NY = 128
x = np.linspace(0,(NX-1)*DELTA,NX)
y = np.linspace(0,(NY-1)*DELTA,NY)
X, Y = np.meshgrid(x, y)

# output
OUTPUT_FPS = 24
PRINT_FREQUENCY = 100
OUTPUT_TIME_STEP = 1/OUTPUT_FPS
OUTPATH = rf"C:\Blenderzeug\BlenderCFD\Test\Test.nc"
OUTPUT_STATUS = True

# initial conditions
u_initial = np.ones_like(X).astype(PRECISION)*5
v_initial = np.zeros_like(X).astype(PRECISION)
p_initial = np.zeros_like(X).astype(PRECISION)
T_initial = np.ones_like(X).astype(PRECISION)*300

# Geometry
circle1_mask = Obstacles.circle(X,Y,NX*0.2*DELTA,NY*0.5*DELTA,0.5)
circle2_mask = Obstacles.circle(X,Y,NX*0.25*DELTA,NY*0.3*DELTA,0.6)
circle3_mask = Obstacles.circle(X,Y,NX*0.4*DELTA,NY*0.8*DELTA,0.3)

obstacle_mask = circle1_mask | circle2_mask | circle3_mask

# ===============================
# Functions
# ===============================

@njit(parallel=CPU_PARALLEL)
def general_scalar_transport_equation(phi,u,v,dt,nu,Source=None):
    """
    Updates a scalar field phi based on a general transport equation. Discretization with first order upwind
    for the convection term. Central differences for the Diffusion term. Source term is optional

    Args:
        phi (2d-array): scalar field
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        dt (float): time step
        nu (float): viscosity of phi
        Source (2d-array,optional): phi source field
    Returns:
        phin (2d-array): new phi field
    """
    Nx, Ny = u.shape
    phin = phi.copy()

    dt_over_DELTA = dt / DELTA
    dt_over_DELTA2 = dt / (DELTA**2)

    # Loop over interior points, parallel over rows
    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            # Convection in x-direction (first-order upwind)
            if u[i, j] >= 0:
                phi_east = phi[i, j]
                phi_west = phi[i, j-1]
            else:
                phi_east = phi[i, j+1]
                phi_west = phi[i, j]

            # Convection in y-direction (first-order upwind)
            if v[i, j] >= 0:
                phi_north = phi[i, j]
                phi_south = phi[i-1, j]
            else:
                phi_north = phi[i+1, j]
                phi_south = phi[i, j]

            convection = dt_over_DELTA * (u[i, j] * (phi_east - phi_west) + v[i, j] * (phi_north - phi_south))

            # Diffusion (central difference)
            diffusion = nu * dt_over_DELTA2 * (
                (phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]) +
                (phi[i+1, j] - 2*phi[i, j] + phi[i-1, j])
            )

            # Source term
            source = dt* Source[i, j] if Source is not None else 0.0

            # Update velocity
            phin[i, j] = phi[i, j] - convection + diffusion + source

    return phin


@njit(parallel=CPU_PARALLEL)
def update_x_velocity(u, v, p, dt, Fx=None):
    """
    Updates the velocity field in the x direction based on the momentum equation. Discretization with first order upwind
    for the convection term. Central differences for the Diffusion term. Forcing source term is optional

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): p-pressure field
        dt (float): time step
        Fx (2d-array,optional): x-Forcing field
    Returns:
        un (2d-array): new u-velocity field
    """
    Nx, Ny = u.shape
    un = u.copy()

    dt_over_DELTA = dt / DELTA
    dt_over_2RHO_delta = dt / (2 * RHO * DELTA)
    dt_over_DELTA2 = dt / (DELTA**2)

    # Loop over interior points, parallel over rows
    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            # Convection in x-direction (first-order upwind)
            if u[i, j] >= 0:
                u_east = u[i, j]
                u_west = u[i, j-1]
            else:
                u_east = u[i, j+1]
                u_west = u[i, j]

            # Convection in y-direction (first-order upwind)
            if v[i, j] >= 0:
                u_north = u[i, j]
                u_south = u[i-1, j]
            else:
                u_north = u[i+1, j]
                u_south = u[i, j]

            convection = dt_over_DELTA * (u[i, j] * (u_east - u_west) + v[i, j] * (u_north - u_south))

            # Diffusion (central difference)
            diffusion = NU * dt_over_DELTA2 * (
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) +
                (u[i+1, j] - 2*u[i, j] + u[i-1, j])
            )

            # Pressure gradient
            pressure_gradient = dt_over_2RHO_delta * (p[i, j+1] - p[i, j-1])

            # Force term
            force_term_x = dt / RHO * Fx[i, j] if Fx is not None else 0.0

            # Update velocity
            un[i, j] = u[i, j] - convection - pressure_gradient + diffusion + force_term_x

    return un

@njit(parallel=CPU_PARALLEL)
def update_y_velocity(u, v, p, dt, Fy=None):
    """
    Updates the velocity field in the y direction based on the momentum equation. Discretization with first order upwind
    for the convection term. Central differences for the Diffusion term. Forcing source term is optional

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): p-pressure field
        dt (float): time step
        Fy (2d-array,optional): y-Forcing field
    Returns:
        vn (2d-array): new v-velocity field
    """
    Nx, Ny = v.shape
    vn = v.copy()

    dt_over_DELTA = dt / DELTA
    dt_over_2RHO_delta = dt / (2 * RHO * DELTA)
    dt_over_DELTA2 = dt / (DELTA**2)

    # Loop over interior points, parallel over rows
    for i in prange(1, Nx-1):   
        for j in range(1, Ny-1):

            # Convection in x-direction (first-order upwind)
            if u[i, j] >= 0:
                v_east = v[i, j]
                v_west = v[i, j-1]
            else:
                v_east = v[i, j+1]
                v_west = v[i, j]

            # Convection in y-direction (first-order upwind)
            if v[i, j] >= 0:
                v_north = v[i, j]
                v_south = v[i-1, j]
            else:
                v_north = v[i+1, j]
                v_south = v[i, j]

            convection = dt_over_DELTA * (
                u[i, j] * (v_east - v_west) +
                v[i, j] * (v_north - v_south)
            )

            # Diffusion (central difference)
            diffusion = NU * dt_over_DELTA2 * (
                (v[i, j+1] - 2*v[i, j] + v[i, j-1]) +
                (v[i+1, j] - 2*v[i, j] + v[i-1, j])
            )

            # Pressure gradient
            pressure_gradient = dt_over_2RHO_delta * (p[i+1, j] - p[i-1, j])

            # Force term
            force_term_y = dt / RHO * Fy[i, j] if Fy is not None else 0.0

            # Update velocity
            vn[i, j] = v[i, j] - convection - pressure_gradient + diffusion + force_term_y

    return vn

@njit(parallel=CPU_PARALLEL)
def pressure_equation_right_side(u, v, b, dt, Fx=None, Fy=None):
    """
    computes the right hand side of the pressure poisson equation

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        b (2d-array): empty b field
        dt (float): time step
        Fx (2d-array,optional): x-Forcing field
        Fy (2d-array,optional): y-Forcing field

    Returns:
        b (2d-array): right hand side of pressure poisson euaqtion
    """
    u_old = u.copy()
    v_old = v.copy()
    if Fx is not None:
        Fx_old = Fx.copy()
    if Fy is not None:
        Fy_old = Fy.copy()
    
    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            du_dx = (u_old[i, j+1] - u_old[i, j-1]) / (2 * DELTA)
            dv_dy = (v_old[i+1, j] - v_old[i-1, j]) / (2 * DELTA)
            du_dy = (u_old[i+1, j] - u_old[i-1, j]) / (2 * DELTA)
            dv_dx = (v_old[i, j+1] - v_old[i, j-1]) / (2 * DELTA)

            divergence = du_dx + dv_dy
            nonlinear = du_dx**2 + 2 * du_dy * dv_dx + dv_dy**2

            b[i, j] = RHO * ((1/dt) * divergence - nonlinear)

            # add forcing terms
            if Fx_old is not None and Fy_old is not None:
                dFx_dx = (Fx_old[i, j+1] - Fx_old[i, j-1]) / (2*DELTA)
                dFy_dy = (Fy_old[i+1, j] - Fy_old[i-1, j]) / (2*DELTA)
                b[i, j] -= RHO * (dFx_dx + dFy_dy)

    return b

@njit(parallel=CPU_PARALLEL)
def pressure_poisson(u, v, p, dt, Fx=None, Fy=None, dp_target=1e-6, max_iter=500):
    """
    Solves the pressure Poisson equation iteratively until the change in 
    the pressure field is smaller than a target threshold or the max_iter count is reached.

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field (initial guess)
        dt (float): time step
        Fx (2d-array,optional): x-Forcing field
        Fy (2d-array,optional): y-Forcing field
        dp_target (float): target max change in pressure for convergence
        max_iter (int): maximum NUmber of iterations

    Returns:
        p (2d-array): updated pressure field
        niter (int): NUmber of iterations performed
    """
    Ny, Nx = p.shape
    b = np.zeros_like(p)
    b = pressure_equation_right_side(u, v, b, dt, Fx, Fy)

    niter = 0
    dp_max = 1.0

    while dp_max > dp_target and niter < max_iter:
        niter += 1
        dp_max = 0.0

        # Gauss Seidel Red-Black sweep
        for color in (0, 1):  # 0 = red, 1 = black
            for i in prange(1, Ny-1):
                for j in range(1, Nx-1):
                    if (i + j) % 2 == color:
                        temp = 0.25 * (p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1] - DELTA**2 * b[i, j])
                        dp_max = max(dp_max, abs(temp - p[i, j]))
                        p[i, j] = temp

        # BCs
        p = BC.apply_pressure_BC(p)
        p = BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)

    return p, niter

# ===============================
# Main
# ===============================

def main():
    print("Initialise")
    print("Cell count: ",int(NX*NY))

    # fields
    u = u_initial.copy()
    v = v_initial.copy()
    p = p_initial.copy()
    T = T_initial.copy()

    Fx = np.zeros_like(p)
    Fy = np.zeros_like(p)
    un = np.empty_like(u)
    vn = np.empty_like(v)
    pn = np.empty_like(p)
    Tn = np.empty_like(T)

    np.copyto(un, u)
    np.copyto(vn, v)
    np.copyto(pn, p)
    np.copyto(Tn, T)
    
    # Initial BCs
    u, v = BC.apply_velocity_BC(u, v)
    p = BC.apply_pressure_BC(p)
    T = BC.apply_temperature_BC(T)

    u, v = BC.obstacle_boundary_conditions_velocity(u, v, obstacle_mask)
    p = BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)
    T = BC.obstacle_boundary_conditions_scalar(T,obstacle_mask,400)

    # values
    start_total_time = time.time() 
    total_loop_time = 0.0
    t = 0
    dt = Helper_Functions.compute_new_timestep(u,v,DELTA,NU,CFL_MAX)
    next_output_time = 0
    output_index = 0

    dataset, u_var, v_var, p_var, T_var, time_var = Output_Functions.initialize_netcdf(OUTPATH, NX, NY, X, Y)

    print("Start time iteration")
    if OUTPUT_STATUS:
        sys.stdout.write(f"\rProgress: [0%]")

    # main loop
    while t < T_MAX:
        loop_start_time = time.time()

        np.copyto(un, u)
        np.copyto(vn, v)
        np.copyto(pn, p)
        np.copyto(Tn, T)

        # Pressure Poisson
        p, niter = pressure_poisson(un, vn, pn, dt, Fx, Fy, TOLERANCE, MAX_ITER)

        # Velocity updates
        u = update_x_velocity(un, vn, p, dt)
        v = update_y_velocity(un, vn, p, dt)

        # Temperature update
        T = general_scalar_transport_equation(T,u,v,dt,NU_TEMPERATURE)

        # Boundary Conditions & Obstacles
        u, v = BC.apply_velocity_BC(u, v)
        p = BC.apply_pressure_BC(p)
        T = BC.apply_temperature_BC(T)

        u, v = BC.obstacle_boundary_conditions_velocity(u, v, obstacle_mask)
        p = BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)
        T = BC.obstacle_boundary_conditions_scalar(T,obstacle_mask,400)

        CFL = Helper_Functions.compute_CFL(u,v,dt,DELTA)

        if CFL > 1:
            print(rf"CFL condition violated in at time {t} seconds!")

        # Write netCDF
        while t >= next_output_time:
            print(f"Writing frame {output_index} at t={t:.6f}, dt={dt:.6f}")
            Output_Functions.write_to_netcdf(u_var, v_var, p_var, T_var, time_var, output_index, t, u, v, p, T, PRECISION)
            output_index += 1
            next_output_time += OUTPUT_TIME_STEP
            if OUTPUT_STATUS:
                print("#################################################")
                print(f"Simulation time {t} sec")
                print(f"CFL-Condition: {np.round(CFL,5)}")
                print(f"NUmber of pressure iterations: {niter}")
                sys.stdout.write(f"\rProgress: [{(t/T_MAX*100):.3f}%]")
                sys.stdout.flush()

        # loop count
        t += dt

        # dynamic time step
        dt_new = Helper_Functions.compute_new_timestep(u,v,DELTA,NU,CFL_MAX)

        # dt limiter
        dt_max_increase = dt * 1.5
        dt_max_decrease = dt * 0.5

        if dt_new > dt_max_increase:
            dt = dt_max_increase
        elif dt_new < dt_max_decrease:
            dt = dt_max_decrease
        else:
            dt = dt_new

        # timing
        loop_end_time = time.time()
        total_loop_time += loop_end_time - loop_start_time

    Output_Functions.close_netcdf(dataset)
    end_total_time = time.time()

    # conclusion
    print("Simulation finished!")
    print(f"Total runtime: {end_total_time - start_total_time:.4f} seconds")
    print(f"Total time spent in main loop: {total_loop_time:.4f} seconds")

main()