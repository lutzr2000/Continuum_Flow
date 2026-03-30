import numpy as np
import time
import sys

from numba import njit, prange

import Boundary_Conditions as BC
from Obstacles import circle
from Output import initialize_netcdf,write_to_netcdf,close_netcdf
import Helper

# fluid
rho = 1.225
nu = 1.81e-5

# time
t_max = 1
dt_minimum = 0.00001
CFL_max = 0.9

# solver 
tolerance = 0.01
max_iter = 50
precision = np.float32

# resolution
delta = 0.02
nx = 1024
ny = 256
#nt = int(t_max/dt)
x = np.linspace(0,(nx-1)*delta,nx)
y = np.linspace(0,(ny-1)*delta,ny)
X, Y = np.meshgrid(x, y)

# i/o
output_fps = 24
print_frequency = 100
output_time_step = 1/output_fps
outpath = rf"C:\Blenderzeug\BlenderCFD\Test\Test.nc"
output_step_tolerance = output_time_step*0.1
output_status = True

# initial conditions
inflow_speed =  5

u_initial = np.ones_like(X).astype(precision)*inflow_speed
v_initial = np.zeros_like(X).astype(precision)
p_initial = np.zeros_like(X).astype(precision)

# Geometry
circle1_mask = circle(X,Y,nx*0.2*delta,ny*0.5*delta,0.5)
circle2_mask = circle(X,Y,nx*0.25*delta,ny*0.3*delta,0.6)
circle3_mask = circle(X,Y,nx*0.4*delta,ny*0.8*delta,0.3)

obstacle_mask = circle1_mask | circle2_mask | circle3_mask

# Reynolds number
Re = np.max(u_initial)*ny*delta/nu
print(f"Reynolds number: {Re}")
time.sleep(1)
###################################################

@njit
def compute_F(vel):
    """
    Computes mask for the signs of a given velocity field

    Args:
        vel (2d-array): velocity field
    Returns:
        pos_part (2d-array): mask with positive signs
        neg_part (2d-array): mask with negative signs
    """
    Nx, Ny = vel.shape
    pos_part = np.zeros((Nx, Ny), dtype=np.float64)
    neg_part = np.zeros((Nx, Ny), dtype=np.float64)

    for i in range(Nx):
        for j in range(Ny):
            denom = abs(vel[i,j]) + 1e-6
            ratio = vel[i,j] / denom
            pos_part[i,j] = ratio if ratio > 0.0 else 0.0
            neg_part[i,j] = -ratio if ratio < 0.0 else 0.0

    return pos_part, neg_part

@njit(parallel=True)
def update_x_velocity(u, v, p, dt, Fx=None, Fy=None):
    """
    updates the velocity field in the x direction based on the momentum equation. Discretization with first order upwind

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): p-pressure field
    Returns:
        un (2d-array): new u-velocity field
    """
    Nx, Ny = u.shape
    un = u.copy()

    dt_over_delta = dt / delta
    dt_over_2rho_delta = dt / (2 * rho * delta)
    dt_over_delta2 = dt / (delta**2)

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

            convection = dt_over_delta * (u[i, j] * (u_east - u_west) + v[i, j] * (u_north - u_south))

            # Diffusion (central difference)
            diffusion = nu * dt_over_delta2 * (
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) +
                (u[i+1, j] - 2*u[i, j] + u[i-1, j])
            )

            # Pressure gradient
            pressure_gradient = dt_over_2rho_delta * (p[i, j+1] - p[i, j-1])

            # Force term
            force_term_x = dt / rho * Fx[i, j] if Fx is not None else 0.0

            # Update velocity
            un[i, j] = u[i, j] - convection - pressure_gradient + diffusion + force_term_x

    return un

@njit(parallel=True)
def update_y_velocity(u, v, p, dt, Fx=None, Fy=None):
    """
    updates the velocity field in the y direction based on the momentum equation.
    Discretization with first order upwind (consistent with update_x_velocity)

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field

    Returns:
        vn (2d-array): new v-velocity field
    """
    Nx, Ny = v.shape
    vn = v.copy()

    dt_over_delta = dt / delta
    dt_over_2rho_delta = dt / (2 * rho * delta)
    dt_over_delta2 = dt / (delta**2)

    for i in prange(1, Nx-1):   # parallel über i
        for j in range(1, Ny-1):

            # --- Upwind convection in x-direction ---
            if u[i, j] >= 0:
                v_east = v[i, j]
                v_west = v[i, j-1]
            else:
                v_east = v[i, j+1]
                v_west = v[i, j]

            # --- Upwind convection in y-direction ---
            if v[i, j] >= 0:
                v_north = v[i, j]
                v_south = v[i-1, j]
            else:
                v_north = v[i+1, j]
                v_south = v[i, j]

            convection = dt_over_delta * (
                u[i, j] * (v_east - v_west) +
                v[i, j] * (v_north - v_south)
            )

            # --- Diffusion ---
            diffusion = nu * dt_over_delta2 * (
                (v[i, j+1] - 2*v[i, j] + v[i, j-1]) +
                (v[i+1, j] - 2*v[i, j] + v[i-1, j])
            )

            # --- Pressure gradient (y-direction) ---
            pressure_gradient = dt_over_2rho_delta * (p[i+1, j] - p[i-1, j])

            # --- Force term ---
            force_term_y = dt / rho * Fy[i, j] if Fy is not None else 0.0

            # --- Update ---
            vn[i, j] = v[i, j] - convection - pressure_gradient + diffusion + force_term_y

    return vn

@njit
def pressure_equation_right_side(u, v, b, dt, Fx=None, Fy=None):
    """
    computes the right hand side of the pressure poisson equation

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field

    Returns:
        b (2d-array): right hand side of pressure poisson euaqtion
    """
    u_old = u.copy()
    v_old = v.copy()
    if Fx is not None:
        Fx_old = Fx.copy()
    if Fy is not None:
        Fy_old = Fy.copy()

    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            du_dx = (u_old[i, j+1] - u_old[i, j-1]) / (2 * delta)
            dv_dy = (v_old[i+1, j] - v_old[i-1, j]) / (2 * delta)
            du_dy = (u_old[i+1, j] - u_old[i-1, j]) / (2 * delta)
            dv_dx = (v_old[i, j+1] - v_old[i, j-1]) / (2 * delta)

            divergence = du_dx + dv_dy
            nonlinear = du_dx**2 + 2 * du_dy * dv_dx + dv_dy**2

            b[i, j] = rho * ((1/dt) * divergence - nonlinear)

            # add forcing terms
            if Fx is not None and Fy is not None:
                dFx_dx = (Fx_old[i, j+1] - Fx_old[i, j-1]) / (2*delta)
                dFy_dy = (Fy_old[i+1, j] - Fy_old[i-1, j]) / (2*delta)
                b[i, j] -= rho * (dFx_dx + dFy_dy)

    return b

@njit(parallel=True)
def pressure_poisson(u, v, p, dt, Fx=None, Fy=None, dp_target=1e-6, max_iter=500):
    """
    Solves the pressure Poisson equation iteratively until the change in 
    the pressure field is smaller than a target threshold.

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field (initial guess)
        dp_target (float): target max change in pressure for convergence
        max_iter (int): maximum number of iterations

    Returns:
        p (2d-array): updated pressure field
        niter (int): number of iterations performed
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
                        temp = 0.25 * (p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1] - delta**2 * b[i, j])
                        dp_max = max(dp_max, abs(temp - p[i, j]))
                        p[i, j] = temp

            # BCs
            p = apply_pressure_BC(p)
            p = BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)

    return p, niter

@njit
def apply_velocity_BC(u,v):
    """
    Applies a set of velocity boundary conditions to all sides

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field

    Returns:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
    """
    v = BC.dirichlet_boundary_condition(v, "bottom", 0.0) 
    u = BC.neumann_boundary_condition(u, "bottom")

    v = BC.dirichlet_boundary_condition(v, "top", 0.0)  
    u = BC.neumann_boundary_condition(u, "top")  

    u = BC.dirichlet_boundary_condition(u, "left", inflow_speed)
    v = BC.dirichlet_boundary_condition(v, "left", 0.0)

    u = BC.neumann_boundary_condition(u, "right")
    v = BC.neumann_boundary_condition(v, "right")

    return u,v

@njit
def apply_pressure_BC(p):
    """
    Applies a set of pressure boundary conditions to all sides

    Args:
        p (2d-array): pressure field

    Returns:
        p (2d-array): pressure field
    """
    p = BC.neumann_boundary_condition(p, "bottom") 
    p = BC.neumann_boundary_condition(p, "top") 
    p = BC.neumann_boundary_condition(p, "left")  
    p = BC.neumann_boundary_condition(p, "right") 
    return p

def main():
    print("Initialise")
    print("Cell count: ",int(nx*ny))
    start_total_time = time.time() 
    u = u_initial.copy()
    v = v_initial.copy()
    p = p_initial.copy()

    Fx = np.zeros_like(p)
    Fy = np.zeros_like(p)
    un = np.empty_like(u)
    vn = np.empty_like(v)
    pn = np.empty_like(p)
    
    # Initial BCs
    u, v = apply_velocity_BC(u, v)
    p = apply_pressure_BC(p)
    u, v = BC.obstacle_boundary_conditions_velocity(u, v, obstacle_mask)
    p = BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)

    dataset, u_var, v_var, p_var = initialize_netcdf(outpath, nx, ny, X, Y)

    print("Start time iteration")
    if output_status:
        sys.stdout.write(f"\rProgress: [0%]")

    # time summation for every block
    total_loop_time = 0.0

    # dynamic time stepping
    t = 0
    n = 0
    k = 0
    dt = dt_minimum
    output_step_this_loop=True

    np.copyto(un, u)
    np.copyto(vn, v)
    np.copyto(pn, p)

    while t < t_max:
        loop_start_time = time.time()

        Fx.fill(0)
        Fy.fill(0)

        np.copyto(un, u)
        np.copyto(vn, v)
        np.copyto(pn, p)

        # Pressure Poisson
        p, niter = pressure_poisson(un, vn, pn, dt, Fx, Fy, tolerance, max_iter)

        # Velocity updates
        u = update_x_velocity(un, vn, p, dt, Fx, Fy)
        v = update_y_velocity(un, vn, p, dt, Fx, Fy)

        # Boundary Conditions & Obstacles
        u, v = apply_velocity_BC(u, v)
        p = apply_pressure_BC(p)
        u, v = BC.obstacle_boundary_conditions_velocity(u, v, obstacle_mask)
        p = BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)

        # NetCDF schreiben
        if output_step_this_loop:
            timestep_index = int(round(t / output_time_step)) 
            write_to_netcdf(u_var, v_var, p_var, timestep_index, u, v, p)
            n += 1
        if k % 100 == 0:
            if output_status:
                print("#################################################")
                print(f"Simulation time {t} sec")
                print(f"CFL-Condition: {np.round(Helper.compute_CFL(u,v,dt,delta),5)}")
                print(f"Number of pressure iterations: {niter}")
                sys.stdout.write(f"\rProgress: [{(t/t_max*100):.3f}%]")
                sys.stdout.flush()

        # loop count
        t += dt
        k += 1

        # new dt
        dt = Helper.compute_new_timestep(u,v,delta,CFL_max)

        if t > n*output_time_step:
            dt = n*output_time_step-t
            if dt < dt_minimum:
                dt = dt_minimum
            output_step_this_loop = True
        else: 
            output_step_this_loop = False

        loop_end_time = time.time()
        total_loop_time += loop_end_time - loop_start_time
        
    close_netcdf(dataset)
    end_total_time = time.time()

    print("Simulation finished!")
    print(f"Total runtime: {end_total_time - start_total_time:.4f} seconds")
    print(f"Total time spent in main loop: {total_loop_time:.4f} seconds")

main()