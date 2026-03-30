import numpy as np
import time
import sys
from numba import njit
from Boundary_Conditions import neumann_boundary_condition, dirichlet_boundary_condition, obstacle_boundary_conditions_velocity, obstacle_boundary_conditions_pressure
from Obstacles import circle
from Helper import compute_CFL,compute_divergence
from Output import initialize_netcdf,write_to_netcdf,close_netcdf

# fluid
rho = 1.225
nu = 1.81e-5

# time
t_max = 1
dt = 0.001

# solver
tolerance = 0.1
max_iter = 50
precision = np.float32

# resolution
delta = 0.02
nx = 1024
ny = 256
nt = int(t_max/dt)
x = np.linspace(0,(nx-1)*delta,nx)
y = np.linspace(0,(ny-1)*delta,ny)
X, Y = np.meshgrid(x, y)

# i/o
output_fps = 24
print_frequency = 10
output_frequency = int(1/output_fps/dt)
print("Output frequency: ",output_frequency)
outpath = rf"C:\Blenderzeug\BlenderCFD\Test\Test.nc"

# initial conditions
inflow_speed =  5#4 * 5 * y * (ny*delta - y) / (ny*delta)**2

#u_initial = (np.ones_like(X) * inflow_speed[:, np.newaxis]).astype(precision)
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

@njit
def update_x_velocity(u, v, p, Fx=None, Fy=None):
    """
    updates the velocity field in the x direction based on the momentum equation. Discretization with first order upwind

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): p-pressure field
    Returns:
        un (2d-array): new u-velocity field
    """
    un = u.copy()
    Nx, Ny = u.shape
    fe1, fe2 = compute_F(u)
    fw1, fw2 = fe1, fe2
    fnorth1, fnorth2 = compute_F(v)
    fs1, fs2 = fnorth1, fnorth2

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # convection in x-direction
            u_east = u[i, j] * fe1[i, j] + u[i, j+1] * fe2[i, j]
            u_west = u[i, j-1] * fw1[i, j] + u[i, j] * fw2[i, j]

            # convection in y-direction
            u_north = u[i, j] * fnorth1[i, j] + u[i+1, j] * fnorth2[i, j]
            u_south = u[i-1, j] * fs1[i, j] + u[i, j] * fs2[i, j]

            convection = u[i, j] * dt / delta * (u_east - u_west) + v[i, j] * dt / delta * (u_north - u_south)

            # diffusion
            diffusion = nu * (
                dt / delta**2 * (u[i, j+1] - 2*u[i, j] + u[i, j-1]) +
                dt / delta**2 * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
            )

            # pressure gradient
            pressure_gradient = dt / (2 * rho * delta) * (p[i, j+1] - p[i, j-1])

            # force term
            if Fx is not None:
                force_term_x = dt / rho * Fx[i, j]
            else:
                force_term_x = 0

            # update velocity
            un[i, j] = u[i, j] - convection - pressure_gradient + diffusion + force_term_x

    return un

@njit
def update_y_velocity(u, v, p, Fx=None, Fy=None):
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
    vn = v.copy()
    Nx, Ny = v.shape

    fe1, fe2 = compute_F(u)
    fw1, fw2 = fe1, fe2
    fnorth1, fnorth2 = compute_F(v)
    fs1, fs2 = fnorth1, fnorth2

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # convection in x-direction
            v_east = v[i, j] * fe1[i, j] + v[i, j+1] * fe2[i, j]
            v_west = v[i, j-1] * fw1[i, j] + v[i, j] * fw2[i, j]

            # convection in y-direction
            v_north = v[i, j] * fnorth1[i, j] + v[i+1, j] * fnorth2[i, j]
            v_south = v[i-1, j] * fs1[i, j] + v[i, j] * fs2[i, j]

            convection = (u[i, j] * (v_east - v_west) + v[i, j] * (v_north - v_south)) * dt / delta

            # diffusion
            diffusion = nu * (
                dt / delta**2 * (v[i, j+1] - 2*v[i, j] + v[i, j-1]) +
                dt / delta**2 * (v[i+1, j] - 2*v[i, j] + v[i-1, j])
            )

            # pressure gradient in y-direction
            pressure_gradient = dt / (2 * rho * delta) * (p[i+1, j] - p[i-1, j])

            # force term
            if Fy is not None:
                force_term_y = dt / rho * Fy[i, j]
            else:
                force_term_y = 0

            # update velocity
            vn[i, j] = v[i, j] - convection - pressure_gradient + diffusion + force_term_y

    return vn

@njit
def pressure_equation_right_side(u, v, b, Fx=None, Fy=None):
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
def pressure_poisson(u, v, p, Fx=None, Fy=None, dp_target=1e-6, max_iter=500):
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
    b = np.zeros_like(p)
    b = pressure_equation_right_side(u, v, b, Fx, Fy)

    niter = 0
    dp_max = 1.0 

    while dp_max > dp_target and niter < max_iter:
        niter += 1
        p_old = p.copy()

        for i in range(1, p.shape[0] - 1):
            for j in range(1, p.shape[1] - 1):
                laplace_x = p_old[i, j+1] + p_old[i, j-1]
                laplace_y = p_old[i+1, j] + p_old[i-1, j]
                p[i, j] = 0.25 * (laplace_x + laplace_y - delta**2 * b[i, j])

        # BCs
        p = apply_pressure_BC(p)
        p = obstacle_boundary_conditions_pressure(p, obstacle_mask)

        # change of pressure field per itteration
        dp_max = np.max(np.abs(p - p_old))

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
    v = dirichlet_boundary_condition(v, "bottom", 0.0) 
    u = dirichlet_boundary_condition(u, "bottom", 0.0)

    v = dirichlet_boundary_condition(v, "top", 0.0)  
    u = dirichlet_boundary_condition(u, "top", 0.0)  

    u = dirichlet_boundary_condition(u, "left", inflow_speed)
    v = dirichlet_boundary_condition(v, "left", 0.0)

    u = neumann_boundary_condition(u, "right")
    v = neumann_boundary_condition(v, "right")

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
    p = neumann_boundary_condition(p, "bottom") 
    p = neumann_boundary_condition(p, "top") 
    p = neumann_boundary_condition(p, "left")  
    p = neumann_boundary_condition(p, "right") 
    return p


# def main():
#     print("Initialise")
#     start_total_time = time.time() 
#     u = u_initial.copy()
#     v = v_initial.copy()
#     p = p_initial.copy()
#     u,v = apply_velocity_BC(u,v)
#     p = apply_pressure_BC(p)
#     u,v = obstacle_boundary_conditions_velocity(u,v,obstacle_mask)
#     p = obstacle_boundary_conditions_pressure(p,obstacle_mask)

#     dataset, u_var, v_var, p_var = initialize_netcdf(outpath, nx, ny, nt, delta, X, Y)

#     print("Start time itteration")
#     sys.stdout.write(f"\rProgress: [0%]")
#     for n in range(nt):
#         start_time = time.time()

#         Fx=np.zeros_like(p)
#         Fy=np.zeros_like(p)

#         un = u.copy()
#         vn = v.copy()
#         pn = p.copy()

#         p,niter = pressure_poisson(un, vn, pn, Fx, Fy, tolerance, max_iter)
#         u = update_x_velocity(un, vn, p, Fx, Fy)
#         v = update_y_velocity(un, vn, p, Fx, Fy)
        
#         # BCs
#         u,v = apply_velocity_BC(u,v)
#         p = apply_pressure_BC(p)
      
#         u,v = obstacle_boundary_conditions_velocity(u,v,obstacle_mask)
#         p = obstacle_boundary_conditions_pressure(p,obstacle_mask)

#         CFL = compute_CFL(u,v,dt,delta)
#         _ , div_l1 = compute_divergence(u,v,delta)
#         end_time = time.time()

#         if  n % print_frequency == 0:
#             # Output
#             print("#################################################")
#             print(f"Timestep {n} of {nt} steps")
#             print(f"CFL-Condition: {np.round(CFL,5)}")
#             print(f"Number of pressure itterations: {niter}")
#             print(f"Divergence of velocity field: {div_l1}")
#             print(f"Simulation time for timestep: {end_time - start_time:.4f} s")
#             sys.stdout.write(f"\rProgress: [{np.round(n/nt*100,3)}%]")
#             sys.stdout.flush()

#         if n % output_frequency == 0:
#             timestep_index = n // output_frequency
#             write_to_netcdf(u_var, v_var, p_var, timestep_index, u, v, p)
    
#     close_netcdf(dataset)
#     end_total_time = time.time()
#     print("Simulation finished!")
#     print(f"Total runtime: {end_total_time - start_total_time:.4f} seconds")

def main():
    print("Initialise")
    start_total_time = time.time() 
    u = u_initial.copy()
    v = v_initial.copy()
    p = p_initial.copy()
    
    # Initial BCs
    u, v = apply_velocity_BC(u, v)
    p = apply_pressure_BC(p)
    u, v = obstacle_boundary_conditions_velocity(u, v, obstacle_mask)
    p = obstacle_boundary_conditions_pressure(p, obstacle_mask)

    dataset, u_var, v_var, p_var = initialize_netcdf(outpath, nx, ny, nt, delta, X, Y)

    print("Start time iteration")
    sys.stdout.write(f"\rProgress: [0%]")

    # Zeit-Summen für jeden Block
    total_loop_time = 0.0
    total_pressure_time = 0.0
    total_velocity_update_time = 0.0
    total_bc_time = 0.0
    total_netcdf_time = 0.0

    for n in range(nt):
        loop_start_time = time.time()

        Fx = np.zeros_like(p)
        Fy = np.zeros_like(p)

        un = u.copy()
        vn = v.copy()
        pn = p.copy()

        # Pressure Poisson
        t0 = time.time()
        p, niter = pressure_poisson(un, vn, pn, Fx, Fy, tolerance, max_iter)
        t1 = time.time()
        total_pressure_time += t1 - t0

        # Velocity updates
        t0 = time.time()
        u = update_x_velocity(un, vn, p, Fx, Fy)
        v = update_y_velocity(un, vn, p, Fx, Fy)
        t1 = time.time()
        total_velocity_update_time += t1 - t0

        # Boundary Conditions & Obstacles
        t0 = time.time()
        u, v = apply_velocity_BC(u, v)
        p = apply_pressure_BC(p)
        u, v = obstacle_boundary_conditions_velocity(u, v, obstacle_mask)
        p = obstacle_boundary_conditions_pressure(p, obstacle_mask)
        t1 = time.time()
        total_bc_time += t1 - t0

        # CFL & Divergence (nur Berechnung, keine Speicherung)
        CFL = compute_CFL(u, v, dt, delta)
        _, div_l1 = compute_divergence(u, v, delta)

        # NetCDF schreiben
        if n % output_frequency == 0:
            t0 = time.time()
            timestep_index = n // output_frequency
            write_to_netcdf(u_var, v_var, p_var, timestep_index, u, v, p)
            t1 = time.time()
            total_netcdf_time += t1 - t0

        loop_end_time = time.time()
        total_loop_time += loop_end_time - loop_start_time

        if n % print_frequency == 0:
            print("#################################################")
            print(f"Timestep {n} of {nt} steps")
            print(f"CFL-Condition: {np.round(CFL,5)}")
            print(f"Number of pressure iterations: {niter}")
            print(f"Divergence of velocity field: {div_l1}")
            print(f"Simulation time for timestep: {loop_end_time - loop_start_time:.4f} s")
            sys.stdout.write(f"\rProgress: [{np.round(n/nt*100,3)}%]")
            sys.stdout.flush()
    
    close_netcdf(dataset)
    end_total_time = time.time()

    print("Simulation finished!")
    print(f"Total runtime: {end_total_time - start_total_time:.4f} seconds")
    print(f"Total time spent in main loop: {total_loop_time:.4f} seconds")
    print(f"  -> Pressure Poisson: {total_pressure_time:.4f} s")
    print(f"  -> Velocity update (x+y): {total_velocity_update_time:.4f} s")
    print(f"  -> Boundary & obstacle BCs: {total_bc_time:.4f} s")
    print(f"  -> NetCDF writing: {total_netcdf_time:.4f} s")

main()