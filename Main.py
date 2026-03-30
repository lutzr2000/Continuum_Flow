import numpy as np
import time
import sys
from Boundary_Conditions import neumann_boundary_condition, dirichlet_boundary_condition, obstacle_boundary_conditions_velocity, obstacle_boundary_conditions_pressure
from Obstacles import circle
from Helper import compute_CFL,compute_divergence
from Output import initialize_netcdf,write_to_netcdf,close_netcdf

# fluid
rho = 1.225
nu = 1.81e-5

# time
t_max = 5
dt = 0.01

# solver
tolerance = 0.01
max_iter = 50
precision = np.float32

# resolution
delta = 0.08
nx = 512
ny = 64
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
inflow_speed =  4 * 5 * y * (ny*delta - y) / (ny*delta)**2

u_initial = (np.ones_like(X) * inflow_speed[:, np.newaxis]).astype(precision)
#u_initial = np.ones_like(X).astype(precision)*5
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

def compute_F(vel):
    """
    Computes mask for the signs of a given velocity field

    Args:
        vel (2d-array): velocity field
    Returns:
        pos_part (2d-array): mask with positive signs
        neg_part (2d-array): mask with negative signs
    """
    denom = abs(vel) + 1e-6
    pos_part = np.maximum(vel/denom, 0)
    neg_part = np.maximum(-vel/denom, 0)
    return pos_part, neg_part

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
    fe1, fe2 = compute_F(u)       
    fw1, fw2 = fe1, fe2
    u_east = u[1:-1, 1:-1] * fe1[1:-1, 1:-1] + u[1:-1, 2:] * fe2[1:-1, 1:-1]     
    u_west = u[1:-1, 0:-2] * fw1[1:-1, 1:-1] + u[1:-1, 1:-1]* fw2[1:-1, 1:-1]

    fnorth1, fnorth2 = compute_F(v)       
    fs1, fs2 = fnorth1, fnorth2
    u_north = u[1:-1, 1:-1] * fnorth1[1:-1, 1:-1] + u[2:, 1:-1] * fnorth2[1:-1, 1:-1]     
    u_south = u[0:-2, 1:-1] * fs1[1:-1, 1:-1] + u[1:-1, 1:-1]* fs2[1:-1, 1:-1]

    convection = u[1:-1, 1:-1] * dt / delta * (u_east - u_west) + v[1:-1, 1:-1] * dt / delta * (u_north - u_south)
    diffusion = nu * (dt / delta**2 * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2]) + dt / delta**2 * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1]))
    pressure_gradient = dt / (2 * rho * delta) * (p[1:-1, 2:] - p[1:-1, 0:-2])

    if Fx is not None:
        force_term_x = dt / rho * Fx[1:-1, 1:-1]
    else:
        force_term_x = 0
   
    un[1:-1, 1:-1] = u[1:-1, 1:-1] - convection - pressure_gradient + diffusion + force_term_x 
    return un

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
    fe1, fe2 = compute_F(u)  
    fw1, fw2 = fe1, fe2
    v_east = v[1:-1, 1:-1] * fe1[1:-1, 1:-1] + v[1:-1, 2:] * fe2[1:-1, 1:-1]
    v_west = v[1:-1, 0:-2] * fw1[1:-1, 1:-1] + v[1:-1, 1:-1] * fw2[1:-1, 1:-1]

    fnorth1, fnorth2 = compute_F(v)   
    fs1, fs2 = fnorth1, fnorth2
    v_north = v[1:-1, 1:-1] * fnorth1[1:-1, 1:-1] + v[2:, 1:-1] * fnorth2[1:-1, 1:-1]
    v_south = v[0:-2, 1:-1] * fs1[1:-1, 1:-1] + v[1:-1, 1:-1] * fs2[1:-1, 1:-1]

    convection = (u[1:-1, 1:-1] * (v_east - v_west) +v[1:-1, 1:-1] * (v_north - v_south)) * dt / delta
    diffusion = nu * (dt / delta**2 * (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, 0:-2]) + dt / delta**2 * (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[0:-2, 1:-1]))
    pressure_gradient = dt / (2 * rho * delta) * (p[2:, 1:-1] - p[0:-2, 1:-1])

    if Fy is not None:
        force_term_y = dt / rho * Fy[1:-1, 1:-1]
    else:
        force_term_y = 0

    vn[1:-1, 1:-1] = v[1:-1, 1:-1]- convection- pressure_gradient + diffusion + force_term_y

    return vn

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
    du_dx = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * delta)
    dv_dy = (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * delta)
    du_dy = (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * delta)
    dv_dx = (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * delta)
    divergence = du_dx + dv_dy
    nonlinear = du_dx**2 + 2 * du_dy * dv_dx + dv_dy**2

    b[1:-1, 1:-1] = rho * ((1/dt) * divergence - nonlinear)

    # add divergence of forcing terms
    if Fx is not None and Fy is not None:
        dFx_dx = (Fx[1:-1, 2:] - Fx[1:-1, 0:-2]) / (2*delta)
        dFy_dy = (Fy[2:, 1:-1] - Fy[0:-2, 1:-1]) / (2*delta)
        b[1:-1, 1:-1] -= rho * (dFx_dx + dFy_dy)

    return b

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

        laplace_x = p[1:-1, 2:] + p[1:-1, 0:-2]
        laplace_y = p[2:, 1:-1] + p[0:-2, 1:-1]

        p[1:-1, 1:-1] = 0.25 * (laplace_x + laplace_y - delta**2 * b[1:-1, 1:-1])

        # BCs
        p = apply_pressure_BC(p)
        p = obstacle_boundary_conditions_pressure(p, obstacle_mask)

        # change of pressure field per itteration
        dp_max = np.max(np.abs(p - p_old))

    return p, niter

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


def main():
    print("Initialise")
    u = u_initial.copy()
    v = v_initial.copy()
    p = p_initial.copy()
    u,v = apply_velocity_BC(u,v)
    p = apply_pressure_BC(p)
    u,v = obstacle_boundary_conditions_velocity(u,v,obstacle_mask)
    p = obstacle_boundary_conditions_pressure(p,obstacle_mask)

    dataset, u_var, v_var, p_var = initialize_netcdf(outpath, nx, ny, nt, delta, X, Y)

    print("Start time itteration")
    sys.stdout.write(f"\rProgress: [0%]")
    for n in range(nt):
        start_time = time.time()

        Fx=np.zeros_like(p)
        Fy=np.zeros_like(p)

        un = u.copy()
        vn = v.copy()
        pn = p.copy()

        p,niter = pressure_poisson(un, vn, pn, Fx, Fy, tolerance, max_iter)
        u = update_x_velocity(un, vn, p, Fx, Fy)
        v = update_y_velocity(un, vn, p, Fx, Fy)
        
        # BCs
        u,v = apply_velocity_BC(u,v)
        p = apply_pressure_BC(p)
      
        u,v = obstacle_boundary_conditions_velocity(u,v,obstacle_mask)
        p = obstacle_boundary_conditions_pressure(p,obstacle_mask)

        CFL = compute_CFL(u,v,dt,delta)
        _ , div_l1 = compute_divergence(u,v,delta)
        end_time = time.time()

        if  n % print_frequency == 0:
            # Output
            print("#################################################")
            print(f"Timestep {n} of {nt} steps")
            print(f"CFL-Condition: {np.round(CFL,5)}")
            print(f"Number of pressure itterations: {niter}")
            print(f"Divergence of velocity field: {div_l1}")
            print(f"Simulation time for timestep: {end_time - start_time:.4f} s")
            sys.stdout.write(f"\rProgress: [{np.round(n/nt*100,3)}%]")
            sys.stdout.flush()

        if n % output_frequency == 0:
            timestep_index = n // output_frequency
            write_to_netcdf(u_var, v_var, p_var, timestep_index, u, v, p)
    
    close_netcdf(dataset)
    print("Simulation finished!")

main()