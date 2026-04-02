"""
Description:
    A finite difference based CFD solver for Blender. Convetion is discretized by first order upwind,
    diffusion and pressure by central differencing. The pressure velocity coupling is done by solving
    a poisson equation.

    This code is based on the methods presented here:
    https://drzgan.github.io/Python_CFD/15.1.%20Cavity%20flow%20with%20upwind%20scheme.html
    Huge thanks to Dr. Zhengtao Gan.
"""
# TODO move all scalar updates into one method, also move the source terms there
# ===============================
# Import
# ===============================

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
TEMPERATURE_PRODUCTION_RATE = 1
SMOKE_DISSIPATION_RATE = 0.1
SMOKE_PRODUCTION_RATE = 1
FUEL_BURN_RATE = 0.1
FUEL_IGNITION_TEMPERATURE = 100
R_GAS = 287
T_REFERENCE = 300
BUOANCY_FACTOR = 1/300
EXPANSION_RATE = 0.003

# time
T_MAX = 30
CFL_MAX = 0.8

# solver 
MAX_ITER = 4
PRECISION = np.float32
CPU_PARALLEL = True
RESERVE_CPU_CORES_FOR_IO = 1

# resolution
scale_factor=1
DELTA = 0.04/scale_factor
NX = 512*scale_factor
NY = 512*scale_factor
x = np.linspace(0,(NX-1)*DELTA,NX)
y = np.linspace(0,(NY-1)*DELTA,NY)
X, Y = np.meshgrid(x, y)

# output
OUTPUT_FPS = 24
PRINT_FREQUENCY = 100
OUTPUT_TIME_STEP = 1/OUTPUT_FPS
OUTPATH = rf"C:\Blenderzeug\BlenderCFD\Test\Test.nc"
OUTPUT_STATUS = False
WRITE_QUEUE_SIZE = 512
NETCDF_COMPRESSION_LEVEL = 0 

# Boundary_conditons
U_INFLOW = 0
V_INFLOW = 0

# Geometry
circle1_mask = Obstacles.circle(X,Y,NX*0.5*DELTA,NY*0.05*DELTA,0.6)
obstacle_mask = circle1_mask
obstacle_i, obstacle_j = np.where(obstacle_mask)

# ===============================
# Functions
# ===============================

@njit(parallel=CPU_PARALLEL)
def buoancy_approximation(T,Fy,expansion_coefficent,T_ref):
    """
    Computes a forcing in the y-direction based on the Bousinesq approximation.

    Args:
        T (2d-array): temperature field
        Fy (2d-array): y-force field
        expansion_coefficent (float): how strongly the air rises
        T_ref (float): reference temeprature, air hotter than this rises
    Returns:
        Fy (2d-array): y-force field
    """
    Nx, Ny = T.shape
    g=9.81
    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            Fy[i,j]=g*expansion_coefficent*(T[i,j]-T_ref)

    return Fy

@njit(parallel=CPU_PARALLEL)
def update_scalar_fields(T, smoke, fuel, u, v, dt, T_out, smoke_out, fuel_out):
    """
    Updates temperature, smoke and fuel in a single coupled transport sweep. Fuel starts burning if the temperature is higher
    than the ignition temperature. If the fuel burns heat and smoke are produced.

    Args:
        T (2d-array): temperature field
        smoke (2d-array): smoke field
        fuel (2d-array): fuel field
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        dt (float): time step
        T_out (2d-array): preallocated output temperature field
        smoke_out (2d-array): preallocated output smoke field
        fuel_out (2d-array): preallocated output fuel field
    Returns:
        T_out, smoke_out, fuel_out (2d-array): updated scalar fields
    """
    Nx, Ny = u.shape

    dt_over_delta = dt / DELTA
    dt_over_delta2 = dt / (DELTA * DELTA)
    temp_diffusion_coeff = NU_TEMPERATURE * dt_over_delta2
    smoke_diffusion_coeff = NU_SMOKE * dt_over_delta2
    fuel_diffusion_coeff = NU_FUEL * dt_over_delta2

    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            uij = u[i, j]
            vij = v[i, j]
            # ----------Upwinding----------------
            if uij >= 0:
                t_east = T[i, j]
                t_west = T[i, j-1]
                smoke_east = smoke[i, j]
                smoke_west = smoke[i, j-1]
                fuel_east = fuel[i, j]
                fuel_west = fuel[i, j-1]
            else:
                t_east = T[i, j+1]
                t_west = T[i, j]
                smoke_east = smoke[i, j+1]
                smoke_west = smoke[i, j]
                fuel_east = fuel[i, j+1]
                fuel_west = fuel[i, j]

            if vij >= 0:
                t_north = T[i, j]
                t_south = T[i-1, j]
                smoke_north = smoke[i, j]
                smoke_south = smoke[i-1, j]
                fuel_north = fuel[i, j]
                fuel_south = fuel[i-1, j]
            else:
                t_north = T[i+1, j]
                t_south = T[i, j]
                smoke_north = smoke[i+1, j]
                smoke_south = smoke[i, j]
                fuel_north = fuel[i+1, j]
                fuel_south = fuel[i, j]

            T_center = T[i, j]
            smoke_center = smoke[i, j]
            fuel_center = fuel[i, j]

            # ----------Convecting----------------
            temp_convection = dt_over_delta * (uij * (t_east - t_west) + vij * (t_north - t_south))
            smoke_convection = dt_over_delta * (uij * (smoke_east - smoke_west) + vij * (smoke_north - smoke_south))
            fuel_convection = dt_over_delta * (uij * (fuel_east - fuel_west) + vij * (fuel_north - fuel_south))

            # ----------Diffusion----------------
            temp_diffusion = temp_diffusion_coeff * (
                (T[i, j+1] - 2.0 * T_center + T[i, j-1]) +
                (T[i+1, j] - 2.0 * T_center + T[i-1, j])
            )
            smoke_diffusion = smoke_diffusion_coeff * (
                (smoke[i, j+1] - 2.0 * smoke_center + smoke[i, j-1]) +
                (smoke[i+1, j] - 2.0 * smoke_center + smoke[i-1, j])
            )
            fuel_diffusion = fuel_diffusion_coeff * (
                (fuel[i, j+1] - 2.0 * fuel_center + fuel[i, j-1]) +
                (fuel[i+1, j] - 2.0 * fuel_center + fuel[i-1, j])
            )

            # ----------Source terms----------------
            # if the temperature is higher than the ignition temperature fuel burns
            if T_center > FUEL_IGNITION_TEMPERATURE:
                fuel_source = -FUEL_BURN_RATE * fuel_center
            else:
                fuel_source = 0.0

            # the temperature increases if there is a fuel source e.g. burning
            temperature_source = (
                -TEMPERATURE_DISSIPATION_RATE * (T_center - T_REFERENCE) +
                TEMPERATURE_PRODUCTION_RATE * (-fuel_source)
            )
            # smoke is also produced if there is a fuel source
            smoke_source = SMOKE_PRODUCTION_RATE * (-fuel_source) - SMOKE_DISSIPATION_RATE * smoke_center

            # ----------Updating----------------
            T_out[i, j] = T_center - temp_convection + temp_diffusion + dt * temperature_source
            smoke_out[i, j] = smoke_center - smoke_convection + smoke_diffusion + dt * smoke_source
            fuel_out[i, j] = fuel_center - fuel_convection + fuel_diffusion + dt * fuel_source

    return T_out, smoke_out, fuel_out


@njit(parallel=CPU_PARALLEL)
def update_x_velocity(u, v, p, dt, Fx, un):
    """
    Updates the velocity field in the x direction based on the momentum equation. Discretization with first order upwind
    for the convection term. Central differences for the Diffusion term. 

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): p-pressure field
        dt (float): time step
        Fx (2d-array): x-Forcing field
        un (2d-array): preallocated output array
    Returns:
        un (2d-array): new u-velocity field
    """
    Nx, Ny = u.shape

    dt_over_delta = dt / DELTA
    pressure_coeff = dt / (2 * RHO * DELTA)
    diffusion_coeff = NU * dt / (DELTA * DELTA)
    force_coeff = dt / RHO

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

            u_center = u[i, j]
            convection = dt_over_delta * (u_center * (u_east - u_west) + v[i, j] * (u_north - u_south))

            # Diffusion (central difference)
            diffusion = diffusion_coeff * (
                (u[i, j+1] - 2.0 * u_center + u[i, j-1]) +
                (u[i+1, j] - 2.0 * u_center + u[i-1, j])
            )

            # Pressure gradient
            pressure_gradient = pressure_coeff * (p[i, j+1] - p[i, j-1])

            # Force term
            force_term_x = force_coeff * Fx[i, j]

            # Update velocity
            un[i, j] = u_center - convection - pressure_gradient + diffusion + force_term_x

    return un

@njit(parallel=CPU_PARALLEL)
def update_y_velocity(u, v, p, dt, Fy, vn):
    """
    Updates the velocity field in the y direction based on the momentum equation. Discretization with first order upwind
    for the convection term. Central differences for the Diffusion term. 

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): p-pressure field
        dt (float): time step
        Fy (2d-array): y-Forcing field
        vn (2d-array): preallocated output array
    Returns:
        vn (2d-array): new v-velocity field
    """
    Nx, Ny = v.shape

    dt_over_delta = dt / DELTA
    pressure_coeff = dt / (2 * RHO * DELTA)
    diffusion_coeff = NU * dt / (DELTA * DELTA)
    force_coeff = dt / RHO

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

            v_center = v[i, j]
            convection = dt_over_delta * (
                u[i, j] * (v_east - v_west) +
                v_center * (v_north - v_south)
            )

            # Diffusion (central difference)
            diffusion = diffusion_coeff * (
                (v[i, j+1] - 2.0 * v_center + v[i, j-1]) +
                (v[i+1, j] - 2.0 * v_center + v[i-1, j])
            )

            # Pressure gradient
            pressure_gradient = pressure_coeff * (p[i+1, j] - p[i-1, j])

            # Force term
            force_term_y = force_coeff * Fy[i, j]

            # Update velocity
            vn[i, j] = v_center - convection - pressure_gradient + diffusion + force_term_y

    return vn

@njit(parallel=CPU_PARALLEL)
def pressure_equation_right_side(u, v, T, b, dt, Fx, Fy):
    """
    computes the right hand side of the pressure poisson equation

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        b (2d-array): empty b field
        dt (float): time step
        Fx (2d-array): x-Forcing field
        Fy (2d-array): y-Forcing field

    Returns:
        b (2d-array): right hand side of pressure poisson euaqtion
    """
    Nx, Ny = u.shape
    half_inv_delta = 0.5 / DELTA
    rho_over_dt = RHO / dt

    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            # Central differences
            du_dx = (u[i, j+1] - u[i, j-1]) * half_inv_delta
            dv_dy = (v[i+1, j] - v[i-1, j]) * half_inv_delta
            du_dy = (u[i+1, j] - u[i-1, j]) * half_inv_delta
            dv_dx = (v[i, j+1] - v[i, j-1]) * half_inv_delta

            divergence = du_dx + dv_dy
            nonlinear = du_dx * du_dx + 2.0 * du_dy * dv_dx + dv_dy * dv_dy

            dFx_dx = (Fx[i, j+1] - Fx[i, j-1]) * half_inv_delta
            dFy_dy = (Fy[i+1, j] - Fy[i-1, j]) * half_inv_delta

            thermal_divergence = EXPANSION_RATE * (T[i,j] - T_REFERENCE)

            b[i, j] = rho_over_dt * (divergence - thermal_divergence) - RHO * (nonlinear + dFx_dx + dFy_dy)

    return b

@njit(parallel=CPU_PARALLEL)
def pressure_poisson(u, v, p, T, p_work, b, dt, Fx, Fy, max_iter=10):
    """
    Solves the pressure Poisson equation iteratively until the change in 
    the pressure field is smaller than a target threshold or the max_iter count is reached.

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field (initial guess)
        p_work (2d-array): work array for Jacobi iterations
        b (2d-array): work array for the right hand side
        dt (float): time step
        Fx (2d-array): x-Forcing field
        Fy (2d-array): y-Forcing field
        dp_target (float): target max change in pressure for convergence
        max_iter (int): maximum NUmber of iterations

    Returns:
        p (2d-array): updated pressure field
        niter (int): NUmber of iterations performed
    """
    Nx, Ny = p.shape
    delta2 = DELTA * DELTA
    b = pressure_equation_right_side(u, v, T, b, dt, Fx, Fy)
    p_old = p
    p_new = p_work
    
    for it in range(max_iter):
        for i in prange(1, Nx-1):
            for j in range(1, Ny-1):
                p_new[i, j] = 0.25 * (
                    p_old[i+1, j] + p_old[i-1, j] +
                    p_old[i, j+1] + p_old[i, j-1] -
                    delta2 * b[i, j]
                )
        
        for j in range(Ny):
            p[0, j] = p[1, j]
            p[Nx - 1, j] = p[Nx - 2, j]

        for i in range(Nx):
            p[i, 0] = p[i, 1]
            p[i, Ny - 1] = p[i, Ny - 2]

        p = Obstacle_BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)
        
        # Swap references
        p_old, p_new = p_new, p_old

    return p_old

# ===============================
# Main
# ===============================

def main():
    # =============================
    # Start
    # =============================
    print("Initialise")
    print("Cell count: ",int(NX*NY))

    if CPU_PARALLEL:
        available_threads = os.cpu_count() or 1
        solver_threads = max(1, available_threads - RESERVE_CPU_CORES_FOR_IO)
        set_num_threads(solver_threads)
        print(f"Numba solver threads: {get_num_threads()} / {available_threads} Cores")

    # ----------Inialise fields----------------
    u = np.ones_like(X).astype(PRECISION)*U_INFLOW
    v = np.ones_like(X).astype(PRECISION)*V_INFLOW
    un = np.empty_like(u)
    vn = np.empty_like(v)
    u_work = np.empty_like(u)
    v_work = np.empty_like(v)
    np.copyto(un, u)
    np.copyto(vn, v)

    p = np.zeros_like(X).astype(PRECISION)
    pressure_work = np.empty_like(p)
    pressure_rhs = np.empty_like(p)

    T = np.ones_like(X).astype(PRECISION)*T_REFERENCE
    temperature_work = np.empty_like(T)

    smoke = np.zeros_like(X).astype(PRECISION)
    smoke_work = np.empty_like(smoke)

    fuel = np.zeros_like(X).astype(PRECISION)
    fuel_work = np.empty_like(fuel)

    Fx = np.zeros_like(p)
    Fy = np.zeros_like(p)

    timestep_row_max_u = np.empty(NX, dtype=PRECISION)
    timestep_row_max_v = np.empty(NX, dtype=PRECISION)
    timestep_row_max_F = np.empty(NX, dtype=PRECISION)
    
    # ----------BCs----------------
    u,v,p,T = BC.outflow_BC(u,v,p,T,"x_low")
    u,v,p,T = BC.outflow_BC(u,v,p,T,"x_high")
    u,v,p,T = BC.no_slip_wall_BC(u,v,p,T,"y_low")
    u,v,p,T = BC.outflow_BC(u,v,p,T,"y_high")

    u, v = Obstacle_BC.obstacle_boundary_conditions_velocity(u, v, obstacle_mask)
    p = Obstacle_BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)
    T = Obstacle_BC.obstacle_boundary_conditions_scalar(T,obstacle_mask,600)
    fuel = Obstacle_BC.obstacle_boundary_conditions_scalar(fuel,obstacle_mask,1)

    # ----------Timing----------------
    start_total_time = time.time() 
    total_loop_time = 0.0
    time_pressure_update = 0.0
    time_velocity_update = 0.0
    time_scalar_update = 0.0
    time_BC = 0.0
    time_obstacle = 0.0
    time_start_copy = 0.0
    time_compute_step = 0.0

    # ----------Compute time step----------------
    t = 0
    dt = Helper_Functions.compute_new_timestep(
        u, v, Fy, RHO, DELTA, NU, CFL_MAX,
        timestep_row_max_u, timestep_row_max_v, timestep_row_max_F
    )

    if dt > 1/OUTPUT_FPS:
        dt = 1/OUTPUT_FPS

    next_output_time = 0
    output_index = 0

    # =============================
    # ASYNC WRITER SETUP
    # =============================
    dataset, u_var, v_var, p_var, T_var, smoke_var, fuel_var, time_var = Output_Functions.initialize_netcdf(
        OUTPATH, NX, NY, X, Y, comp_level=NETCDF_COMPRESSION_LEVEL
    )

    write_queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
    buffer_pool = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
    max_write_queue_fill = 0

    for _ in range(WRITE_QUEUE_SIZE):
        buffer_pool.put({
            'u': np.empty_like(u),
            'v': np.empty_like(v),
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

            (output_index, t, fields) = item

            Output_Functions.write_to_netcdf(
                u_var, v_var, p_var, T_var, smoke_var, fuel_var, time_var,
                output_index, t, fields['u'], fields['v'], fields['p'], fields['T'], fields['smoke'], fields['fuel'], PRECISION
            )

            buffer_pool.put(fields)
            write_queue.task_done()

    # Start writer thread
    writer_thread = threading.Thread(target=writer_thread_func, daemon=True)
    writer_thread.start()

    # =============================
    # Main loop
    # =============================

    print("Start time iteration")
    sys.stdout.write(f"\rProgress: [0%]")

    while t < T_MAX:
        loop_start_time = time.time()
        t0 = time.perf_counter()
        np.copyto(un, u)
        np.copyto(vn, v)
        t1 = time.perf_counter()
        time_start_copy += (t1-t0)

        # ----------Compute buoancy----------------
        Fy = buoancy_approximation(T,Fy,BUOANCY_FACTOR,T_REFERENCE)

        # ----------Pressure poisson----------------
        t0 = time.perf_counter()
        p = pressure_poisson(un, vn, p, T, pressure_work, pressure_rhs, dt, Fx, Fy, MAX_ITER)
        t1 = time.perf_counter()
        time_pressure_update += (t1-t0)

        # ----------Velocity----------------
        t0 = time.perf_counter()
        u = update_x_velocity(un, vn, p, dt, Fx, u_work)
        v = update_y_velocity(un, vn, p, dt, Fy, v_work)
        t1 = time.perf_counter()
        time_velocity_update += (t1-t0)

        # ----------Scalars----------------
        t0 = time.perf_counter()
        T, smoke, fuel = update_scalar_fields(T, smoke, fuel, u, v, dt, temperature_work, smoke_work, fuel_work)
        t1 = time.perf_counter()
        time_scalar_update += (t1-t0)

        # ----------Enforce BCs----------------
        t0 = time.perf_counter()
        u,v,p,T = BC.outflow_BC(u,v,p,T,"x_low")
        u,v,p,T = BC.outflow_BC(u,v,p,T,"x_high")
        u,v,p,T = BC.no_slip_wall_BC(u,v,p,T,"y_low")
        u,v,p,T = BC.outflow_BC(u,v,p,T,"y_high")
        t1 = time.perf_counter()
        time_BC += (t1-t0)

        # ----------Obstacles----------------
        t0 = time.perf_counter()
        u, v = Obstacle_BC.obstacle_boundary_conditions_velocity(u, v, obstacle_mask)
        p = Obstacle_BC.obstacle_boundary_conditions_pressure(p, obstacle_mask)
        T = Obstacle_BC.obstacle_boundary_conditions_scalar(T,obstacle_mask,600)
        fuel = Obstacle_BC.obstacle_boundary_conditions_scalar(fuel,obstacle_mask,1)
        t1 = time.perf_counter()
        time_obstacle += (t1-t0)

        # =============================
        # ASYNC NETCDF WRITING
        # =============================
        while t >= next_output_time:
            fields = buffer_pool.get()
            np.copyto(fields['u'], u)
            np.copyto(fields['v'], v)
            np.copyto(fields['p'], p)
            np.copyto(fields['T'], T)
            np.copyto(fields['smoke'], smoke)
            np.copyto(fields['fuel'], fuel)
            write_queue.put((output_index, t, fields))
            max_write_queue_fill = max(max_write_queue_fill, write_queue.qsize())

            output_index += 1
            next_output_time += OUTPUT_TIME_STEP
            sys.stdout.write(f"\rProgress: [{(t/T_MAX*100):.3f}%]")
            sys.stdout.flush()

            if OUTPUT_STATUS:
                print("#################################################")
                print(f"Simulation time {t} sec")
                CFL = Helper_Functions.compute_CFL(u,v,dt,DELTA)
                print(f"Current dt: {np.round(dt,5)}")
                print(f"CFL-Condition: {np.round(CFL,5)}")

        # ----------Dynamic time step----------------
        t0 = time.perf_counter()
        t += dt

        dt_new = Helper_Functions.compute_new_timestep(
            u, v, Fy, RHO, DELTA, NU, CFL_MAX,
            timestep_row_max_u, timestep_row_max_v, timestep_row_max_F
        )

        # dt limiter
        dt_max_increase = dt * 1.5
        dt_max_decrease = dt * 0.5

        if dt_new > dt_max_increase:
            dt = dt_max_increase
        elif dt_new < dt_max_decrease:
            dt = dt_max_decrease
        else:
            dt = dt_new

        if dt > 1/OUTPUT_FPS:
            dt = 1/OUTPUT_FPS

        t1 = time.perf_counter()
        time_compute_step += (t1-t0)

        # timing
        loop_end_time = time.time()
        total_loop_time += loop_end_time - loop_start_time

    # =============================
    # CLEAN SHUTDOWN
    # =============================
    write_queue.join()
    write_queue.put(None)
    write_queue.join()
    writer_thread.join()

    Output_Functions.close_netcdf(dataset)
    end_total_time = time.time()

    # ----------Conclusion----------------
    print("Simulation finished!")
    print(f"Total runtime: {end_total_time - start_total_time:.4f} seconds")
    print(f"Total time spent in main loop: {total_loop_time:.4f} seconds")
    print(f"Time spend on array copying: {time_start_copy:.4f} seconds")
    print(f"Time spend on pressure: {time_pressure_update:.4f} seconds")
    print(f"Time spend on velocity solve: {time_velocity_update:.4f} seconds")
    print(f"Time spend on scalar update: {time_scalar_update:.4f} seconds")
    print(f"Time spend on boundary condtions: {time_BC:.4f} seconds")
    print(f"Time spend on obstacles: {time_obstacle:.4f} seconds")
    print(f"Time spend on computing next time step: {time_compute_step:.4f} seconds")
    print(f"Max async write queue fill: {max_write_queue_fill}/{WRITE_QUEUE_SIZE}")

main()
