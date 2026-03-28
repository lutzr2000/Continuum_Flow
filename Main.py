import matplotlib.pyplot as plt
import numpy as np
import time
from Boundary_Conditions import pressure_BC,velocity_BC
from Helper import compute_CFL,compute_divergence
from numba import njit, prange
from scipy.sparse import diags, kron, eye

# fluid
rho = 1.225
nu = 1.5e-5

# time
t_max = 10
dt = 0.002

# solver
div_target = 0.001
max_iter = 500
precision = np.float32

# resolution
delta = 0.01
nx = 1024
ny = 256
nt = int(t_max/dt)
x = np.linspace(0,(nx-1)*delta,nx)
y = np.linspace(0,(ny-1)*delta,ny)
X, Y = np.meshgrid(x, y)

###################################################
import numpy as np
from scipy.ndimage import gaussian_filter

noise = np.random.randn(*X.shape)
smooth_noise = gaussian_filter(noise, sigma=4)  # sigma = Glättungsstärke
###################################################

# initial conditions
u_initial = np.ones_like(X).astype(precision)*2+smooth_noise*10
v_initial = np.zeros_like(X).astype(precision)
p_initial = np.zeros_like(X).astype(precision)

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

@njit(parallel=True)
def pressure_poisson(u, v, p, Fx=None, Fy=None, div_target=1e-5, max_iter=500):
    """
    Solves the pressure Poisson equation iteratively until the velocity field
    is sufficiently divergence-free.

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field (initial guess)
        div_target (float): target divergence for convergence
        max_iter (int): maximum number of iterations

    Returns:
        p (2d-array): updated pressure field
        niter (int): number of iterations performed
    """
    b = np.zeros_like(p)
    b = pressure_equation_right_side(u, v, b, Fx, Fy)

    niter = 0
    divergence_l1 = 1.0

    while divergence_l1 > div_target and niter < max_iter:
        niter += 1
        laplace_x = p[1:-1, 2:] + p[1:-1, 0:-2]
        laplace_y = p[2:, 1:-1] + p[0:-2, 1:-1]

        p[1:-1, 1:-1] = 0.25 * (laplace_x + laplace_y - delta**2 * b[1:-1, 1:-1])
        p = pressure_BC(p)
        
        du_dx = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2*delta)
        dv_dy = (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2*delta)
        divergence = du_dx + dv_dy

        divergence_l1 = np.mean(np.abs(divergence))
    
    return p, niter

def main():
    print("Initialise")
    u = u_initial.copy()
    v = v_initial.copy()
    p = p_initial.copy()

    plot = True
    n_plot = 1

    ###########################################################################
    if plot:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,10))

        # Geschwindigkeit initial
        vmag = np.sqrt(u**2 + v**2)
        vmin_v, vmax_v = -5, 5  # Farbskala für Geschwindigkeit
        im_vel = ax1.imshow(vmag, origin='lower', cmap='viridis',
                            extent=[x.min(), x.max(), y.min(), y.max()],
                            vmin=vmin_v, vmax=vmax_v)
        cbar1 = plt.colorbar(im_vel, ax=ax1, label='Velocity magnitude')
        ax1.set_title("Velocity magnitude")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        # Druck initial
        pmin, pmax = -100.0, 100.0  # Beispielwerte für Farbskala
        im_p = ax2.imshow(p, origin='lower', cmap='coolwarm',
                          extent=[x.min(), x.max(), y.min(), y.max()],
                          vmin=pmin, vmax=pmax)
        cbar2 = plt.colorbar(im_p, ax=ax2, label='Pressure')
        ax2.set_title("Pressure")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
    ###########################################################################
    print("Start time itteration")
    for n in range(nt):
        start_time = time.time()

        Fx=np.zeros_like(p)
        Fy=np.zeros_like(p)

        un = u.copy()
        vn = v.copy()
        pn = p.copy()

        p,niter = pressure_poisson(un, vn, pn, Fx, Fy, div_target, max_iter)
        u = update_x_velocity(un, vn, p, Fx, Fy)
        v = update_y_velocity(un, vn, p, Fx, Fy)
        u, v = velocity_BC(u, v)

        CFL = compute_CFL(u,v,dt,delta)
        _ , div_l1 = compute_divergence(u,v,delta)
        end_time = time.time()

        ###########################################################################
        if plot and n % n_plot == 0:  
            vmag = np.sqrt(u**2 + v**2)
            im_vel.set_data(vmag)
            im_vel.set_clim(vmin_v, vmax_v)
            ax1.set_title(f"Velocity magnitude, Time = {n*dt:.3f} s")

            im_p.set_data(p)
            im_p.set_clim(pmin, pmax)
            ax2.set_title(f"Pressure, Time = {n*dt:.3f} s")

            plt.pause(0.000001)  
        ###########################################################################
        # Output
        print("#################################################")
        print(f"Timestep {n} of {nt} steps")
        print(f"CFL-Condition: {np.round(CFL,5)}")
        print(f"Number of pressure itterations: {niter}")
        print(f"Divergence of velocity field: {div_l1}")
        print(f"Time per timestep: {end_time - start_time:.4f} s")

main()