import numpy as np
from Boundary_Conditions import pressure_BC,velocity_BC
from Helper import compute_CFL

rho = 1
nu = 0.001
dt = 0.001
delta = 0.01

nx = 512
ny = 128
t_max = 10
nt = int(t_max/dt)
 
x = np.linspace(0,(nx-1)*delta,nx)
y = np.linspace(0,(ny-1)*delta,ny)
X, Y = np.meshgrid(x, y)

u_initial = np.zeros_like(X)
v_initial = np.zeros_like(X)
p_initial = np.zeros_like(X)

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

def update_x_velocity(u,v,p):
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
    fe1, fe2 = compute_F(un)       
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
   
    un[1:-1, 1:-1] = u[1:-1, 1:-1] - convection - pressure_gradient + diffusion
    return un

def update_y_velocity(u, v, p):
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

    vn[1:-1, 1:-1] = v[1:-1, 1:-1]- convection- pressure_gradient+ diffusion

    return vn

def pressure_equation_right_side(u,v,b):
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
    b[1:-1, 1:-1] = rho * ( (1 / dt) * divergence - nonlinear )

    return b

def pressure_poisson_l1norm(u, v, p, l1norm_target, max_iter=50):
    """
    Solves the pressure Poisson equation iteratively using L1 norm convergence criterion.

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field (initial guess)
        l1norm_target (float): target L1 norm for convergence
        max_iter (int): maximum number of iterations

    Returns:
        p (2d-array): updated pressure field
        niter (int): number of iterations performed
    """
    b = np.zeros_like(p)
    b = pressure_equation_right_side(u, v, b)
    l1norm = 1.0
    niter = 0

    while l1norm > l1norm_target and niter < max_iter:
        niter += 1
        pn = p.copy()

        laplace_x = pn[1:-1, 2:] + pn[1:-1, 0:-2]
        laplace_y = pn[2:, 1:-1] + pn[0:-2, 1:-1]

        p[1:-1, 1:-1] = (laplace_x * delta**2 + laplace_y * delta**2) / (2 * (delta**2 + delta**2)) - (delta**2 * delta**2) / (2 * (delta**2 + delta**2)) * b[1:-1, 1:-1]

        p = pressure_BC(p)

        l1norm = np.sum(np.abs(p - pn)) / (np.sum(np.abs(pn)) + 1e-8)

    return p, niter

def apply_obstacle(u, v, X, Y, x0, y0, radius):
    """
    Setzt die Geschwindigkeit innerhalb eines kreisförmigen Hindernisses auf 0 (no-slip)
    
    Parameters:
        u, v : np.ndarray
            Geschwindigkeit
        X, Y : np.ndarray
            Gitterkoordinaten
        x0, y0 : float
            Zentrum des Kreises
        radius : float
            Radius des Kreises
    """
    mask = (X - x0)**2 + (Y - y0)**2 <= radius**2
    u[mask] = 0.0
    v[mask] = 0.0
    return u, v

def main():
    print("Initialise")
    u = u_initial.copy()
    v = v_initial.copy()
    p = p_initial.copy()

    plot = True

    ###########################################################################
    if plot == True:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(figsize=(7,5))

        # Geschwindigkeit als Betrag initial
        vmag = np.sqrt(u**2 + v**2)
        
        # feste Farbskala von Anfang an
        vmin, vmax = 0, 4.0  # hier z.B. 0 bis 2 m/s, anpassen je nach erwarteter Geschwindigkeit

        # imshow für animierte Kontur
        im = ax.imshow(vmag, origin='lower', cmap='viridis',
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    vmin=vmin, vmax=vmax)  # feste Farbskala
        cbar = plt.colorbar(im, ax=ax, label='Velocity magnitude')  # einmalig

        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ###########################################################################
    print("Start time itteration")
    for n in range(nt):
        print(f"Timestep {n} of {nt} steps")
        un = u.copy()
        vn = v.copy()
        pn = p.copy()

        p,niter = pressure_poisson_l1norm(un, vn, pn, 1e-3)
        print(f"Number of pressure itterations: {niter}")
        u = update_x_velocity(un, vn, p)
        v = update_y_velocity(un, vn, p)
        u, v = velocity_BC(u, v)

        # circle_collider
        x0 = 0.2 * x.max()       # center in x
        y0 = 0.5 * y.max()       # center in y
        radius = 0.3           # R
        u, v = apply_obstacle(u, v, X, Y, x0, y0, radius)

        CFL = compute_CFL(u,v,dt,delta)
        print(f"CFL-Condition: {np.round(CFL,5)}")
        ###########################################################################
        if plot == True:
            # Geschwindigkeit Betrag
            vmag = np.sqrt(u**2 + v**2)

            # Update imshow-Daten
            im.set_data(vmag)

            ax.set_title(f"Time {n*dt:.3f} s")
            plt.pause(0.0001)
        ###########################################################################
    if plot == True:
        plt.ioff()
        plt.show()

main()