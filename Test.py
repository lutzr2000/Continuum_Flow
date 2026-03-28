import numpy as np


rho = 1
nu = 0.01
dt = 0.001
delta = 0.01

nx = 512
ny = 128
t_max = 10
nt = int(t_max/dt)
 
x = np.linspace(0,(nx-1)*delta,nx)
y = np.linspace(0,(ny-1)*delta,ny)
X, Y = np.meshgrid(x, y)


u_initial = np.ones_like(X)*0.1
v_initial = np.zeros_like(X)
p_initial = np.zeros_like(X)

def compute_F(c):
    denom = abs(c) + 1e-6
    pos_part = np.maximum(c/denom, 0)
    neg_part = np.maximum(-c/denom, 0)
    return pos_part, neg_part

def update_x_velocity(u,v,p):
    un = u.copy()
    fe1, fe2 = compute_F(un)       
    fw1, fw2 = fe1, fe2
    ue = u[1:-1, 1:-1] * fe1[1:-1, 1:-1] + u[1:-1, 2:] * fe2[1:-1, 1:-1]     
    uw = u[1:-1, 0:-2] * fw1[1:-1, 1:-1] + u[1:-1, 1:-1]* fw2[1:-1, 1:-1]

    fnorth1, fnorth2 = compute_F(v)       
    fs1, fs2 = fnorth1, fnorth2
    unorth = u[1:-1, 1:-1] * fnorth1[1:-1, 1:-1] + u[2:, 1:-1] * fnorth2[1:-1, 1:-1]     
    us = u[0:-2, 1:-1] * fs1[1:-1, 1:-1] + u[1:-1, 1:-1]* fs2[1:-1, 1:-1]
   
    un[1:-1, 1:-1] = (u[1:-1, 1:-1]-
                    u[1:-1, 1:-1] * dt / delta *
                    (ue - uw) -
                    v[1:-1, 1:-1] * dt / delta *
                    (unorth - us) -
                    dt / (2 * rho * delta) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                    nu * (dt / delta**2 *
                    (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2]) +
                    dt / delta**2 *
                    (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1])))
    return un

def update_y_velocity(u,v,p):
    vn = v.copy()
    fe1, fe2 = compute_F(u)       
    fw1, fw2 = fe1, fe2
    ve = v[1:-1, 1:-1] * fe1[1:-1, 1:-1] + v[1:-1, 2:] * fe2[1:-1, 1:-1]     
    vw = v[1:-1, 0:-2] * fw1[1:-1, 1:-1] + v[1:-1, 1:-1]* fw2[1:-1, 1:-1]

    fnorth1, fnorth2 = compute_F(v)       
    fs1, fs2 = fnorth1, fnorth2
    vnorth = v[1:-1, 1:-1] * fnorth1[1:-1, 1:-1] + v[2:, 1:-1] * fnorth2[1:-1, 1:-1]     
    vs = v[0:-2, 1:-1] * fs1[1:-1, 1:-1] + v[1:-1, 1:-1]* fs2[1:-1, 1:-1]
    
    vn[1:-1,1:-1] = (v[1:-1, 1:-1] -
                    u[1:-1, 1:-1] * dt / delta *
                    (ve - vw) -
                    v[1:-1, 1:-1] * dt / delta *
                    (vnorth - vs) -
                    dt / (2 * rho * delta) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                    nu * (dt / delta**2 *
                    (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, 0:-2]) +
                    dt / delta**2 *
                    (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[0:-2, 1:-1])))
    return vn

def pressure_equation_right_side(u,v,b):
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * delta) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * delta)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (delta))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * delta) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * delta))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * delta))**2))
    return b

def pressure_poisson(u,v,p, nit=50):
    b = np.zeros_like(p)
    b = pressure_equation_right_side(u,v,b)
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * delta**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * delta**2) /
                          (2 * (delta**2 + delta**2)) -
                          delta**2 * delta**2 / (2 * (delta**2 + delta**2)) * 
                          b[1:-1,1:-1])
        p = pressure_BC(p)
    return p

def pressure_BC(p):
    p[0, :] = p[1, :]   # dp/dy = 0
    p[-1, :] = p[-2, :] # dp/dy = 0
    p[:, 0] = p[:, 1]
    p[:, -1] = 0
    return p

def velocity_BC(u, v):
    u[0, :] = 0       # y = 0
    u[-1, :] = 0      # y = H
    v[0, :] = 0
    v[-1, :] = 0
    u[:, 0] = 1
    v[:, 0] = 0
    u[:, -1] = u[:, -2]  # ∂u/∂x = 0
    v[:, -1] = v[:, -2]  # ∂v/∂x = 0
    return u, v

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
    u = u_initial.copy()
    v = v_initial.copy()
    p = p_initial.copy()

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

    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        pn = p.copy()

        p = pressure_poisson(un, vn, pn)
        u = update_x_velocity(un, vn, p)
        v = update_y_velocity(un, vn, p)
        u, v = velocity_BC(u, v)

        # Hindernis in der Mitte des Kanals
        x0 = 0.2 * x.max()       # Mitte in x
        y0 = 0.5 * y.max()       # Mitte in y
        radius = 0.3           # Radius des Zylinders
        u, v = apply_obstacle(u, v, X, Y, x0, y0, radius)

        # Geschwindigkeit Betrag
        vmag = np.sqrt(u**2 + v**2)

        # Update imshow-Daten
        im.set_data(vmag)

        ax.set_title(f"Time {n*dt:.3f} s")
        plt.pause(0.0001)

    plt.ioff()
    plt.show()
main()