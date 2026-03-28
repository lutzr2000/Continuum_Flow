import numpy as np

def compute_CFL(u,v,dt,delta):
    """
    computes the CFL condition

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        dt (float): time step size
        delta (float): space resolution

    Returns:
        b (2d-array): right hand side of pressure poisson euaqtion
    """
    CFL_x = np.max(np.abs(u)) * dt / delta
    CFL_y = np.max(np.abs(v)) * dt / delta

    CFL = max(CFL_x, CFL_y)
    return CFL