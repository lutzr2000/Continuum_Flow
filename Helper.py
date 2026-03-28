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

def compute_divergence(u, v, delta):
    """
    computes divergence of a 2-d velocity field without the edges

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        delta (float): space resolution

    Returns:
        div (2d-array): divergence
        div_l1 (2d-array): L1 norm of divergence
    """
    div = np.zeros_like(u)
    
    # central difference
    div[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2*delta) \
                     + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2*delta)
    
    div_l1 = np.sum(np.abs(div[1:-1, 1:-1])) / ((div.shape[0]-2)*(div.shape[1]-2))
    return div, div_l1