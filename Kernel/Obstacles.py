import numpy as np

def sphere(nx, ny, nz, delta, x_center, y_center, z_center, radius):
    """
    creates a boolean mask for a spherical obstacle directly from grid metadata.

    This avoids materializing full X/Y/Z coordinate volumes during startup.

    Args:
        nx (int): number of cells in x-direction
        ny (int): number of cells in y-direction
        nz (int): number of cells in z-direction
        delta (float): grid spacing
        x_center (float): x-position of the sphere center
        y_center (float): y-position of the sphere center
        z_center (float): z-position of the sphere center
        radius (float): sphere radius
    Returns:
        mask (3d-array): boolean mask of the spherical obstacle
    """
    x2 = (np.arange(nx, dtype=np.float32) * delta - x_center) ** 2
    y2 = (np.arange(ny, dtype=np.float32) * delta - y_center) ** 2
    z2 = (np.arange(nz, dtype=np.float32) * delta - z_center) ** 2
    return (x2[:, None, None] + y2[None, :, None] + z2[None, None, :]) <= radius ** 2
