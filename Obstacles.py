def sphere(X, Y, Z, x_center, y_center, z_center, radius):
    """
    creates a boolean mask for a spherical obstacle.

    Args:
        X (3d-array): x-coordinate grid
        Y (3d-array): y-coordinate grid
        Z (3d-array): z-coordinate grid
        x_center (float): x-position of the sphere center
        y_center (float): y-position of the sphere center
        z_center (float): z-position of the sphere center
        radius (float): sphere radius
    Returns:
        mask (3d-array): boolean mask of the spherical obstacle
    """
    return (
        (X - x_center) ** 2 +
        (Y - y_center) ** 2 +
        (Z - z_center) ** 2
    ) <= radius ** 2


def cylinder_z(X, Y, x_center, y_center, radius):
    """
    creates a boolean mask for a cylinder aligned with the z-axis.

    Args:
        X (2d-array): x-coordinate grid
        Y (2d-array): y-coordinate grid
        x_center (float): x-position of the cylinder center
        y_center (float): y-position of the cylinder center
        radius (float): cylinder radius
    Returns:
        mask (2d-array): boolean mask of the cylindrical obstacle
    """
    return (X - x_center) ** 2 + (Y - y_center) ** 2 <= radius ** 2
