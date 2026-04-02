def sphere(X, Y, Z, x_center, y_center, z_center, radius):
    """
    Creates a boolean mask for a spherical obstacle.
    """
    return (
        (X - x_center) ** 2 +
        (Y - y_center) ** 2 +
        (Z - z_center) ** 2
    ) <= radius ** 2


def cylinder_z(X, Y, x_center, y_center, radius):
    """
    Creates a boolean mask for a cylinder aligned with the z-axis.
    """
    return (X - x_center) ** 2 + (Y - y_center) ** 2 <= radius ** 2
