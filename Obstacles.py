from numba import njit, prange

def circle(X,Y,x_center,y_center,R):
    """
    Creates a boolean mask for a circular obstacle
    
    Args:
        X, Y: meshgrid arrays
        x_center, y_center: center of circle
        R: radius
        
    Returns:
        mask (2D boolean array)
    """
    return (X - x_center)**2 + (Y - y_center)**2 <= R**2

def rectangle(X, Y, x_min, x_max, y_min, y_max):
    """
    Creates a boolean mask for a rectangular obstacle
    
    Args:
        X, Y: meshgrid arrays
        x_min, x_max: horizontal boundaries
        y_min, y_max: vertical boundaries
        
    Returns:
        mask (2D boolean array)
    """
    return (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

def triangle(X, Y, v1, v2, v3):
    """
    Creates a boolean mask for a triangular obstacle using barycentric coordinates
    
    Args:
        X, Y: meshgrid arrays
        v1, v2, v3: vertices of the triangle [(x1,y1), (x2,y2), (x3,y3)]
        
    Returns:
        mask (2D boolean array)
    """
    x = X
    y = Y
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    # Barycentric coordinates
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    c = 1 - a - b
    
    return (a >= 0) & (b >= 0) & (c >= 0)