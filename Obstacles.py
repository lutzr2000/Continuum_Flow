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
