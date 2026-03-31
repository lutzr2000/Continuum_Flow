import netCDF4 as nc
import numpy as np

def initialize_netcdf(filepath, nx, ny, X, Y, comp_level=2):
    """
    Create an empty netcdf file.

    Args:
        filepath (string): path to save the file
        nx (int): numver of x values
        ny (int): numver of y values
        X (2d-array): meshgrid of x values
        Y (2d-array): meshgrid of y values
        comp_level (int,optional): compression level

    Returns:
        dataset (netcdf.dataset): dataset to write into
        u_var (netcdf.variable): u-velocity variable in netcdf file
        v_var (netcdf.variable): v-velocity variable in netcdf file
        p_var (netcdf.variable): p-velocity variable in netcdf file
    """
    dataset = nc.Dataset(filepath, 'w', format='NETCDF4')

    # Dimension
    dataset.createDimension('x', nx)
    dataset.createDimension('y', ny)
    dataset.createDimension('time', None)  # unlimited

    # Cooridnates
    x_var = dataset.createVariable('x', 'f4', ('x',))
    y_var = dataset.createVariable('y', 'f4', ('y',))
    x_var[:] = X[0, :]
    y_var[:] = Y[:, 0]

    # Variables with compression
    u_var = dataset.createVariable('u', 'f4', ('time', 'y', 'x'), zlib=True, complevel=comp_level)
    v_var = dataset.createVariable('v', 'f4', ('time', 'y', 'x'), zlib=True, complevel=comp_level)
    p_var = dataset.createVariable('p', 'f4', ('time', 'y', 'x'), zlib=True, complevel=comp_level)

    # Attributes
    u_var.units = 'm/s'
    v_var.units = 'm/s'
    p_var.units = 'Pa'

    return dataset, u_var, v_var, p_var

def write_to_netcdf(u_var, v_var, p_var, timestep, u, v, p, precision):
    """
    Writes the current timestep into the datefile created by initialize_netcdf()

    Args:
        u_var (netcdf.variable): u-velocity variable in netcdf file
        v_var (netcdf.variable): v-velocity variable in netcdf file
        p_var (netcdf.variable): p-velocity variable in netcdf file
        timestep (float): time step
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field
        precision (dtype): precision used for writing

    Returns:
        None
    """
    u_var[timestep, :, :] = u.astype(precision)
    v_var[timestep, :, :] = v.astype(precision)
    p_var[timestep, :, :] = p.astype(precision)

def close_netcdf(dataset):
    """
    Closes the netcdf dataset.
    """
    dataset.close()