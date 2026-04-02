import netCDF4 as nc
import numpy as np

def initialize_netcdf(filepath, nx, ny, X, Y, comp_level=0):
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
        T_var (netcdf.variable): temperature variable in netcdf file
        smoke_var (netcdf.variable): smoke variable in netcdf file
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

    compression_enabled = comp_level > 0
    variable_kwargs = {
        'zlib': compression_enabled,
        'chunksizes': (1, ny, nx),
    }
    if compression_enabled:
        variable_kwargs['complevel'] = comp_level

    # Variables with optional compression
    u_var = dataset.createVariable('u', 'f4', ('time', 'y', 'x'), **variable_kwargs)
    v_var = dataset.createVariable('v', 'f4', ('time', 'y', 'x'), **variable_kwargs)
    p_var = dataset.createVariable('p', 'f4', ('time', 'y', 'x'), **variable_kwargs)
    T_var = dataset.createVariable('T', 'f4', ('time', 'y', 'x'), **variable_kwargs)
    smoke_var = dataset.createVariable('smoke', 'f4', ('time', 'y', 'x'), **variable_kwargs)

    # time for paraview
    time_var = dataset.createVariable('time', 'f8', ('time',))
    time_var.units = "seconds since 2026-04-01 00:00:00"
    time_var.calendar = "standard"

    # Attributes
    u_var.units = 'm/s'
    v_var.units = 'm/s'
    p_var.units = 'Pa'
    T_var.units = '°K'

    smoke_var.units = '1'

    return dataset, u_var, v_var, p_var, T_var, smoke_var, time_var

def write_to_netcdf(u_var, v_var, p_var, T_var, smoke_var, time_var, timestep, time_value, u, v, p, T, smoke, precision):
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
        T (2d-array): temperature field
        smoke (2d-array): smoke field
        precision (dtype): precision used for writing

    Returns:
        None
    """
    time_var[timestep] = time_value
    u_var[timestep, :, :] = u
    v_var[timestep, :, :] = v
    p_var[timestep, :, :] = p
    T_var[timestep, :, :] = T
    smoke_var[timestep, :, :] = smoke

def close_netcdf(dataset):
    """
    Closes the netcdf dataset.
    """
    dataset.close()
