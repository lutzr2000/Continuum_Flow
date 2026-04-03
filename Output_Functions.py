import netCDF4 as nc


def initialize_netcdf(filepath, nx, ny, nz, x, y, z, comp_level=0):
    """
    creates an empty 3D netcdf file and initializes the output variables.

    Args:
        filepath (str): path to the netcdf output file
        nx (int): number of cells in x-direction
        ny (int): number of cells in y-direction
        nz (int): number of cells in z-direction
        x (1d-array): x-coordinate values
        y (1d-array): y-coordinate values
        z (1d-array): z-coordinate values
        comp_level (int): netcdf compression level
    Returns:
        dataset (netcdf-dataset): opened netcdf dataset
        u_var (netcdf-variable): output variable for x-velocity
        v_var (netcdf-variable): output variable for y-velocity
        w_var (netcdf-variable): output variable for z-velocity
        p_var (netcdf-variable): output variable for pressure
        T_var (netcdf-variable): output variable for temperature
        smoke_var (netcdf-variable): output variable for smoke
        fuel_var (netcdf-variable): output variable for fuel
        time_var (netcdf-variable): output variable for time
    """
    dataset = nc.Dataset(filepath, 'w', format='NETCDF4')

    dataset.createDimension('x', nx)
    dataset.createDimension('y', ny)
    dataset.createDimension('z', nz)
    dataset.createDimension('time', None)

    x_var = dataset.createVariable('x', 'f4', ('x',))
    y_var = dataset.createVariable('y', 'f4', ('y',))
    z_var = dataset.createVariable('z', 'f4', ('z',))
    x_var[:] = x
    y_var[:] = y
    z_var[:] = z

    compression_enabled = comp_level > 0
    variable_kwargs = {
        'zlib': compression_enabled,
        'chunksizes': (1, nz, ny, nx),
    }
    if compression_enabled:
        variable_kwargs['complevel'] = comp_level

    u_var = dataset.createVariable('u', 'f4', ('time', 'z', 'y', 'x'), **variable_kwargs)
    v_var = dataset.createVariable('v', 'f4', ('time', 'z', 'y', 'x'), **variable_kwargs)
    w_var = dataset.createVariable('w', 'f4', ('time', 'z', 'y', 'x'), **variable_kwargs)
    p_var = dataset.createVariable('p', 'f4', ('time', 'z', 'y', 'x'), **variable_kwargs)
    T_var = dataset.createVariable('T', 'f4', ('time', 'z', 'y', 'x'), **variable_kwargs)
    smoke_var = dataset.createVariable('smoke', 'f4', ('time', 'z', 'y', 'x'), **variable_kwargs)
    fuel_var = dataset.createVariable('fuel', 'f4', ('time', 'z', 'y', 'x'), **variable_kwargs)

    time_var = dataset.createVariable('time', 'f8', ('time',))
    time_var.units = 'seconds since 2026-04-01 00:00:00'
    time_var.calendar = 'standard'

    u_var.units = 'm/s'
    v_var.units = 'm/s'
    w_var.units = 'm/s'
    p_var.units = 'Pa'
    T_var.units = 'K'
    smoke_var.units = '1'
    fuel_var.units = '1'

    return dataset, u_var, v_var, w_var, p_var, T_var, smoke_var, fuel_var, time_var


def write_to_netcdf(u_var, v_var, w_var, p_var, T_var, smoke_var, fuel_var,
                    time_var, timestep, time_value, u, v, w, p, T, smoke, fuel):
    """
    writes the current timestep data into the netcdf file.

    Args:
        u_var (netcdf-variable): output variable for x-velocity
        v_var (netcdf-variable): output variable for y-velocity
        w_var (netcdf-variable): output variable for z-velocity
        p_var (netcdf-variable): output variable for pressure
        T_var (netcdf-variable): output variable for temperature
        smoke_var (netcdf-variable): output variable for smoke
        fuel_var (netcdf-variable): output variable for fuel
        time_var (netcdf-variable): output variable for time
        timestep (int): output timestep index
        time_value (float): physical simulation time
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        p (3d-array): pressure field
        T (3d-array): temperature field
        smoke (3d-array): smoke field
        fuel (3d-array): fuel field
    Returns:
        None
    """
    time_var[timestep] = time_value
    u_var[timestep, :, :, :] = u.transpose(2, 1, 0)
    v_var[timestep, :, :, :] = v.transpose(2, 1, 0)
    w_var[timestep, :, :, :] = w.transpose(2, 1, 0)
    p_var[timestep, :, :, :] = p.transpose(2, 1, 0)
    T_var[timestep, :, :, :] = T.transpose(2, 1, 0)
    smoke_var[timestep, :, :, :] = smoke.transpose(2, 1, 0)
    fuel_var[timestep, :, :, :] = fuel.transpose(2, 1, 0)


def close_netcdf(dataset):
    """
    closes the netcdf dataset.

    Args:
        dataset (netcdf-dataset): opened netcdf dataset
    Returns:
        None
    """
    dataset.close()
