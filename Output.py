import netCDF4 as nc
import numpy as np

def initialize_netcdf(filename, nx, ny, X, Y):
    """
    Erstellt eine leere NetCDF Datei mit den Variablen u, v, p.
    Kompression aktiviert, Datentyp float32.
    """
    dataset = nc.Dataset(filename, 'w', format='NETCDF4')

    # Dimensionen
    dataset.createDimension('x', nx)
    dataset.createDimension('y', ny)
    dataset.createDimension('time', None)  # unlimited

    # Koordinaten
    x_var = dataset.createVariable('x', 'f4', ('x',))
    y_var = dataset.createVariable('y', 'f4', ('y',))
    x_var[:] = X[0, :]
    y_var[:] = Y[:, 0]

    # Variablen mit Kompression
    u_var = dataset.createVariable('u', 'f4', ('time', 'y', 'x'), zlib=True, complevel=4)
    v_var = dataset.createVariable('v', 'f4', ('time', 'y', 'x'), zlib=True, complevel=4)
    p_var = dataset.createVariable('p', 'f4', ('time', 'y', 'x'), zlib=True, complevel=4)

    # Attribute
    u_var.units = 'm/s'
    v_var.units = 'm/s'
    p_var.units = 'Pa'

    return dataset, u_var, v_var, p_var

def write_to_netcdf(u_var, v_var, p_var, timestep, u, v, p):
    """
    Schreibt einen Zeitschritt in die NetCDF Variablen.
    """
    u_var[timestep, :, :] = u.astype(np.float32)
    v_var[timestep, :, :] = v.astype(np.float32)
    p_var[timestep, :, :] = p.astype(np.float32)

def close_netcdf(dataset):
    """
    Schließt die NetCDF Datei.
    """
    dataset.close()