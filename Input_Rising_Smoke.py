import os
import numpy as np


CONFIG = {
    "fluid": {
        "RHO": 1.225,
        "NU": 1.81e-5,
        "NU_TEMPERATURE": 0.01,
        "NU_SMOKE": 0.001,
        "NU_FUEL": 0.001,
        "TEMPERATURE_DISSIPATION_RATE": 0.1,
        "TEMPERATURE_PRODUCTION_RATE": 1.0,
        "SMOKE_DISSIPATION_RATE": 0.1,
        "SMOKE_PRODUCTION_RATE": 1.0,
        "FUEL_BURN_RATE": 0.1,
        "FUEL_IGNITION_TEMPERATURE": 500.0,
        "T_REFERENCE": 300.0,
        "BUOANCY_FACTOR": 1 / 300.0,
        "EXPANSION_RATE": 0.003,
    },
    "time": {
        "T_MAX": 30.0,
        "CFL_MAX": 0.8,
    },
    "solver": {
        "MAX_ITER": 4,
        "PRECISION": np.float32,
        "CPU_COUNT": 28,
    },
    "resolution": {
        "DELTA": 0.05,
        "NX": 256,
        "NY": 256,
        "NZ": 256,
    },
    "output": {
        "OUTPUT_FPS": 24,
        "PRINT_FREQUENCY": 100,
        "OUTPUT_STATUS": False,
        "WRITE_QUEUE_SIZE": 512,
        "OUTPATH": r"C:\Blenderzeug\BlenderCFD\Test",
        "OUTPUT_VARIABLES": ["smoke"],
    },
    "boundary_conditions": {
        "BC_CONFIG": {
            "x_low": {"type": "outflow"},
            "x_high": {"type": "outflow"},
            "y_low": {"type": "outflow"},
            "y_high": {"type": "outflow"},
            "z_low": {"type": "no_slip_wall"},
            "z_high": {"type": "no_slip_wall"},
        },
        "U_INFLOW": 0.0,
        "V_INFLOW": 0.0,
        "W_INFLOW": 0.0,
    },
    "obstacle": {
        "shape": "sphere",
        "solid": False,
        "initial_temperature": 600.0,
        "initial_smoke": 100.0,
        "initial_fuel": 0.0,
        "sphere": {
            "x_factor": 0.5,
            "y_factor": 0.5,
            "z_factor": 0.05,
            "radius": 0.6,
        },
    },
}

