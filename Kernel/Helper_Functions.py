import numpy as np
from numba import cuda

import Boundary_Conditions as BC
import Obstacle_Boundary_Conditions as Obstacle_BC
import Source_Boundary_Conditions as Source_BC

REDUCTION_THREADS_PER_BLOCK = 256
THREADS_PER_BLOCK_3D = (8, 8, 8)
THREADS_PER_BLOCK_2D = (4, 4)
BOUNDARY_FACE_NAMES = ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high")


def build_boundary_config(domain_cfg):
    """Build BC config and extract the first inflow velocity from boundary faces."""
    zero_velocity = (0.0, 0.0, 0.0)
    boundary_cfg = domain_cfg.get("boundary_conditions", {})
    bc_config = {}
    inflow_velocity = zero_velocity

    for face_name in BOUNDARY_FACE_NAMES:
        face_cfg = boundary_cfg.get(face_name, {})
        bc_type = face_cfg.get("type", "OUTFLOW")
        bc_config[face_name] = {
            "type": bc_type,
        }

        if bc_type == "INFLOW" and inflow_velocity == zero_velocity:
            velocity = face_cfg.get("velocity", zero_velocity)
            if len(velocity) >= 3:
                inflow_velocity = (
                    float(velocity[0]),
                    float(velocity[1]),
                    float(velocity[2]),
                )

    return bc_config, inflow_velocity


def collect_output_variables(output_cfg):
    """Translate exported output field toggles to kernel field identifiers."""
    field_mapping = {
        "u": "u",
        "v": "v",
        "w": "w",
        "p": "p",
        "t": "T",
        "smoke": "smoke",
        "fuel": "fuel",
        "flame": "flame",
    }

    enabled_fields = []
    for field_name, is_enabled in output_cfg.get("fields", {}).items():
        if is_enabled and field_name in field_mapping:
            enabled_fields.append(field_mapping[field_name])
    return enabled_fields


def apply_config(config, blender_python_exe, vdb_writer_script):
    """Extract kernel settings and persistent data from the exported config."""
    simulations = config.get("simulations")
    simulation_cfg = simulations[0]
    domain_cfg = simulation_cfg.get("domain")
    physics_cfg = simulation_cfg.get("physics")
    output_cfg = simulation_cfg.get("outputs", [None])[0]
    obstacle_entries = simulation_cfg.get("obstacles", [])
    source_entries = simulation_cfg.get("sources", [])

    bc_config, inflow_velocity = build_boundary_config(domain_cfg)
    obstacle_data = Obstacle_BC.build_obstacle_data(domain_cfg, obstacle_entries)
    source_data = Source_BC.build_source_data(domain_cfg, source_entries)

    BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D
    BC.THREADS_PER_BLOCK_2D = THREADS_PER_BLOCK_2D
    Obstacle_BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D
    Source_BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D

    return {
        "RHO": float(physics_cfg["fluid"]["density"]),
        "NU": float(physics_cfg["fluid"]["viscosity"]),
        "NU_TEMPERATURE": float(physics_cfg["temperature"]["diffusion"]),
        "NU_SMOKE": float(physics_cfg["smoke"]["diffusion"]),
        "NU_FUEL": float(physics_cfg["fuel"]["diffusion"]),
        "TEMPERATURE_DISSIPATION_RATE": float(physics_cfg["temperature"]["dissipation"]),
        "TEMPERATURE_PRODUCTION_RATE": 1.0,
        "SMOKE_DISSIPATION_RATE": float(physics_cfg["smoke"]["dissipation"]),
        "SMOKE_PRODUCTION_RATE": 1.0,
        "FUEL_BURN_RATE": float(physics_cfg["fuel"]["burn_rate"]),
        "FUEL_IGNITION_TEMPERATURE": float(physics_cfg["fuel"]["ignition_temperature"]),
        "T_REFERENCE": float(physics_cfg["temperature"]["reference_temperature"]),
        "BUOANCY_FACTOR": float(physics_cfg["temperature"]["buoyancy"]),
        "EXPANSION_RATE": float(physics_cfg["temperature"]["expansion_rate"]),
        "T_MAX": float(simulation_cfg["settings"]["simulation_length"]),
        "CFL_MAX": float(simulation_cfg["settings"]["cfl"]),
        "MAX_ITER": int(simulation_cfg["settings"]["iterations"]),
        "PRECISION": np.float32,
        "CPU_COUNT": 1,
        "DELTA": float(domain_cfg["resolution"]),
        "NX": int(domain_cfg["grid"]["nx"]),
        "NY": int(domain_cfg["grid"]["ny"]),
        "NZ": int(domain_cfg["grid"]["nz"]),
        "OUTPUT_FPS": int(output_cfg["fps"]),
        "PRINT_FREQUENCY": 100,
        "OUTPUT_TIME_STEP": 1.0 / int(output_cfg["fps"]),
        "OUTPUT_STATUS": False,
        "WRITE_QUEUE_SIZE": 512,
        "OUTPATH": output_cfg.get("output_path", ""),
        "OUTPUT_VARIABLES": collect_output_variables(output_cfg),
        "BLENDER_PYTHON_EXE": blender_python_exe,
        "VDB_WRITER_SCRIPT": vdb_writer_script,
        "BC_CONFIG": bc_config,
        "U_INFLOW": inflow_velocity[0],
        "V_INFLOW": inflow_velocity[1],
        "W_INFLOW": inflow_velocity[2],
        "obstacle_mask": obstacle_data["mask"],
        "source_field_data": source_data,
    }


def upload_simulation_state_to_gpu(simulation_params):
    """
    Allocate the simulation fields on the host and upload persistent arrays to the GPU.
    """
    precision_dtype = np.dtype(simulation_params["PRECISION"])
    nx = simulation_params["NX"]
    ny = simulation_params["NY"]
    nz = simulation_params["NZ"]

    u = np.full((nx, ny, nz), simulation_params["U_INFLOW"], dtype=precision_dtype)
    v = np.full((nx, ny, nz), simulation_params["V_INFLOW"], dtype=precision_dtype)
    w = np.full((nx, ny, nz), simulation_params["W_INFLOW"], dtype=precision_dtype)

    p = np.zeros((nx, ny, nz), dtype=precision_dtype)
    T = np.full((nx, ny, nz), simulation_params["T_REFERENCE"], dtype=precision_dtype)
    smoke = np.zeros((nx, ny, nz), dtype=precision_dtype)
    fuel = np.zeros((nx, ny, nz), dtype=precision_dtype)
    flame = np.zeros((nx, ny, nz), dtype=precision_dtype)

    Fx = np.zeros((nx, ny, nz), dtype=precision_dtype)
    Fy = np.zeros((nx, ny, nz), dtype=precision_dtype)
    Fz = np.zeros((nx, ny, nz), dtype=precision_dtype)

    obstacle_mask_host = np.asarray(simulation_params["obstacle_mask"])
    source_field_data = simulation_params["source_field_data"]
    source_mask_host = np.asarray(source_field_data["mask"])
    source_temperature_host = np.asarray(source_field_data["temperature"], dtype=precision_dtype)
    source_smoke_host = np.asarray(source_field_data["smoke"], dtype=precision_dtype)
    source_fuel_host = np.asarray(source_field_data["fuel"], dtype=precision_dtype)

    T = np.maximum(T, source_temperature_host)
    smoke = np.maximum(smoke, source_smoke_host)
    fuel = np.maximum(fuel, source_fuel_host)

    device_state = {
        "u": cuda.to_device(u),
        "v": cuda.to_device(v),
        "w": cuda.to_device(w),
        "u_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "v_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "w_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "p": cuda.to_device(p),
        "pressure_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "pressure_rhs": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "T": cuda.to_device(T),
        "temperature_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "smoke": cuda.to_device(smoke),
        "smoke_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "fuel": cuda.to_device(fuel),
        "fuel_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "flame": cuda.to_device(flame),
        "flame_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "Fx": cuda.to_device(Fx),
        "Fy": cuda.to_device(Fy),
        "Fz": cuda.to_device(Fz),
        "obstacle_mask": cuda.to_device(obstacle_mask_host),
        "source_mask": cuda.to_device(source_mask_host),
        "source_temperature": cuda.to_device(source_temperature_host),
        "source_smoke": cuda.to_device(source_smoke_host),
        "source_fuel": cuda.to_device(source_fuel_host),
    }

    gpu_constants = {
        "RHO": precision_dtype.type(simulation_params["RHO"]),
        "NU": precision_dtype.type(simulation_params["NU"]),
        "NU_TEMPERATURE": precision_dtype.type(simulation_params["NU_TEMPERATURE"]),
        "NU_SMOKE": precision_dtype.type(simulation_params["NU_SMOKE"]),
        "NU_FUEL": precision_dtype.type(simulation_params["NU_FUEL"]),
        "TEMPERATURE_DISSIPATION_RATE": precision_dtype.type(simulation_params["TEMPERATURE_DISSIPATION_RATE"]),
        "TEMPERATURE_PRODUCTION_RATE": precision_dtype.type(simulation_params["TEMPERATURE_PRODUCTION_RATE"]),
        "SMOKE_DISSIPATION_RATE": precision_dtype.type(simulation_params["SMOKE_DISSIPATION_RATE"]),
        "SMOKE_PRODUCTION_RATE": precision_dtype.type(simulation_params["SMOKE_PRODUCTION_RATE"]),
        "FUEL_BURN_RATE": precision_dtype.type(simulation_params["FUEL_BURN_RATE"]),
        "FUEL_IGNITION_TEMPERATURE": precision_dtype.type(simulation_params["FUEL_IGNITION_TEMPERATURE"]),
        "T_REFERENCE": precision_dtype.type(simulation_params["T_REFERENCE"]),
        "BUOANCY_FACTOR": precision_dtype.type(simulation_params["BUOANCY_FACTOR"]),
        "EXPANSION_RATE": precision_dtype.type(simulation_params["EXPANSION_RATE"]),
        "DELTA": precision_dtype.type(simulation_params["DELTA"]),
        "U_INFLOW": precision_dtype.type(simulation_params["U_INFLOW"]),
        "V_INFLOW": precision_dtype.type(simulation_params["V_INFLOW"]),
        "W_INFLOW": precision_dtype.type(simulation_params["W_INFLOW"]),
        "NX": simulation_params["NX"],
        "NY": simulation_params["NY"],
        "NZ": simulation_params["NZ"],
    }

    return device_state, gpu_constants


def copy_device_fields_to_host(field_map):
    """Copy a dictionary of device arrays to host NumPy arrays."""
    return {name: device_array.copy_to_host() for name, device_array in field_map.items()}


def select_fields(field_map, field_names):
    """Build a dictionary view containing only the requested field names."""
    return {name: field_map[name] for name in field_names}


@cuda.jit
def _velocity_maxima_timestep(u, v, w, maxima, total_size):
    """
    computes global velocity maxima in one GPU kernel.

    Each CUDA block scans a strided chunk of the three velocity fields, reduces
    local maxima in shared memory and atomically updates one global maximum per
    velocity component.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        maxima (device array): output array with shape (3,) for global maxima
        total_size (int): flattened number of elements in each field
    """
    s_u = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)
    s_v = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)
    s_w = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)

    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    idx = cuda.grid(1)

    u_flat = u.reshape(total_size)
    v_flat = v.reshape(total_size)
    w_flat = w.reshape(total_size)

    max_u = 0.0
    max_v = 0.0
    max_w = 0.0

    while idx < total_size:
        val_u = abs(u_flat[idx])
        val_v = abs(v_flat[idx])
        val_w = abs(w_flat[idx])

        if val_u > max_u:
            max_u = val_u
        if val_v > max_v:
            max_v = val_v
        if val_w > max_w:
            max_w = val_w

        idx += stride

    s_u[tid] = max_u
    s_v[tid] = max_v
    s_w[tid] = max_w
    cuda.syncthreads()

    offset = cuda.blockDim.x // 2
    while offset > 0:
        if tid < offset:
            if s_u[tid + offset] > s_u[tid]:
                s_u[tid] = s_u[tid + offset]
            if s_v[tid + offset] > s_v[tid]:
                s_v[tid] = s_v[tid + offset]
            if s_w[tid + offset] > s_w[tid]:
                s_w[tid] = s_w[tid + offset]
        cuda.syncthreads()
        offset //= 2

    if tid == 0:
        cuda.atomic.max(maxima, 0, s_u[0])
        cuda.atomic.max(maxima, 1, s_v[0])
        cuda.atomic.max(maxima, 2, s_w[0])


def compute_new_timestep_gpu(u, v, w, fx_max, fy_max, fz_max, rho, delta, nu, cfl_max):
    """
    computes a stable timestep from convection, diffusion and forcing limits on the GPU.

    A GPU reduction pass determines the maximum absolute values of the three
    velocity components. Directional force maxima are provided directly by the
    caller and used on the host to evaluate the forcing timestep restriction.
    The smallest of the convective, diffusive and forcing restrictions is
    returned.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        fx_max (float): maximum absolute x-direction force used in the timestep limiter
        fy_max (float): maximum absolute y-direction force used in the timestep limiter
        fz_max (float): maximum absolute z-direction force used in the timestep limiter
        rho (float): fluid density
        delta (float): grid spacing
        nu (float): kinematic viscosity
        cfl_max (float): maximum admissible CFL number
    Returns:
        float: stable timestep
    """
    eps = 1e-12
    total_size = u.size
    blockspergrid = min(1024, (total_size + REDUCTION_THREADS_PER_BLOCK - 1) // REDUCTION_THREADS_PER_BLOCK)
    maxima = cuda.to_device(np.zeros(3, dtype=np.float32))

    _velocity_maxima_timestep[blockspergrid, REDUCTION_THREADS_PER_BLOCK](u, v, w, maxima, total_size)

    abs_u_max, abs_v_max, abs_w_max = maxima.copy_to_host()

    cfl_delta = cfl_max * delta
    dt_conv = min(
        cfl_delta / max(float(abs_u_max), eps),
        cfl_delta / max(float(abs_v_max), eps),
        cfl_delta / max(float(abs_w_max), eps),
    )
    dt_diff = delta * delta / (6.0 * nu)
    dt_forcing = min(
        cfl_delta * rho / max(abs(float(fx_max)), eps),
        cfl_delta * rho / max(abs(float(fy_max)), eps),
        cfl_delta * rho / max(abs(float(fz_max)), eps),
    )

    return min(dt_conv, dt_diff, dt_forcing)
