import numpy as np
from numba import cuda

REDUCTION_THREADS_PER_BLOCK = 256

@cuda.jit
def _partial_maxima_timestep(u, v, w, Fx, Fy, Fz, partial_maxima, total_size):
    """
    computes per-block partial maxima for velocity and forcing fields in one GPU pass.

    Each CUDA block scans a strided chunk of all six fields, stores the largest
    absolute values for each quantity in shared memory and reduces them to one
    partial maximum per block. The host combines these partial maxima into the
    final timestep limits.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        Fx (device array): x-direction force field
        Fy (device array): y-direction force field
        Fz (device array): z-direction force field
        partial_maxima (device array): per-block output maxima for all six fields
        total_size (int): flattened number of elements in each field
    """
    s_u = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)
    s_v = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)
    s_w = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)
    s_Fx = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)
    s_Fy = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)
    s_Fz = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)

    tid = cuda.threadIdx.x
    block = cuda.blockIdx.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    idx = cuda.grid(1)

    u_flat = u.reshape(total_size)
    v_flat = v.reshape(total_size)
    w_flat = w.reshape(total_size)
    fx_flat = Fx.reshape(total_size)
    fy_flat = Fy.reshape(total_size)
    fz_flat = Fz.reshape(total_size)

    max_u = 0.0
    max_v = 0.0
    max_w = 0.0
    max_Fx = 0.0
    max_Fy = 0.0
    max_Fz = 0.0

    while idx < total_size:
        val_u = abs(u_flat[idx])
        val_v = abs(v_flat[idx])
        val_w = abs(w_flat[idx])
        val_Fx = abs(fx_flat[idx])
        val_Fy = abs(fy_flat[idx])
        val_Fz = abs(fz_flat[idx])

        if val_u > max_u:
            max_u = val_u
        if val_v > max_v:
            max_v = val_v
        if val_w > max_w:
            max_w = val_w
        if val_Fx > max_Fx:
            max_Fx = val_Fx
        if val_Fy > max_Fy:
            max_Fy = val_Fy
        if val_Fz > max_Fz:
            max_Fz = val_Fz

        idx += stride

    s_u[tid] = max_u
    s_v[tid] = max_v
    s_w[tid] = max_w
    s_Fx[tid] = max_Fx
    s_Fy[tid] = max_Fy
    s_Fz[tid] = max_Fz
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
            if s_Fx[tid + offset] > s_Fx[tid]:
                s_Fx[tid] = s_Fx[tid + offset]
            if s_Fy[tid + offset] > s_Fy[tid]:
                s_Fy[tid] = s_Fy[tid + offset]
            if s_Fz[tid + offset] > s_Fz[tid]:
                s_Fz[tid] = s_Fz[tid + offset]
        cuda.syncthreads()
        offset //= 2

    if tid == 0:
        partial_maxima[block, 0] = s_u[0]
        partial_maxima[block, 1] = s_v[0]
        partial_maxima[block, 2] = s_w[0]
        partial_maxima[block, 3] = s_Fx[0]
        partial_maxima[block, 4] = s_Fy[0]
        partial_maxima[block, 5] = s_Fz[0]


def compute_new_timestep_gpu(u, v, w, Fx, Fy, Fz, rho, delta, nu, cfl_max):
    """
    computes a stable timestep from convection, diffusion and forcing limits on the GPU.

    A combined GPU reduction pass determines the maximum absolute values of all
    three velocity components and all three force components. These maxima are
    then used on the host to evaluate the convective, diffusive and forcing
    timestep restrictions. The smallest of these restrictions is returned.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        Fx (device array): x-direction force field used in the timestep limiter
        Fy (device array): y-direction force field used in the timestep limiter
        Fz (device array): z-direction force field used in the timestep limiter
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
    partial_maxima = cuda.device_array((blockspergrid, 6), dtype=np.float32)

    _partial_maxima_timestep[blockspergrid, REDUCTION_THREADS_PER_BLOCK](
        u, v, w, Fx, Fy, Fz, partial_maxima, total_size
    )

    partial_host = partial_maxima.copy_to_host()
    abs_u_max = float(np.max(partial_host[:, 0]))
    abs_v_max = float(np.max(partial_host[:, 1]))
    abs_w_max = float(np.max(partial_host[:, 2]))
    abs_Fx_max = float(np.max(partial_host[:, 3]))
    abs_Fy_max = float(np.max(partial_host[:, 4]))
    abs_Fz_max = float(np.max(partial_host[:, 5]))

    cfl_delta = cfl_max * delta
    dt_conv = min(
        cfl_delta / max(abs_u_max, eps),
        cfl_delta / max(abs_v_max, eps),
        cfl_delta / max(abs_w_max, eps),
    )
    dt_diff = delta * delta / (6.0 * nu)
    dt_forcing = min(
        cfl_delta * rho / max(abs_Fx_max, eps),
        cfl_delta * rho / max(abs_Fy_max, eps),
        cfl_delta * rho / max(abs_Fz_max, eps),
    )

    return min(dt_conv, dt_diff, dt_forcing)

