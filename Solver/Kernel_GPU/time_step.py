import numpy as np
from numba import cuda

from Solver.Kernel_GPU.kernel_config import (
    REDUCTION_THREADS_PER_BLOCK,
    reduction_blocks_per_grid,
)


def reset_velocity_maxima(maxima, host_zeros):
    """
    Reset the persistent velocity-maxima reduction buffer to zero.
    """
    maxima.copy_to_device(host_zeros)


@cuda.jit
def _velocity_maxima_timestep(u, v, w, maxima, total_size):
    """
    computes global velocity maxima in one GPU kernel.

    Each CUDA block scans a strided chunk of the three velocity fields, reduces
    local maxima in shared memory and atomically updates one global maximum per
    velocity component.

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


def compute_new_timestep_gpu(
    u, v, w, maxima, delta, cfl_max, max_dt=None
):
    """
    computes a stable timestep from convection, diffusion and forcing limits on the GPU.

    A GPU reduction pass determines the maximum absolute values of the three
    velocity components. Directional force maxima are provided directly by the
    caller and used on the host to evaluate the forcing timestep restriction.
    The smallest of the convective, diffusive and forcing restrictions is
    returned.

    """
    eps = 1e-12
    total_size = u.size
    blockspergrid = reduction_blocks_per_grid(total_size)

    _velocity_maxima_timestep[blockspergrid, REDUCTION_THREADS_PER_BLOCK](
        u, v, w, maxima, total_size
    )

    abs_u_max, abs_v_max, abs_w_max = maxima.copy_to_host()

    cfl_delta = cfl_max * delta
    dt_conv = min(
        cfl_delta / max(float(abs_u_max), eps),
        cfl_delta / max(float(abs_v_max), eps),
        cfl_delta / max(float(abs_w_max), eps),
    )

    dt = min(dt_conv, float(max_dt))

    return dt