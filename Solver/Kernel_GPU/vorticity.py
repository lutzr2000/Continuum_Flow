from numba import cuda
import math

import Solver.Kernel_GPU.sparse_managment as sparse_managment
import Solver.Kernel_GPU.kernel_config as kernel_config

@cuda.jit(cache=True)
def compute_vorticity(
    u,
    v,
    w,
    u_initial,
    v_initial,
    w_initial,
    obstacle_mask,
    vorticity_magnitude,
    delta,
    tile_map,
    nx,
    ny,
    nz,
):
    (
        tile_i,
        tile_j,
        tile_k,
        local_i,
        local_j,
        local_k,
        i,
        j,
        k,
    ) = sparse_managment.tile_to_index()

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        vorticity_magnitude[tile_index, local_i, local_j, local_k] = 0.0
        return

    if obstacle_mask[i, j, k]:
        vorticity_magnitude[tile_index, local_i, local_j, local_k] = 0.0
        return

    half_inv_delta = 0.5 / delta

    du_dy = (
        sparse_managment.get_pool_value(u, tile_map, i, j + 1, k, u_initial)
        - sparse_managment.get_pool_value(u, tile_map, i, j - 1, k, u_initial)
    ) * half_inv_delta
    du_dz = (
        sparse_managment.get_pool_value(u, tile_map, i, j, k + 1, u_initial)
        - sparse_managment.get_pool_value(u, tile_map, i, j, k - 1, u_initial)
    ) * half_inv_delta

    dv_dx = (
        sparse_managment.get_pool_value(v, tile_map, i + 1, j, k, v_initial)
        - sparse_managment.get_pool_value(v, tile_map, i - 1, j, k, v_initial)
    ) * half_inv_delta
    dv_dz = (
        sparse_managment.get_pool_value(v, tile_map, i, j, k + 1, v_initial)
        - sparse_managment.get_pool_value(v, tile_map, i, j, k - 1, v_initial)
    ) * half_inv_delta

    dw_dx = (
        sparse_managment.get_pool_value(w, tile_map, i + 1, j, k, w_initial)
        - sparse_managment.get_pool_value(w, tile_map, i - 1, j, k, w_initial)
    ) * half_inv_delta
    dw_dy = (
        sparse_managment.get_pool_value(w, tile_map, i, j + 1, k, w_initial)
        - sparse_managment.get_pool_value(w, tile_map, i, j - 1, k, w_initial)
    ) * half_inv_delta

    wx = dw_dy - dv_dz
    wy = du_dz - dw_dx
    wz = dv_dx - du_dy

    vorticity_magnitude[tile_index, local_i, local_j, local_k] = math.sqrt(
        wx * wx + wy * wy + wz * wz
    )


@cuda.jit(device=True, inline=True, cache=True)
def apply_vorticity_confinement(
    u,
    v,
    w,
    obstacle_mask,
    omega_magnitude,
    i,
    j,
    k,
    delta,
    vorticity_strength,
    tile_map,
    u_initial,
    v_initial,
    w_initial,
    nx,
    ny,
    nz,
):
    """
    Compute the local vorticity confinement force in one GPU cell.
    """
    tile_i = i // kernel_config.TILE_SIZE
    tile_j = j // kernel_config.TILE_SIZE
    tile_k = k // kernel_config.TILE_SIZE

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return 0.0, 0.0, 0.0

    if (
        i < 2
        or j < 2
        or k < 2
        or i >= nx - 2
        or j >= ny - 2
        or k >= nz - 2
        or obstacle_mask[i, j, k]
    ):
        return 0.0, 0.0, 0.0

    half_inv_delta = 0.5 / delta

    grad_x = (
        sparse_managment.get_pool_value(omega_magnitude, tile_map, i + 1, j, k, 0.0)
        - sparse_managment.get_pool_value(omega_magnitude, tile_map, i - 1, j, k, 0.0)
    ) * half_inv_delta

    grad_y = (
        sparse_managment.get_pool_value(omega_magnitude, tile_map, i, j + 1, k, 0.0)
        - sparse_managment.get_pool_value(omega_magnitude, tile_map, i, j - 1, k, 0.0)
    ) * half_inv_delta

    grad_z = (
        sparse_managment.get_pool_value(omega_magnitude, tile_map, i, j, k + 1, 0.0)
        - sparse_managment.get_pool_value(omega_magnitude, tile_map, i, j, k - 1, 0.0)
    ) * half_inv_delta

    grad_length = math.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)

    if grad_length <= 1.0e-12:
        return 0.0, 0.0, 0.0

    nx_dir = grad_x / grad_length
    ny_dir = grad_y / grad_length
    nz_dir = grad_z / grad_length

    du_dy = (
        sparse_managment.get_pool_value(u, tile_map, i, j + 1, k, u_initial)
        - sparse_managment.get_pool_value(u, tile_map, i, j - 1, k, u_initial)
    ) * half_inv_delta
    du_dz = (
        sparse_managment.get_pool_value(u, tile_map, i, j, k + 1, u_initial)
        - sparse_managment.get_pool_value(u, tile_map, i, j, k - 1, u_initial)
    ) * half_inv_delta

    dv_dx = (
        sparse_managment.get_pool_value(v, tile_map, i + 1, j, k, v_initial)
        - sparse_managment.get_pool_value(v, tile_map, i - 1, j, k, v_initial)
    ) * half_inv_delta
    dv_dz = (
        sparse_managment.get_pool_value(v, tile_map, i, j, k + 1, v_initial)
        - sparse_managment.get_pool_value(v, tile_map, i, j, k - 1, v_initial)
    ) * half_inv_delta

    dw_dx = (
        sparse_managment.get_pool_value(w, tile_map, i + 1, j, k, w_initial)
        - sparse_managment.get_pool_value(w, tile_map, i - 1, j, k, w_initial)
    ) * half_inv_delta
    dw_dy = (
        sparse_managment.get_pool_value(w, tile_map, i, j + 1, k, w_initial)
        - sparse_managment.get_pool_value(w, tile_map, i, j - 1, k, w_initial)
    ) * half_inv_delta

    wx = dw_dy - dv_dz
    wy = du_dz - dw_dx
    wz = dv_dx - du_dy

    fx = vorticity_strength * (ny_dir * wz - nz_dir * wy)
    fy = vorticity_strength * (nz_dir * wx - nx_dir * wz)
    fz = vorticity_strength * (nx_dir * wy - ny_dir * wx)

    return fx, fy, fz
