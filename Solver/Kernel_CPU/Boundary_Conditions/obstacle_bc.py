from numba import njit, prange


@njit(cache=True, parallel=True)
def obstacle_bc(
    u,
    v,
    w,
    smoke,
    fuel,
    flame,
    mask,
    has_obstacle_velocity,
    obstacle_velocity_x,
    obstacle_velocity_y,
    obstacle_velocity_z,
):
    """
    Apply all obstacle boundary conditions to the CPU field state.

    Each worker scans a disjoint chunk of the flattened arrays and overwrites
    obstacle cells with either obstacle wall velocities or zeros while clearing
    supported scalar fields.
    """
    total_size = mask.size
    u_flat = u.reshape(total_size)
    v_flat = v.reshape(total_size)
    w_flat = w.reshape(total_size)
    smoke_flat = smoke.reshape(total_size)
    fuel_flat = fuel.reshape(total_size)
    flame_flat = flame.reshape(total_size)
    mask_flat = mask.reshape(total_size)
    velocity_x_flat = obstacle_velocity_x.reshape(total_size)
    velocity_y_flat = obstacle_velocity_y.reshape(total_size)
    velocity_z_flat = obstacle_velocity_z.reshape(total_size)

    for idx in prange(total_size):
        if not mask_flat[idx]:
            continue

        if has_obstacle_velocity:
            u_flat[idx] = velocity_x_flat[idx]
            v_flat[idx] = velocity_y_flat[idx]
            w_flat[idx] = velocity_z_flat[idx]
        else:
            u_flat[idx] = 0.0
            v_flat[idx] = 0.0
            w_flat[idx] = 0.0

        smoke_flat[idx] = 0.0
        fuel_flat[idx] = 0.0
        flame_flat[idx] = 0.0

    return u, v, w, smoke, fuel, flame
