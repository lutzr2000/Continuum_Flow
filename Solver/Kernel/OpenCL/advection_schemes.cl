#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "sparse_managment.cl"

inline float clamp_value(
    const float value,
    const float lower,
    const float upper
)
{
    if (value < lower)
        return lower;

    if (value > upper)
        return upper;

    return value;
}


inline void prepare_trilinear_coords(
    float x,
    float y,
    float z,
    const int nx,
    const int ny,
    const int nz,
    int *x0,
    int *y0,
    int *z0,
    int *x1,
    int *y1,
    int *z1,
    float *tx,
    float *ty,
    float *tz
)
{
    if (x < 0.0f)
        x = 0.0f;
    else if (x > nx - 1)
        x = nx - 1.0f;

    if (y < 0.0f)
        y = 0.0f;
    else if (y > ny - 1)
        y = ny - 1.0f;

    if (z < 0.0f)
        z = 0.0f;
    else if (z > nz - 1)
        z = nz - 1.0f;

    *x0 = (int)x;
    *y0 = (int)y;
    *z0 = (int)z;

    *x1 = *x0 + 1;
    *y1 = *y0 + 1;
    *z1 = *z0 + 1;

    if (*x1 >= nx)
        *x1 = nx - 1;

    if (*y1 >= ny)
        *y1 = ny - 1;

    if (*z1 >= nz)
        *z1 = nz - 1;

    *tx = x - *x0;
    *ty = y - *y0;
    *tz = z - *z0;
}


inline float sample_trilinear_inner_sparse(
    __global const float *field,
    __global const int *tile_map,
    const int x0,
    const int y0,
    const int z0,
    const int x1,
    const int y1,
    const int z1,
    const float tx,
    const float ty,
    const float tz,
    const float default_value,
    const int tiles_y,
    const int tiles_z
)
{
    const float c000 =
        get_pool_value(
            field, tile_map,
            x0, y0, z0,
            default_value,
            tiles_y, tiles_z
        );

    const float c100 =
        get_pool_value(
            field, tile_map,
            x1, y0, z0,
            default_value,
            tiles_y, tiles_z
        );

    const float c010 =
        get_pool_value(
            field, tile_map,
            x0, y1, z0,
            default_value,
            tiles_y, tiles_z
        );

    const float c110 =
        get_pool_value(
            field, tile_map,
            x1, y1, z0,
            default_value,
            tiles_y, tiles_z
        );

    const float c001 =
        get_pool_value(
            field, tile_map,
            x0, y0, z1,
            default_value,
            tiles_y, tiles_z
        );

    const float c101 =
        get_pool_value(
            field, tile_map,
            x1, y0, z1,
            default_value,
            tiles_y, tiles_z
        );

    const float c011 =
        get_pool_value(
            field, tile_map,
            x0, y1, z1,
            default_value,
            tiles_y, tiles_z
        );

    const float c111 =
        get_pool_value(
            field, tile_map,
            x1, y1, z1,
            default_value,
            tiles_y, tiles_z
        );

    const float c00 =
        c000 + tx * (c100 - c000);

    const float c10 =
        c010 + tx * (c110 - c010);

    const float c01 =
        c001 + tx * (c101 - c001);

    const float c11 =
        c011 + tx * (c111 - c011);

    const float c0 =
        c00 + ty * (c10 - c00);

    const float c1 =
        c01 + ty * (c11 - c01);

    return c0 + tz * (c1 - c0);
}


inline void sample_cell_extrema_inner_sparse(
    __global const float *field,
    __global const int *tile_map,
    const int x0,
    const int y0,
    const int z0,
    const int x1,
    const int y1,
    const int z1,
    const float default_value,
    const int tiles_y,
    const int tiles_z,
    float *lower,
    float *upper
)
{
    const float c000 =
        get_pool_value(
            field, tile_map,
            x0, y0, z0,
            default_value,
            tiles_y, tiles_z
        );

    const float c100 =
        get_pool_value(
            field, tile_map,
            x1, y0, z0,
            default_value,
            tiles_y, tiles_z
        );

    const float c010 =
        get_pool_value(
            field, tile_map,
            x0, y1, z0,
            default_value,
            tiles_y, tiles_z
        );

    const float c110 =
        get_pool_value(
            field, tile_map,
            x1, y1, z0,
            default_value,
            tiles_y, tiles_z
        );

    const float c001 =
        get_pool_value(
            field, tile_map,
            x0, y0, z1,
            default_value,
            tiles_y, tiles_z
        );

    const float c101 =
        get_pool_value(
            field, tile_map,
            x1, y0, z1,
            default_value,
            tiles_y, tiles_z
        );

    const float c011 =
        get_pool_value(
            field, tile_map,
            x0, y1, z1,
            default_value,
            tiles_y, tiles_z
        );

    const float c111 =
        get_pool_value(
            field, tile_map,
            x1, y1, z1,
            default_value,
            tiles_y, tiles_z
        );

    *lower = fmin(
        fmin(
            fmin(c000, c100),
            fmin(c010, c110)
        ),
        fmin(
            fmin(c001, c101),
            fmin(c011, c111)
        )
    );

    *upper = fmax(
        fmax(
            fmax(c000, c100),
            fmax(c010, c110)
        ),
        fmax(
            fmax(c001, c101),
            fmax(c011, c111)
        )
    );
}


inline void sample_trilinear_vec3_sparse(
    __global const float *field_x,
    __global const float *field_y,
    __global const float *field_z,
    __global const int *tile_map,
    const float x,
    const float y,
    const float z,
    const int nx,
    const int ny,
    const int nz,
    const float default_x,
    const float default_y,
    const float default_z,
    const int tiles_y,
    const int tiles_z,
    float *sample_x,
    float *sample_y,
    float *sample_z
)
{
    int x0;
    int y0;
    int z0;
    int x1;
    int y1;
    int z1;

    float tx;
    float ty;
    float tz;

    prepare_trilinear_coords(
        x,
        y,
        z,
        nx,
        ny,
        nz,
        &x0,
        &y0,
        &z0,
        &x1,
        &y1,
        &z1,
        &tx,
        &ty,
        &tz
    );

    *sample_x =
        sample_trilinear_inner_sparse(
            field_x,
            tile_map,
            x0,
            y0,
            z0,
            x1,
            y1,
            z1,
            tx,
            ty,
            tz,
            default_x,
            tiles_y,
            tiles_z
        );

    *sample_y =
        sample_trilinear_inner_sparse(
            field_y,
            tile_map,
            x0,
            y0,
            z0,
            x1,
            y1,
            z1,
            tx,
            ty,
            tz,
            default_y,
            tiles_y,
            tiles_z
        );

    *sample_z =
        sample_trilinear_inner_sparse(
            field_z,
            tile_map,
            x0,
            y0,
            z0,
            x1,
            y1,
            z1,
            tx,
            ty,
            tz,
            default_z,
            tiles_y,
            tiles_z
        );
}


inline void backtrace_position_sparse(
    __global const float *u,
    __global const float *v,
    __global const float *w,
    __global const int *tile_map,
    const float x_start,
    const float y_start,
    const float z_start,
    const float dt_over_delta,
    const int n_substeps,
    const int nx,
    const int ny,
    const int nz,
    const float u_initial,
    const float v_initial,
    const float w_initial,
    const int tiles_y,
    const int tiles_z,
    float *x_result,
    float *y_result,
    float *z_result
)
{
    const float substep_dt =
        dt_over_delta / (float)n_substeps;

    float x_pos = x_start;
    float y_pos = y_start;
    float z_pos = z_start;

    for (int step = 0; step < n_substeps; ++step)
    {
        float u_sample;
        float v_sample;
        float w_sample;

        sample_trilinear_vec3_sparse(
            u,
            v,
            w,
            tile_map,
            x_pos,
            y_pos,
            z_pos,
            nx,
            ny,
            nz,
            u_initial,
            v_initial,
            w_initial,
            tiles_y,
            tiles_z,
            &u_sample,
            &v_sample,
            &w_sample
        );

        x_pos -= substep_dt * u_sample;
        y_pos -= substep_dt * v_sample;
        z_pos -= substep_dt * w_sample;
    }

    *x_result = x_pos;
    *y_result = y_pos;
    *z_result = z_pos;
}


inline void forward_trace_position_sparse(
    __global const float *u,
    __global const float *v,
    __global const float *w,
    __global const int *tile_map,
    const float x_start,
    const float y_start,
    const float z_start,
    const float dt_over_delta,
    const int n_substeps,
    const int nx,
    const int ny,
    const int nz,
    const float u_initial,
    const float v_initial,
    const float w_initial,
    const int tiles_y,
    const int tiles_z,
    float *x_result,
    float *y_result,
    float *z_result
)
{
    backtrace_position_sparse(
        u,
        v,
        w,
        tile_map,
        x_start,
        y_start,
        z_start,
        -dt_over_delta,
        n_substeps,
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
        tiles_y,
        tiles_z,
        x_result,
        y_result,
        z_result
    );
}