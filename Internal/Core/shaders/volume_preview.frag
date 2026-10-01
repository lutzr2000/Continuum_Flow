vec2 intersect_box(vec3 origin, vec3 direction)
{
    vec3 inverse_direction = 1.0 / direction;

    vec3 near_plane =
        (volume_parameters.bounds_min_and_smoke_density.xyz - origin) *
        inverse_direction;

    vec3 far_plane =
        (volume_parameters.bounds_max_and_flame_density.xyz - origin) *
        inverse_direction;

    vec3 entry = min(near_plane, far_plane);
    vec3 exit = max(near_plane, far_plane);

    float entry_distance =
        max(max(entry.x, entry.y), entry.z);

    float exit_distance =
        min(min(exit.x, exit.y), exit.z);

    return vec2(entry_distance, exit_distance);
}


vec2 fetch_sparse_cell(ivec3 cell)
{
    ivec4 tile_shape_and_size =
        ivec4(volume_parameters.tile_shape_and_size);

    ivec3 atlas_tile_shape =
        ivec3(volume_parameters.atlas_tile_shape.xyz);

    ivec3 cell_shape =
        tile_shape_and_size.xyz *
        tile_shape_and_size.w;

    if (any(lessThan(cell, ivec3(0))) ||
        any(greaterThanEqual(cell, cell_shape)))
    {
        return vec2(0.0);
    }

    ivec3 tile =
        cell / tile_shape_and_size.w;

    int packed_slot = int(
        round(
            texelFetch(
                tile_lookup_texture,
                tile,
                0
            ).r
        )
    );

    if (packed_slot < 0) {
        return vec2(0.0);
    }

    ivec3 atlas_tile;

    atlas_tile.x =
        packed_slot %
        atlas_tile_shape.x;

    atlas_tile.y =
        (packed_slot / atlas_tile_shape.x) %
        atlas_tile_shape.y;

    atlas_tile.z =
        packed_slot /
        (
            atlas_tile_shape.x *
            atlas_tile_shape.y
        );

    ivec3 local_cell =
        cell %
        tile_shape_and_size.w;

    ivec3 atlas_cell =
        atlas_tile *
        tile_shape_and_size.w +
        local_cell;

    return texelFetch(
        field_atlas_texture,
        atlas_cell,
        0
    ).rg;
}


vec2 sample_sparse_fields(vec3 texture_coordinate)
{
    ivec4 tile_shape_and_size =
        ivec4(volume_parameters.tile_shape_and_size);

    ivec3 cell_shape =
        tile_shape_and_size.xyz *
        tile_shape_and_size.w;

    vec3 cell_position =
        texture_coordinate *
        vec3(cell_shape) -
        vec3(0.5);

    ivec3 cell0 =
        ivec3(floor(cell_position));

    ivec3 cell1 =
        cell0 + ivec3(1);

    vec3 fraction =
        fract(cell_position);

    vec2 c000 =
        fetch_sparse_cell(
            ivec3(cell0.x, cell0.y, cell0.z)
        );

    vec2 c100 =
        fetch_sparse_cell(
            ivec3(cell1.x, cell0.y, cell0.z)
        );

    vec2 c010 =
        fetch_sparse_cell(
            ivec3(cell0.x, cell1.y, cell0.z)
        );

    vec2 c110 =
        fetch_sparse_cell(
            ivec3(cell1.x, cell1.y, cell0.z)
        );

    vec2 c001 =
        fetch_sparse_cell(
            ivec3(cell0.x, cell0.y, cell1.z)
        );

    vec2 c101 =
        fetch_sparse_cell(
            ivec3(cell1.x, cell0.y, cell1.z)
        );

    vec2 c011 =
        fetch_sparse_cell(
            ivec3(cell0.x, cell1.y, cell1.z)
        );

    vec2 c111 =
        fetch_sparse_cell(
            ivec3(cell1.x, cell1.y, cell1.z)
        );

    vec2 c00 =
        mix(c000, c100, fraction.x);

    vec2 c10 =
        mix(c010, c110, fraction.x);

    vec2 c01 =
        mix(c001, c101, fraction.x);

    vec2 c11 =
        mix(c011, c111, fraction.x);

    vec2 c0 =
        mix(c00, c10, fraction.y);

    vec2 c1 =
        mix(c01, c11, fraction.y);

    return mix(
        c0,
        c1,
        fraction.z
    );
}


void main()
{
    vec3 ray_direction =
        normalize(
            world_position -
            volume_parameters.camera_position_and_step_size.xyz
        );

    vec2 hit =
        intersect_box(
            volume_parameters.camera_position_and_step_size.xyz,
            ray_direction
        );

    float distance_along_ray =
        max(hit.x, 0.0);

    if (hit.y <= distance_along_ray) {
        discard;
    }


    vec4 accumulated =
        vec4(0.0);

    float first_contribution_distance =
        hit.y;

    bool has_contribution =
        false;


    vec3 inverse_bounds_size =
        1.0 /
        (
            volume_parameters.bounds_max_and_flame_density.xyz -
            volume_parameters.bounds_min_and_smoke_density.xyz
        );


    float smoke_step_scale =
        volume_parameters.bounds_min_and_smoke_density.w *
        volume_parameters.camera_position_and_step_size.w;

    float flame_step_scale =
        volume_parameters.bounds_max_and_flame_density.w *
        volume_parameters.camera_position_and_step_size.w;


    float ray_entry_distance =
        distance_along_ray;

    float ray_segment_length =
        max(
            hit.y - ray_entry_distance,
            0.0001
        );


    for (
        int sample_index = 0;
        sample_index < 512;
        ++sample_index
    )
    {
        if (
            distance_along_ray >= hit.y ||
            accumulated.a >= 0.995
        )
        {
            break;
        }


        vec3 sample_position =
            volume_parameters.camera_position_and_step_size.xyz +
            ray_direction *
            distance_along_ray;


        vec3 texture_coordinate =
            (
                sample_position -
                volume_parameters.bounds_min_and_smoke_density.xyz
            ) *
            inverse_bounds_size;


        vec2 fields =
            sample_sparse_fields(
                texture_coordinate
            );


        float smoke =
            max(fields.r, 0.0);

        float flame =
            max(fields.g, 0.0);


        if (
            smoke < 0.0001 &&
            flame < 0.0001
        )
        {
            distance_along_ray +=
                volume_parameters.camera_position_and_step_size.w;

            continue;
        }


        if (!has_contribution) {
            first_contribution_distance =
                distance_along_ray;

            has_contribution =
                true;
        }


        float smoke_alpha =
            1.0 -
            exp(
                -smoke *
                smoke_step_scale
            );


        float flame_emission =
            min(
                flame *
                flame_step_scale,
                1.0
            );


        float normalized_camera_depth =
            clamp(
                (
                    distance_along_ray -
                    ray_entry_distance
                ) /
                ray_segment_length,
                0.0,
                1.0
            );


        float smoke_depth_shading =
            mix(
                1.0,
                0.1,
                normalized_camera_depth
            );


        vec3 smoke_sample_color =
            volume_parameters.smoke_color.rgb *
            smoke_depth_shading *
            smoke_alpha;


        vec3 flame_sample_color =
            volume_parameters.flame_color.rgb *
            flame_emission;


        float transmittance =
            1.0 -
            accumulated.a;


        accumulated.rgb +=
            transmittance *
            (
                smoke_sample_color +
                flame_sample_color
            );


        accumulated.a +=
            transmittance *
            smoke_alpha;


        distance_along_ray +=
            volume_parameters.camera_position_and_step_size.w;
    }


    if (!has_contribution) {
        discard;
    }


    vec3 first_contribution_position =
        volume_parameters.camera_position_and_step_size.xyz +
        ray_direction *
        first_contribution_distance;


    vec4 first_contribution_clip =
        view_projection_matrix *
        model_matrix *
        vec4(
            first_contribution_position,
            1.0
        );


    gl_FragDepth =
        first_contribution_clip.z /
        first_contribution_clip.w *
        0.5 +
        0.5;


    frag_color =
        accumulated;
}