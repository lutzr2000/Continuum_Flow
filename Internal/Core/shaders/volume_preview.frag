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
    vec3 exit  = max(near_plane, far_plane);

    float entry_distance =
        max(max(entry.x, entry.y), entry.z);

    float exit_distance =
        min(min(exit.x, exit.y), exit.z);

    return vec2(entry_distance, exit_distance);
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


    vec4 accumulated = vec4(0.0);
    float first_contribution_distance = hit.y;
    bool has_contribution = false;

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

    float ray_entry_distance = distance_along_ray;
    
    float ray_segment_length =
        max(hit.y - ray_entry_distance, 0.0001);


    for (int sample_index = 0;
         sample_index < 512;
         ++sample_index)
    {

        if (distance_along_ray >= hit.y ||
            accumulated.a >= 0.995)
        {
            break;
        }


        vec3 sample_position =
            volume_parameters.camera_position_and_step_size.xyz +
            ray_direction * distance_along_ray;

        vec3 texture_coordinate =
            (
                sample_position -
                volume_parameters.bounds_min_and_smoke_density.xyz
            ) *
            inverse_bounds_size;


        vec2 fields =
            texture(volume_texture, texture_coordinate).rg;

        float smoke =
            max(fields.r, 0.0);

        float flame =
            max(fields.g, 0.0);

        if (smoke < 0.0001 &&
            flame < 0.0001)
        {
            distance_along_ray +=
                volume_parameters.camera_position_and_step_size.w;
            continue;
        }

        if (!has_contribution) {
            first_contribution_distance = distance_along_ray;
            has_contribution = true;
        }

        float smoke_alpha =
            1.0 - exp(-smoke * smoke_step_scale);


        float flame_emission =
            min(flame * flame_step_scale, 1.0);


        float normalized_camera_depth =
            clamp(
                (distance_along_ray - ray_entry_distance) /
                ray_segment_length,
                0.0,
                1.0
            );

        float smoke_depth_shading =
            mix(1.0, 0.1, normalized_camera_depth);


        vec3 smoke_sample_color =
            volume_parameters.smoke_color.rgb *
            smoke_depth_shading *
            smoke_alpha;

        vec3 flame_sample_color =
            volume_parameters.flame_color.rgb *
            flame_emission;


        float transmittance =
            1.0 - accumulated.a;


        accumulated.rgb +=
            transmittance *
            (smoke_sample_color + flame_sample_color);

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
        ray_direction * first_contribution_distance;

    vec4 first_contribution_clip =
        view_projection_matrix *
        model_matrix *
        vec4(first_contribution_position, 1.0);

    gl_FragDepth =
        first_contribution_clip.z /
        first_contribution_clip.w * 0.5 + 0.5;

    frag_color = accumulated;
}
