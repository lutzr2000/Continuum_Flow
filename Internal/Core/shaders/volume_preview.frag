vec2 intersect_box(vec3 origin, vec3 direction)
{
    vec3 inverse_direction = 1.0 / direction;
    vec3 near_plane = (bounds_min - origin) * inverse_direction;
    vec3 far_plane = (bounds_max - origin) * inverse_direction;
    vec3 entry = min(near_plane, far_plane);
    vec3 exit = max(near_plane, far_plane);
    return vec2(max(max(entry.x, entry.y), entry.z),
                min(min(exit.x, exit.y), exit.z));
}

void main()
{
    vec3 ray_direction = normalize(world_position - camera_position);
    vec2 hit = intersect_box(camera_position, ray_direction);
    float distance_along_ray = max(hit.x, 0.0);
    if (hit.y <= distance_along_ray) {
        discard;
    }

    vec4 accumulated = vec4(0.0);
    for (int sample_index = 0; sample_index < 512; ++sample_index) {
        if (distance_along_ray >= hit.y || accumulated.a >= 0.995) {
            break;
        }

        vec3 sample_position = camera_position + ray_direction * distance_along_ray;
        vec3 texture_coordinate =
            (sample_position - bounds_min) / (bounds_max - bounds_min);
        vec2 fields = texture(volume_texture, texture_coordinate).rg;
        float smoke = max(fields.r, 0.0);
        float flame = max(fields.g, 0.0);

        float smoke_alpha = 1.0 - exp(-smoke * density_scale * step_size);
        float flame_emission = 1.0 - exp(-flame * flame_scale * step_size);
        vec3 smoke_color = vec3(0.32, 0.34, 0.38) * smoke_alpha;
        vec3 flame_color = vec3(1.0, 0.12, 0.01) * flame_emission;
        float transmittance = 1.0 - accumulated.a;

        accumulated.rgb += transmittance * (smoke_color + flame_color);
        accumulated.a += transmittance * smoke_alpha;
        distance_along_ray += step_size;
    }

    frag_color = accumulated;
}
