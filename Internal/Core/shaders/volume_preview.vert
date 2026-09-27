void main()
{
    world_position = position;
    gl_Position = view_projection_matrix * model_matrix * vec4(position, 1.0);
}
