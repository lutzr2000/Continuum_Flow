void main()
{
    noise_position = coordinate;
    gl_Position = view_projection_matrix * vec4(position, 1.0);
}
