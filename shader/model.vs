#version 330 core

layout (location = 0) in float a_id;

// Uniforms
// ------------------------------------
uniform mat4  u_model;
uniform mat4  u_view;
uniform mat4  u_projection;
uniform float u_size;
uniform float u_num;
uniform float u_rad;
uniform vec3  u_color;
uniform vec3  u_pos[100];

out vec4 f_color;
out float f_size;
flat out int f_ignore;

void main (void)
{
    float v_size = 8.0 * u_rad * u_size;
    float v_linewidth = 1.0;
    float v_antialias = 1.0;
    
    int pointNumber = int(u_num);
    int pointId = int(a_id);

    if(pointId < pointNumber)
    {
        vec3 pos = u_pos[int(a_id)];
        gl_Position = u_projection * u_view * u_model * vec4(pos, 1.0);
        gl_PointSize = v_size + 2.*(v_linewidth + 1.5 * v_antialias);
        f_color = vec4(u_color, 1.0);
        f_size = v_size;
        f_ignore = 0;
    }
    else
    {
        gl_Position = vec4(0, 0, 0, 0);
        gl_PointSize = 0;
        f_color = vec4(0, 0, 0, 0);
        f_size = 0.0;
        f_ignore = 1;
    }
}