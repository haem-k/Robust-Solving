#version 330 core

in vec4 f_color;
in float f_size;
flat in int f_ignore;

// Functions
// ------------------------------------
// ----------------
float disc(vec2 P, float size)
{
    float r = length((P.xy - vec2(0.5,0.5)) * size);
    r -= f_size / 2.0;
    return r;
}

// Main
// ------------------------------------
void main()
{
    if(f_ignore == 1)
    {
        discard;
    }

    float v_linewidth = 0.1;
    float v_antialias = 1.0;
    float v_size = f_size;

    vec4 v_fg_color = vec4(0, 0, 0, 1);
    vec4 v_bg_color = f_color;

    float size = v_size + 2.0 * (v_linewidth + 1.5 * v_antialias);
    float t = v_linewidth / 2.0 - v_antialias;
    float r = disc(gl_PointCoord, size);
    float d = abs(r) - t;

    if( r > (v_linewidth / 2.0 + v_antialias))
    {
        discard;
    }
    else if( d < 0.0 )
    {
       gl_FragColor = v_fg_color;
    }
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > 0.)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}

// in vec4 f_color;

// void main()
// {
//     gl_FragColor = f_color;
// }