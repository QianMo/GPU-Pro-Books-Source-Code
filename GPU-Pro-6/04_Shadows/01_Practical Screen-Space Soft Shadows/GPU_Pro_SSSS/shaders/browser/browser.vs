#version 430 core

const mat4 mvp = mat4
( 1, 0,  0,    -0,
  0, 1,  0,    -0,
  0, 0, -0.01, -0,
  0, 0,  0,     1 );

layout(location=0) in vec4 in_vertex;
layout(location=1) in vec2 in_texture;

out cross_shader_data
{
  vec2 tex_coord;
} o; 

void main()
{
  o.tex_coord = in_texture;
  gl_Position = mvp * in_vertex;
}