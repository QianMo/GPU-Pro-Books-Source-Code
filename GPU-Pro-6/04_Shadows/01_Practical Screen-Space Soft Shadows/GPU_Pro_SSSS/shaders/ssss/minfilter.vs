#version 430 core

uniform mat4 mvp;

layout(location=0) in vec4 in_vertex;
layout(location=1) in vec2 in_texture;

out vec2 tex_coord;

void main()
{
  tex_coord = in_texture;
  gl_Position = mvp * in_vertex;
}
