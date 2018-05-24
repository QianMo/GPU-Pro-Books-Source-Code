#version 430 core

uniform mat4 mvp;
uniform mat4 mv;

layout(location=0) in vec4 in_vertex;
layout(location=1) in vec2 in_texture;

out vec2 tex_coord;
out vec4 vs_pos;

void main()
{
	tex_coord = in_texture;
	vs_pos = mv * in_vertex;
  gl_Position = mvp * in_vertex;
}
