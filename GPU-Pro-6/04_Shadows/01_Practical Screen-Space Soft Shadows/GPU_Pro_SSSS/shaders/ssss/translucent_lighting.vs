#version 430 core

uniform mat4 mvp;
uniform mat3 normal_mat;

layout(location=0) in vec4 in_vertex;
layout(location=2) in vec3 in_normal;

out vec3 normal;

void main()
{
	normal = normal_mat * in_normal;
  gl_Position = mvp * in_vertex;
}
