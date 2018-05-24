#version 420 core

uniform mat4 mvp;

layout(location=0) in vec4 in_vertex;

void main()
{
  gl_Position = mvp * in_vertex;
}
