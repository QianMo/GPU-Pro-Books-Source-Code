#version 430 core

uniform mat4 mvp;

layout(location=0) in vec4 in_vertex;

out vec3 pos;

void main()
{
  pos = in_vertex.xyz; //model space pos
  gl_Position = mvp * in_vertex;
}
