#version 430 core

uniform mat4 light_model_mat;

layout(location=0) in vec4 in_vertex;

void main()
{
  gl_Position = light_model_mat * in_vertex;
}
