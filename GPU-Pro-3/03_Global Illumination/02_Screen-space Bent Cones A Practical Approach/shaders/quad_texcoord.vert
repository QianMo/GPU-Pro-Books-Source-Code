#version 330 core  

layout(location = 0) in vec2 position;

out vec2 texcoord;

void main() {
    texcoord = step(0.0, position);
    gl_Position = vec4(position, 0.0, 1.0);
}