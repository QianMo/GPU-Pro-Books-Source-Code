#version 330 core

uniform mat4 MVP;
uniform mat3 normalM;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

out Vert {
	vec3 position;
	vec3 normal;
    vec2 texcoord;
} vert;


void main() {
    vec4 inPos = vec4(position, 1.0);

	vert.position = position;
    //vert.normal = normalM * normalize(normal);
    vert.normal = normalize(normal);
    vert.texcoord = texcoord;

    gl_Position = MVP * inPos;
}