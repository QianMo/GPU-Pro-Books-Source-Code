#version 330 core

struct MaterialParameters {
    // rgb
    //vec3 ambient;
    // diffuse provided as texture?!
    vec3 diffuse;
};

uniform sampler2D diffuseTexture;
uniform MaterialParameters material;
uniform float gamma; // gamma correction

in Vert {
	vec3 position;
	vec3 normal;
    vec2 texcoord;
} vert;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outDiffuse;


void main() {
    outDiffuse = texture(diffuseTexture, vert.texcoord).rgb * material.diffuse;
	outDiffuse = pow(outDiffuse, vec3(gamma));

	outPosition = vec4(vert.position, 1.0);
	outNormal = normalize(vert.normal);
}