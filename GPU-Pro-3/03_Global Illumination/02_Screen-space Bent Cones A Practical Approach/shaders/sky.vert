#version 330 core  

layout(location = 0) in vec2 position;

uniform mat4 invViewProjection;

out vec3 viewDir;

void main() {
	vec4 homViewDir = invViewProjection * vec4(position, 1.0, 1.0);
	homViewDir.xyz /= homViewDir.w;
	vec4 homViewDir2 = invViewProjection * vec4(position, -1.0, 1.0);
	homViewDir2.xyz /= homViewDir2.w;
	viewDir.xyz = normalize(homViewDir.xyz - homViewDir2.xyz);

	gl_Position = vec4(position, 0.99999, 1.0);
}