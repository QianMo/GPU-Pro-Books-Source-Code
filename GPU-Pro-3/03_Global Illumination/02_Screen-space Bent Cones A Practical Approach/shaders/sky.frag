#version 330 core

uniform samplerCube envMap;

in vec3 viewDir;

out vec4 outColor;

void main() {
	vec3 viewDirection = normalize(viewDir);
	outColor.rgb = texture(envMap, viewDirection).rgb;
	//outColor.rgb = vec3(dot(viewDirection, vec3(0.0, 0.0, 1.0)));
	//outColor.rgb = viewDirection;

	outColor.a = 1.0;
}