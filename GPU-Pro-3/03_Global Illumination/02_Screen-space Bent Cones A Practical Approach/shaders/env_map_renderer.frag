#version 330 core

#define final

#ifndef NUMLIGHTS
#define NUMLIGHTS 1
#endif

struct LightParameters {
	vec3 color;
	vec3 direction;
	float innerAngle;
	float outerAngle;
};

uniform LightParameters lights[NUMLIGHTS];

in vec2 texcoord;

layout(location = 0) out vec3 posXLayer;
layout(location = 1) out vec3 negXLayer;
layout(location = 2) out vec3 posYLayer;
layout(location = 3) out vec3 negYLayer;
layout(location = 4) out vec3 posZLayer;
layout(location = 5) out vec3 negZLayer;

float lightFalloff(const in vec3 direction, const in vec3 referenceDirection, const in float innerAngle, const in float fallOffAngle) {
	return 1 - smoothstep(innerAngle, innerAngle+fallOffAngle, acos(dot(direction, referenceDirection)));
}

vec3 computeRadianceForDirection(const in vec3 direction) {
	vec3 outRadiance = vec3(0.0);
	for(int i=0; i<NUMLIGHTS; ++i) {
		outRadiance += lights[i].color * lightFalloff(normalize(lights[i].direction), direction, lights[i].innerAngle, lights[i].outerAngle);
	}
	return outRadiance;
}

void main() {
    final float s = 2.0 * texcoord.x - 1.0;
	final float t = 2.0 * texcoord.y - 1.0;
	
/*
	major axis
	direction     target                             sc     tc    ma
	----------    -------------------------------    ---    ---   ---
	+rx          TEXTURE_CUBE_MAP_POSITIVE_X_ARB    -rz    -ry   rx
	-rx          TEXTURE_CUBE_MAP_NEGATIVE_X_ARB    +rz    -ry   rx
	+ry          TEXTURE_CUBE_MAP_POSITIVE_Y_ARB    +rx    +rz   ry
	-ry          TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB    +rx    -rz   ry
	+rz          TEXTURE_CUBE_MAP_POSITIVE_Z_ARB    +rx    -ry   rz
	-rz          TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB    -rx    -ry   rz
*/

	final vec3 posXDirection = normalize(vec3( 1, -t, -s));
	final vec3 negXDirection = normalize(vec3(-1, -t,  s));	
	final vec3 posYDirection = normalize(vec3( s,  1,  t));
	final vec3 negYDirection = normalize(vec3( s, -1, -t));
	final vec3 posZDirection = normalize(vec3( s, -t,  1));
	final vec3 negZDirection = normalize(vec3(-s, -t, -1));

	posXLayer = computeRadianceForDirection(posXDirection);
	negXLayer = computeRadianceForDirection(negXDirection);
	posYLayer = computeRadianceForDirection(posYDirection);
	negYLayer = computeRadianceForDirection(negYDirection);
	posZLayer = computeRadianceForDirection(posZDirection);
	negZLayer = computeRadianceForDirection(negZDirection);
}