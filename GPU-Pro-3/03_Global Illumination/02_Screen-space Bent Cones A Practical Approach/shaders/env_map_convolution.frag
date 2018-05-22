#version 330 core

#define final
#define M_PI	3.14159265358979323846 
#define M_1_PI	0.318309886183790671538

uniform samplerCube inputCube;

uniform int sampleCount;
uniform float cutOffAngle;

in vec2 texcoord;

layout(location = 0) out vec3 posXLayer;
layout(location = 1) out vec3 negXLayer;
layout(location = 2) out vec3 posYLayer;
layout(location = 3) out vec3 negYLayer;
layout(location = 4) out vec3 posZLayer;
layout(location = 5) out vec3 negZLayer;

float computeGeometricTerm(const in vec3 inDirection, const in vec3 outDirection) {
	return max(0.0, dot(inDirection, outDirection));
}

float computeReflectance(const in vec3 inDirection, const in vec3 outDirection) {
	return M_1_PI; // lambert
}

vec3 computeIncomingRadiance(const in vec3 inDirection) {
	return texture(inputCube, inDirection).rgb;
}

float computePDF(const in vec3 inDirection) {
	return M_1_PI * 0.5; // uniform
}

vec3 computeRadiance(const in vec3 inDirection, const in vec3 outDirection) {	
	return 
		computeIncomingRadiance(inDirection) * 		 
		computeReflectance(inDirection, outDirection) * 
		computeGeometricTerm(inDirection, outDirection) /
		computePDF(inDirection);
}

float computeConeWeight(const in vec3 inDirection, const in vec3 outDirection, const in float cutOffAngleCosine) {
	float angle = dot(inDirection, outDirection);
	if(angle > cutOffAngleCosine) return 1.0;
	return 0.0;
}

vec3 unitSphericalToCarthesian(const vec2 spherical) {
	final float phi = spherical.x;
	final float theta = spherical.y;
	final float x = sin(phi) * sin(theta);
	final float y = cos(phi) * sin(theta);
	final float z = cos(theta);
	return vec3(x, y, z);
}

void createOrthoNormalBasis(const in vec3 n, out vec3 tangent, out vec3 binormal, out vec3 normal) {
	normal = normalize(n);

	if(abs(normal.x) > abs(normal.z)) {
		binormal = vec3(-normal.y, normal.x, 0.0);
	}
	else {
		binormal = vec3(0.0, -normal.z, normal.y);
	}

	binormal = normalize(binormal);
	tangent = cross(binormal, normal);
}

vec3 computeConvolutionForDirection(const in vec3 direction) {
	vec3 radianceSum = vec3(0.0);

	vec3 u,v,w;
	createOrthoNormalBasis(direction, u,v,w);

	final float cutOffAngleCosine = cos(cutOffAngle);

	ivec2 sampleCount = ivec2(sampleCount, sampleCount);
	float weigthsSum = 0.0;

	for(int i = 0; i < sampleCount.y; i++) {
		for(int j = 0; j < sampleCount.x; j++) {
			final float s = 2.0 * M_PI * (float(j)+0.5) / float(sampleCount.x);
			// uniform sampling of hemisphere
			//final float t = acos((float(i)+0.5) / float(sampleCount.y));
			final float t = acos((float(i)+0.5) / float(sampleCount.y) * ((1.0 - cutOffAngleCosine) + cutOffAngleCosine));
			final vec3 sampleDirection = unitSphericalToCarthesian(vec2(s, t));

			// transform to orthonormal basis of direction
			final vec3 finalSampleDirection = sampleDirection.x * u + sampleDirection.y * v + sampleDirection.z * w;

			final vec3 radianceSample = computeRadiance(finalSampleDirection, direction);
			final float coneWeight = 1.0; // all samples in cone
			weigthsSum += coneWeight;
			radianceSum += coneWeight * radianceSample;
		}
	}
	radianceSum.rgb /= weigthsSum;

	return radianceSum.rgb;
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

	posXLayer = computeConvolutionForDirection(posXDirection);
	negXLayer = computeConvolutionForDirection(negXDirection);
	posYLayer = computeConvolutionForDirection(posYDirection);
	negYLayer = computeConvolutionForDirection(negYDirection);
	posZLayer = computeConvolutionForDirection(posZDirection);
	negZLayer = computeConvolutionForDirection(negZDirection);
}