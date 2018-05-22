#version 330 core

#define final

uniform sampler2D positionTexture;
uniform sampler2D normalTexture;

uniform mat4 viewMatrix;
uniform mat4 viewProjectionMatrix;

uniform float sampleRadius;
uniform float maxDistance;

uniform int sampleCount;
uniform int numRayMarchingSteps;
uniform int patternSize;

uniform sampler2D seedTexture;

in vec2 texcoord;

layout(location = 0) out vec4 outBNAO;

vec3 encodeNormal(const in vec3 normal) {
	return normal * 0.5 + 0.5;
}

void checkSSVisibilitySmooth(
	const in vec3 csPosition, const in sampler2D positionTexture,
	const in mat4 viewMatrix, const in mat4 viewProjectionMatrix,
	const in vec3 wsSample, const in float outlierDistance,
	inout float visibility, inout bool nonOutlier)
{
	// transform world space sample to camera space
	vec3 csSample = (viewMatrix * vec4(wsSample, 1.0)).xyz;

	// project to ndc and then to texture space to sample depth/position buffer to check for occlusion
	vec4 ndcSamplePosition = viewProjectionMatrix * vec4(wsSample, 1.0);
	//if(ndcSamplePosition.w == 0.0) continue;
	ndcSamplePosition.xy /= ndcSamplePosition.w;
	vec2 tsSamplePosition = ndcSamplePosition.xy * 0.5 + 0.5;

	// optimization: replace position buffer with depth buffer
	// here we get a world space position
	// we found the background...
	vec4 wsReferenceSamplePosition = texture(positionTexture, tsSamplePosition);
	if(wsReferenceSamplePosition.w == 0.0) {
		nonOutlier = true;
		//visibility *= 1.0;
		return;
	}
	// transform to camera space
	vec3 csReferenceSamplePosition = (viewMatrix * vec4(wsReferenceSamplePosition.xyz, 1.0)).xyz;

	//////////////////////////////////////////////////////////////////////////
	// optional: apply some code to handle background
	//////////////////////////////////////////////////////////////////////////

	// check for occlusion (within camera space, simple test along z axis; remember z axis goes along -z in OpenGL!)
	// optimized code checks depth values
	// could apply small depth bias here to avoid precision errors
	if(-csReferenceSamplePosition.z < -csSample.z) {
		// we have a occlusion
		// check, if the occlusion is within a application-defined range
		if(abs(csPosition.z - csReferenceSamplePosition.z) < outlierDistance) {
			// valid occlusion
			//visibility = 0.0;
			//visibility *= sqrt(abs(csPosition.z - csReferenceSamplePosition.z) / outlierDistance);
			visibility *= pow(abs(csPosition.z - csReferenceSamplePosition.z) / outlierDistance, 2.0);
			nonOutlier = true;
		}
		else {
			// occluder is too far away, we don't know, what that means for our test
			// test for this sample is "undefined"
		}
	}
	else {
		// we have no occlusion, thus:
		// -csReferenceSamplePosition.z == -csSample.z
		nonOutlier = true;
	}
}

void checkSSVisibilityWithRayMarchingSmooth(
	const in vec3 csPosition, const in sampler2D positionTexture,
	const in mat4 viewMatrix, const in mat4 viewProjectionMatrix,
	const in vec3 wsPosition, const in float outlierDistance,
	const in vec3 ray, const in float sampleRadius,
	const in int rayMarchingSteps, const in float rayMarchingStartOffset,
	inout float visibility, inout bool nonOutlier)
{
	for(int k=0; k<rayMarchingSteps; k++) {
		// world space sample radius within we check for occluders (larger radius needs more ray marching steps)
		vec3 wsSample = wsPosition + ray * (sampleRadius * (float(k) / float(rayMarchingSteps) + rayMarchingStartOffset));
		checkSSVisibilitySmooth(
			csPosition, positionTexture,
			viewMatrix, viewProjectionMatrix,
			wsSample, outlierDistance, 
			visibility, nonOutlier);
	}
}

// normal is parallel to n
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

void createOrthoNormalBasisUnsafe(const in vec3 n, out vec3 tangent, out vec3 binormal, out vec3 normal) {
	normal = normalize(n);

	binormal = vec3(-normal.y, normal.x, 0.0);
	binormal = normalize(binormal);
	tangent = cross(binormal, normal);
}

void main() {
	outBNAO = vec4(0.0);

	//////////////////////////////////////////////////////////////////////////
	// ws = world space
	// cs = camera space
	// ndc = normal device coordinates
	// ts = texture space
	//////////////////////////////////////////////////////////////////////////
	final vec4 wsPosition = texelFetch(positionTexture, ivec2(gl_FragCoord.xy), 0);
	// background
	if(wsPosition.w == 0) return;
	final vec3 csPosition = (viewMatrix * vec4(wsPosition.xyz, 1.0)).xyz;
	final vec3 wsNormal = texelFetch(normalTexture, ivec2(gl_FragCoord.xy), 0).rgb;

	int validTests = 0;
	float ao = 0.0;
	vec3 bentNormal = vec3(0.0);
	float visibilityRays = 0.0;

	// get a different set of samples, depending on pixel position in pattern
	int patternIndex = 
		(int(gl_FragCoord.x) & (patternSize - 1)) + 
		(int(gl_FragCoord.y) & (patternSize - 1)) * patternSize;

	//vec3 u,v,w;
	//createOrthoNormalBasisUnsafe(wsNormal, u, v, w);
	//createOrthoNormalBasis(wsNormal, u, v, w);

	for(int i=0; i<sampleCount; i++) {
		//////////////////////////////////////////////////////////////////////////
		// seed texture holds samples
		// y selects pattern
		// x coordinate gives samples from set
		vec4 data = texelFetch(seedTexture, ivec2(i, patternIndex), 0);
		float rayMarchingStartOffset = data.a;
		vec3 ray = data.rgb;
		//final vec3 ray = data.x*u + data.y*v + data.z*w;
		// bring it to the local hemisphere
		// we do not need a ONB, because we have a uniform distribution
		// simple inversion is ok
		if(dot(ray, wsNormal) < 0.0) {
			ray *= -1.0;
		}

		////////////////////////////////////////////////////////////////////////////
		//// perform ray marching along the direction
		//bool occluded = false;
		//// if the occluder is too far away, we cannot count the sample
		//// screen-space limitation, which could be reduced by depth peeling or scene voxelization
		//bool nonOutlier = false;
		//
		//checkSSVisibilityWithRayMarching(
		//	csPosition, positionTexture,
		//	viewMatrix, viewProjectionMatrix,
		//	wsPosition.xyz, maxDistance,
		//	ray, sampleRadius, rayMarchingSteps, rayMarchingStartOffset,
		//	occluded, nonOutlier);
		////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////
		// perform ray marching along the direction
		float visibility = 1.0;
		// if the occluder is too far away, we cannot count the sample
		// screen-space limitation, which could be reduced by depth peeling or scene voxelization
		bool nonOutlier = false;

		checkSSVisibilityWithRayMarchingSmooth(
			csPosition, positionTexture,
			viewMatrix, viewProjectionMatrix,
			wsPosition.xyz, maxDistance,
			ray, sampleRadius, numRayMarchingSteps, rayMarchingStartOffset,
			visibility, nonOutlier);
		//////////////////////////////////////////////////////////////////////////


		// evaluate the ray marching steps
		// visibility encodes how much this direction is occluded
		// one could also do that binary, but that may cause artifacts, when visibilty is checked in SS

		// note: we assume no occlusion if the occluder is too far away
		// for bent normals we cannot simply "skip" these directions
		bentNormal += ray * visibility;
		visibilityRays += visibility;

		if(nonOutlier) {
			validTests++;
			// extension: weight ao by angle and/or distance to occluder
			ao += visibility;
		}
	}

	if(validTests == 0) {
		outBNAO.a = 0.0;
	}
	else {
		outBNAO.a = ao / float(validTests);
	}
	if(dot(bentNormal, vec3(1.0)) != 0.0) {
		bentNormal /= float(visibilityRays);
		outBNAO.xyz = encodeNormal(bentNormal-wsNormal);
	}
}
