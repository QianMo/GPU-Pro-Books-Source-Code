#version 120
#extension GL_EXT_gpu_shader4 : enable

const float PI = 3.141592;

// The following three textures are the deferred buffers for position, normal and direct radiance.
// They all have the same resolution as the final output.

uniform sampler2D positionTexture;
uniform sampler2D normalTexture;
uniform sampler2D colorTexture;
uniform sampler2D directRadianceTexture;


// The radius in world units the samples that come from a unit sphere are scaled to
uniform float sampleRadius;

// The plain modelview and projection matrices
// We hope, GLSL combines the product modelView * projection product into a single matrix mul
uniform mat4 modelviewMatrix;
uniform mat4 projectionMatrix;


// Strength of the occlusion effect itself
uniform float strength;

// Singularity for distance
uniform float singularity;

uniform float depthBias;

// Strength of the indirect bounce
uniform float bounceStrength;

uniform float bounceSingularity;

uniform sampler2D envmapTexture;

// The number of samples for one pixel
uniform int sampleCount;

// The size in number of pixels of the quasi-random pattern used
uniform int patternSize;

// A texture that contains the random samples in the unit hemisphere
uniform sampler2D seedTexture;

uniform float lightRotationAngle;

vec3 resultColor;

// Helper functions to sample the individual deferred buffers
vec4 sampleBuffer(sampler2D texture, vec2 texCoord) {
	return texelFetch2D(texture, ivec2(texCoord), 0);
}

vec4 samplePosition(vec2 texCoord) {
	return sampleBuffer(positionTexture, texCoord);
}

vec3 sampleNormal(vec2 texCoord) {
	return sampleBuffer(normalTexture, texCoord).rgb;
}

vec4 sampleColor(vec2 texCoord) {
	return sampleBuffer(colorTexture, texCoord).rgba;
}

vec3 sampleDirectRadiance(vec2 texCoord) {
	return sampleBuffer(directRadianceTexture, texCoord).rgb;
}

// compute the local frame (currently no check for singularities)
mat3 computeTripodMatrix(const vec3 direction, const vec3 up = vec3(0.01, 0.99, 0)) 
{			
	vec3 tangent = normalize(cross(up, direction));
	vec3 cotangent = normalize(cross(direction, tangent));
	return mat3(tangent, cotangent, direction);
}

void main() {
	
	// Read position and normal of the pixel to light
	vec3 normal = sampleNormal(gl_FragCoord.xy); 
	vec4 position = samplePosition(gl_FragCoord.xy);
	vec4 pixelColor = sampleColor(gl_FragCoord.xy);           

	// Skip undefined pixels
	if(pixelColor.a > 0.0) {
		// Accumulation over radiance
		vec3 directRadianceSum = vec3(0.0);
		vec3 occluderRadianceSum = vec3(0.0);
		vec3 ambientRadianceSum = vec3(0.0);
		float ambientOcclusion = 0.0;
		
		// A matrix that transforms from the unit hemisphere along z = -1 to the local frame along this normal
		mat3 localMatrix = computeTripodMatrix(normal);
		
		// The index of the current pattern
		// We use one out of patternSize * patternSize pre-defined hemisphere patterns.
		// The i'th pixel in every sub-rectangle uses always the same i-th sub-pattern.
		int patternIndex = int(gl_FragCoord.x) % patternSize + (int(gl_FragCoord.y) % patternSize) * patternSize;
					
		// Loop over all samples from the current pattern
		for(int i = 0; i < sampleCount ; i++) {
		
			// Get the i-th sample direction and tranfrom it to local space.
			vec3 sample = localMatrix * texelFetch2D(seedTexture, ivec2(i, patternIndex), 0).rgb;
			
			vec3 normalizedSample = normalize(sample);			
			
			// Go sample-radius steps along the sample direction, starting at the current pixel world space location
			vec4 worldSampleOccluderPosition = position + sampleRadius * vec4(sample.x, sample.y, sample.z, 0);
			
			// Project this world occluder position in the current eye space using the modelView-projection matrix
			// Note, that we cannot use gl_ModelViewProjection here, cause we work deferred.
			vec4 occluderSamplePosition = (projectionMatrix * modelviewMatrix) * worldSampleOccluderPosition;
			
			// Do a division by w here and map to window coords usign usual GL rules
			vec2 occlusionTexCoord = textureSize2D(positionTexture, 0) * (vec2(0.5) + 0.5 * (occluderSamplePosition.xy / occluderSamplePosition.w));
						
			// Read the occluder position and the occluder normal at the occluder texcoord
			vec4 occluderPosition = samplePosition(occlusionTexCoord);
			vec3 occluderNormal = sampleNormal(occlusionTexCoord);
			
			// remove influence of undefined pixels 
			if (length(occluderNormal) == 0) {
				occluderPosition = vec4(100000.0, 100000.0, 100000.0, 1.0);
			}
			
			// Compute blocking from this occluder
			float depth = (modelviewMatrix * worldSampleOccluderPosition).z;
			float sampleDepth = (modelviewMatrix * occluderPosition).z + depthBias;
					
			// First compute a delta towards the blocker, its length and its normalized version.
			float distanceTerm = abs(depth - sampleDepth) < singularity ? 1.0 : 0.0;
			
			float visibility = 1.0 - strength * (sampleDepth > depth ? 1.0 : 0.0) * distanceTerm;
									
			// Geometric term of the current pixel towards the current sample direction
			float receiverGeometricTerm = max(0.0, dot(normalizedSample, normal));

			// Get the irradiance in the current direction

			float theta = acos(normalizedSample.y);              
			float phi = atan(normalizedSample.z, normalizedSample.x);
			phi += lightRotationAngle;
			if (phi < 0) phi += 2*PI;
			if (phi > 2*PI) phi -= 2*PI;
						
			vec3 senderRadiance = texture2D(envmapTexture, vec2( phi / (2.0*PI), 1.0 - theta / PI ) ).rgb;

			// Compute radiance as the usual triple product of visibility, irradiance and BRDF.
			// Note, that we are not limited to diffuse illumination.
			// For practical reasons, we post-multiply with the diffuse color.

			vec3 radiance = visibility * receiverGeometricTerm * senderRadiance;
			
			// Accumulate the radiance from this sample
			directRadianceSum += radiance;
			
			vec3 ambientRadiance = senderRadiance * receiverGeometricTerm;
			ambientRadianceSum += ambientRadiance;
			ambientOcclusion += visibility;

			// Compute indirect bounce radiance
			// First read sender radiance from occluder			
			vec3 directRadiance = sampleDirectRadiance(occlusionTexCoord);
			
			// Compute the bounce geometric term towards the blocker.
			vec3 delta = position.xyz - occluderPosition.xyz;
			vec3 normalizedDelta = normalize(delta);			
			float unclampedBounceGeometricTerm = 
				max(0.0, dot(normalizedDelta, -normal)) * 
				max(0.0, dot(normalizedDelta, occluderNormal)) /
				max(dot(delta, delta), bounceSingularity);				
			float bounceGeometricTerm = min(unclampedBounceGeometricTerm, bounceSingularity);
			
			// The radiance due to bounce
			vec3 bounceRadiance = bounceStrength * directRadiance * bounceGeometricTerm;						
			
			// Compute radiance from this occcluder (mixing bounce and scatter)
			// vec3 occluderRadiance = bounceRadiance * receiverGeometricTerm;
			vec3 occluderRadiance = bounceRadiance;
			
			// Finally, add the indirect light to the light sum

			occluderRadianceSum += occluderRadiance;			
		}
		
		// Clamp to zero.
		// Althought there should be nothing negative here it is suitable to allow single samples do DARKEN with their contribution.
		// This can be used to exaggerate the directional effect and gives nicely colored shadows (instead of AO).
		directRadianceSum = max(vec3(0), directRadianceSum);
		occluderRadianceSum = max(vec3(0), occluderRadianceSum);

		// Add direct and indirect radiance
		vec3 radianceSum = directRadianceSum + occluderRadianceSum;
		
		// Multiply by solid angle for one sample
		radianceSum *= 2.0 * PI / sampleCount;
				
		// Store final radiance value in the framebuffer		

		gl_FragColor.rgb = radianceSum;		
		// gl_FragColor.rgb = radianceSum * pixelColor.rgb;		
		gl_FragColor.a = 1.0;

	} else {
		// In case we came across an invalid deferred pixel
		gl_FragColor = vec4(0.0);
	}
}
