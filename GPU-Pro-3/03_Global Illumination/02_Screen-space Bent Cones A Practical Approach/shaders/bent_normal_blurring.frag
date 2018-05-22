#version 330 core

#define final

uniform sampler2D positionTexture;
uniform sampler2D normalTexture;

uniform float positionPower;
uniform float normalPower;

uniform int kernelSize;
uniform ivec2 maskDirection;
uniform int subSampling;

uniform sampler2D inputTexture;

//#ifndef SPATIAL_WEIGHTS_SET
//const float spatialWeights[] = float[2](1.0, 1.0);
//#endif

out vec4 outLayer;

vec3 encodeNormal(const in vec3 normal) {
	return normal * 0.5 + 0.5;
}

vec3 decodeNormal(const in vec3 normal) {
	return normal * 2.0 - 1.0;
}

void main() {
	final ivec2 texCoord = ivec2(gl_FragCoord.xy);
	
	final vec4 position = texelFetch(positionTexture, texCoord, 0);
	if(position.w == 0.0) {
		outLayer = vec4(0.0);
		return;
	}
	final vec3 normal = texelFetch(normalTexture, texCoord, 0).rgb;
	
	vec3 bentNormal = vec3(0.0);
	float ao = 0.0;
	float weightSum = 0.0;

	for(int i=0; i<kernelSize; i+=max(1, subSampling)) {
		final ivec2 sampleTexCoord = texCoord + (i - kernelSize/2) * maskDirection;

		final vec3 sampleNormal = texelFetch(normalTexture, sampleTexCoord, 0).rgb;
		final vec4 samplePosition = texelFetch(positionTexture, sampleTexCoord, 0);
		if(samplePosition.w == 0.0) continue;
		// The alpha of the position texture should be used !
		// If it is zero, we are undefined!

		final float normalWeight = pow(dot(sampleNormal, normal) * 0.5 + 0.5, normalPower);

		final float positionWeight = 1.0 / pow(1.0 + distance(position.xyz, samplePosition.xyz), positionPower);

		//// we use a simple box filter to exactly blur the pattern (spatialWeights[?] == 1.0)
		//final float weight = normalWeight * positionWeight * spatialWeights[i];
		final float weight = normalWeight * positionWeight;

		final vec4 sampleData = texelFetch(inputTexture, sampleTexCoord, 0);

		final float aoSample = sampleData.a;
		ao += weight * aoSample;
		
		final vec3 bentNormalSample = decodeNormal(sampleData.rgb);
		bentNormal += weight * bentNormalSample;

		weightSum += weight;
 	}

	outLayer.rgb = encodeNormal(bentNormal / weightSum);
	outLayer.a = ao / weightSum;
}
