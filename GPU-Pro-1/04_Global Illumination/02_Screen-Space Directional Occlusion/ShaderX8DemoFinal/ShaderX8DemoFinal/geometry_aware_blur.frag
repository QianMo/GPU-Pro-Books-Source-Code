
varying vec4 position;
varying vec3 normal;

uniform sampler2D radianceTexture;
uniform sampler2D positionTexture;
uniform sampler2D normalTexture;
uniform sampler2D colorTexture;
uniform sampler2D directLightTexture;

uniform float positionThreshold;
uniform float normalThreshold;

uniform float maxRadiance;

uniform int kernelSize;


void main()
{	
 	ivec2 texCoord = ivec2(gl_FragCoord.xy);
 	
 	vec3 position = texelFetch2D(positionTexture, texCoord, 0).rgb;
 	vec3 normal = texelFetch2D(normalTexture, texCoord, 0).rgb;
 	vec3 color = texelFetch2D(colorTexture, texCoord, 0).rgb;
 	
	// skip undefined pixels
 	if (length(normal) == 0.0) {
 		gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
 		return;
 	}
 	
 	vec3 weightedRadiance = vec3(0);
	vec3 radiance = vec3(0);
 	float weightSum = 0.0;
 	
 	for(int i = -kernelSize; i <= kernelSize; i++) {
 		
 		for(int j = -kernelSize; j <= kernelSize; j++) {
 			
 			ivec2 sampleTexCoord = texCoord + ivec2(j, i);
 			vec3 sampleRadiance = texelFetch2D(radianceTexture, sampleTexCoord, 0).rgb;
 			vec3 samplePosition = texelFetch2D(positionTexture, sampleTexCoord, 0).rgb;
 			vec3 sampleNormal = texelFetch2D(normalTexture, sampleTexCoord, 0).rgb;
 			
 			if(	(length(samplePosition - position) < positionThreshold) && 
 				(dot(sampleNormal, normal) > (1.0 - normalThreshold)))
 			{
 				float weight = 1.0;
 				weightedRadiance += weight * sampleRadiance;				
 				weightSum += weight;
 			} 	
			radiance += sampleRadiance;
 		}
 	}

 	// multiply by surface reflectance
	radiance *= color;
 	weightedRadiance *= color;
 	
 	vec4 result;
	if(weightSum > 0.0) {
 		weightedRadiance /= weightSum;
 		result.rgb = weightedRadiance.rgb; 	   // average of valid pixels	
	} 
	else {
 		result.rgb = radiance.rgb / float((2 * kernelSize + 1) * (2 * kernelSize + 1));    // simple average if weightSum == 0
	}

	// add direct light and indirect light
	vec3 directLight = texelFetch2D(directLightTexture, texCoord, 0).rgb;
	result.rgb = 1.0 * result.rgb + 0.5 * directLight;

	// simple gamma tone mapper 
	float greyRadiance = max(0.001, 0.3 * result.r + 0.6 * result.g + 0.1 * result.b);
	float mappedRadiance = pow(min(greyRadiance / maxRadiance, 1.0), 1.0/2.2);
	result.rgb = (mappedRadiance / greyRadiance) * result.rgb; 

	result.a = 1.0;
	gl_FragColor = result;
}
