
varying vec4 position;
varying vec3 normal;

uniform sampler2D radianceTexture;
uniform sampler2D positionTexture;
uniform sampler2D normalTexture;
uniform sampler2D colorTexture;

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
 	
 	// **** try to avoids this later
 	if (length(normal) == 0.0) {
 		gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
 		return;
 	}
 	
 	vec3 weightedRadiance = vec3(0);
	vec3 radiance = vec3(0);
 	float weightSum = 0.0;
 	
 	for(int i = -kernelSize; i <= kernelSize; i++) {
 			
 		ivec2 sampleTexCoord = texCoord + ivec2(0, i);
 		vec3 sampleRadiance = texelFetch2D(radianceTexture, sampleTexCoord, 0).rgb;
 		vec3 samplePosition = texelFetch2D(positionTexture, sampleTexCoord, 0).rgb;
 		vec3 sampleNormal = texelFetch2D(normalTexture, sampleTexCoord, 0).rgb;
 			
 		if(	(length(samplePosition - position) < positionThreshold) && 
 			(dot(sampleNormal, normal) > (1.0 - normalThreshold)))
 		{
 			float weight = 1.0;//smoothstep(1, 0, sqrt(float(i * i + j * j)) / sqrt(float(2 * kernelSize * kernelSize)));
 			weightedRadiance += weight * sampleRadiance;				
 			weightSum += weight;
 		} 	
		radiance += sampleRadiance;
 	}
 	
 	vec4 result;
	if(weightSum > 0.0) {
 		weightedRadiance /= weightSum;
 		result.rgb = weightedRadiance.rgb; 		
	} else {
 		result.rgb = radiance.rgb / float(2 * kernelSize + 1);
	}
	
	result.a = 1.0;
	gl_FragColor = result;
}
