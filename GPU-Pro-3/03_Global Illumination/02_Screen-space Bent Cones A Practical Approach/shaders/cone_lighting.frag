#version 330 core

#define final

#extension GL_ARB_texture_cube_map_array : enable

uniform sampler2D bnAOTexture; // direction

uniform sampler2D normalTexture;
uniform sampler2D diffuseTexture;

uniform samplerCubeArray convolvedEnvMapArray;

uniform float cubeMapArrayLayerCount;

in vec2 texcoord;

out vec4 radianceLayer;

vec4 lerpTextureCubeArray(const in samplerCubeArray textureCUBEArray, const in vec3 direction, const in float index) {
	final vec4 texel0 = texture(textureCUBEArray, vec4(direction, int(index) + 0));
	final vec4 texel1 = texture(textureCUBEArray, vec4(direction, min(float(int(index) + 1), cubeMapArrayLayerCount-1.0)));
	final float weight = fract(index);
	return mix(texel0, texel1, weight);
}

float coneAngleToArrayIndex(const in float angle) {
	return max(0.0, angle * (cubeMapArrayLayerCount-1.0));
}

float arrayIndexFrom01(const in float value) {
	return value * (cubeMapArrayLayerCount-1.0);
}

vec3 decodeNormal(const in vec3 normal) {
	return normal * 2.0 - 1.0;
}

void main() {
	final vec3 normal = texelFetch(normalTexture, ivec2(gl_FragCoord.xy), 0).rgb;
	final vec4 bnAo = texelFetch(bnAOTexture, ivec2(gl_FragCoord.xy), 0);
	final vec3 bentNormal = decodeNormal(bnAo.xyz) + normal;
	final float ao = bnAo.w;

	radianceLayer = vec4(0.0);

	if(dot(normal, vec3(1.0)) == 0.0) {
		return;
	}
	radianceLayer.w = 1.0;

	final float arrayIndex = arrayIndexFrom01(clamp(length(bentNormal) * 2.0 - 1.0, 0.0, 1.0));

	vec3 bentNormalNormalized = normalize(bentNormal);
	final vec3 direction = bentNormalNormalized;

	vec3 diffuseColor = texelFetch(diffuseTexture, ivec2(gl_FragCoord.xy), 0).rgb;

	// geometric term heuristic
	radianceLayer.rgb = lerpTextureCubeArray(convolvedEnvMapArray, direction, arrayIndex).rgb * dot(normal, bentNormalNormalized);
	radianceLayer.rgb *= ao;
	radianceLayer.rgb *= diffuseColor;
	
	//if(texcoord.x > 0.5) {
	//	if(texcoord.y > 0.5) {
	//		// ao only
	//		radianceLayer.rgb = vec3(ao);
	//	}
	//	else {
	//		// ao lighting
	//		radianceLayer.rgb = lerpTextureCubeArray(convolvedEnvMapArray, normal, 0.0).rgb;
	//		radianceLayer.rgb *= ao;
	//		radianceLayer.rgb *= diffuseColor;
	//	}
	//}
	//else {
	//	//radianceLayer.rgb = normal;
	//	if(texcoord.y > 0.5) {
	//		// bent normal lighting
	//		radianceLayer.rgb = lerpTextureCubeArray(convolvedEnvMapArray, direction, 0.0).rgb;
	//		radianceLayer.rgb *= ao;
	//		radianceLayer.rgb *= diffuseColor;
	//	}
	//	else {
	//		// bent cone lighting
	//		radianceLayer.rgb = lerpTextureCubeArray(convolvedEnvMapArray, direction, arrayIndex).rgb * dot(normal, bentNormalNormalized);
	//		radianceLayer.rgb *= ao;
	//		radianceLayer.rgb *= diffuseColor;
	//	}
	//}
}
