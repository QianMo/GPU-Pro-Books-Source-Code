#include "shaders/DeferredRendererShader_ShadowFunctions.h"

// 16-tap Gauss sample distribution with variable size kernel and position-dependent
// jittering. Good for soft shadows (penumbra depends on distance from point of contact).
//
char DRShader_ShadowFunctions::DRSH_Shadow_Gaussian[] = "\n\
float shadowAttenuation(in sampler2D shadowmap, in vec4 point_WCS, in mat4 light_mat) \n\
{ \n\
    vec4 pos_LCS = light_mat*point_WCS; \n\
	vec2 map_coords = vec2( 0.5*pos_LCS.x/pos_LCS.w + 0.5, \n\
	                        0.5*pos_LCS.y/pos_LCS.w + 0.5); \n\
	if ((clamp(map_coords,vec2(0,0),vec2(1,1))-map_coords)!=vec2(0.0,0.0)) \n\
        return 0.0; \n\
	vec4 shadow = texture2D(shadowmap,map_coords); \n\
	int i, numSamples = 16; \n\
	float radius = 1+light_size*abs(2*shadow.z-1+0.001-pos_LCS.z/pos_LCS.w); \n\
	vec2 offset; \n\
	float test=0.0; \n\
	for (i=0; i<numSamples; i++) \n\
	{\n\
	offset = radius*(vec2( kernel[2*i],kernel[2*i+1])+0.5*(texture2D(noise,10*pos_LCS).xy-vec2(0.5,0.5))); \n\
		shadow = texture2D(shadowmap,map_coords+offset/shadow_size); \n\
        test += clamp(sign(2*shadow.z-1+0.004-pos_LCS.z/pos_LCS.w),0.0,1.0); \n\
	}\n\
	return test/numSamples; \n\
} \n";
