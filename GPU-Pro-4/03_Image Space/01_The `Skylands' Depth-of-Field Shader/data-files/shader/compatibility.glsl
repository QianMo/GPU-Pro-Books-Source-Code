// -*- c++ -*-
#ifndef compatability_glsl
#define compatability_glsl
#line 1005
/**
 Support for some GL 4.0 shader calls on older versions of OpenGL, and 
 support for some HLSL types and functions
 */      
#extension GL_EXT_gpu_shader4 : require

#if __VERSION__ == 120
#   define texture      texture2D
#   define texelFetch   texelFetch2D
#   define textureSize  textureSize2D
#endif

/////////////////////////////////////////////////////////////////////////////
// HLSL compatability

#define int2 ivec2
#define int3 ivec3
#define int4 ivec4

#define float2 vec2
#define float3 vec3
#define float4 vec4

#define bool2 bvec2
#define bool3 bvec3
#define bool4 bvec4

#define half float
#define half2 vec2
#define half3 vec3
#define half4 vec4

#define tex2D texture2D

#define lerp mix

float saturate(float x) {
	return clamp(x, 0.0, 1.0);
}

float2 saturate(float2 x) {
	return clamp(x, float2(0.0), float2(1.0));
}

float3 saturate(float3 x) {
	return clamp(x, float3(0.0), float3(1.0));
}

float4 saturate(float4 x) {
	return clamp(x, float4(0.0), float4(1.0));
}

#endif
