/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Milan Magdics
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for any non-commercial programs.
 * 
 * Use it for your own risk. The author(s) do(es) not take
 * responsibility or liability for the damages or harms caused by
 * this software.
**********************************************************************
*/

// culling functions. only frustum culling is implemented here

#include "modules.fx"

cbuffer cullingSettings
{
	int enableModuleCulling;
}

cbuffer cameraParams
{
	float3 cameraPos;
	float farClip;

	float4 p_left;
	float4 p_right;
	float4 p_top;
	float4 p_bottom;
	float4 p_near;
	float4 p_far;
}

bool clipPlane( float4 plane, float3 pos, float size )
{
	return dot( plane, float4(pos,1) ) < -size;
}

// culling with bounding sphere is implemented here
//   and in this example, it is much bigger than is should be
// (i.e. less modules are culled)
// implement other culling methods here, like AABB
const float cullSizeScaler = sqrt(2.0);
bool cullFrustumModuleSphere( Module module )
{
	bool output = false;
	output = output || clipPlane( p_near, module.position.xyz, module.size * cullSizeScaler );
	output = output || clipPlane( p_far, module.position.xyz, module.size * cullSizeScaler );
	output = output || clipPlane( p_left, module.position.xyz, module.size * cullSizeScaler );
	output = output || clipPlane( p_right, module.position.xyz, module.size * cullSizeScaler );
	output = output || clipPlane( p_top, module.position.xyz, module.size * cullSizeScaler );
	output = output || clipPlane( p_bottom, module.position.xyz, module.size * cullSizeScaler );
	return output;
}


//********************************
// Function of Module culling step
//********************************

bool cullModule
( 
	Module module					// input module to cull
)
{
	if ( enableModuleCulling )
		return cullFrustumModuleSphere( module );
	else
		return false;
}