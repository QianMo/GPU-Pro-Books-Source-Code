#ifndef BE_PERSPECTIVE_H
#define BE_PERSPECTIVE_H

#include "Engine/BindPoints.fx"

/// Perspective constants.
struct PerspectiveLayout
{
	float4x4 ViewProj;		///< View-projection matrix.
	float4x4 ViewProjInv;	///< View-projection matrix inverse.

	float4x4 View;			///< View matrix.
	float4x4 ViewInv;		///< View matrix inverse.

	float4x4 Proj;			///< Projection matrix.
	float4x4 ProjInv;		///< Projection matrix inverse.
	
	float3 CamRight;		///< Camera right.
	float _pad1;
	float3 CamUp;			///< Camera up.
	float _pad2;
	float3 CamDir;			///< Camera direction.
	float _pad3;
	float3 CamPos;			///< Camera position.
	float _pad4;

	float2 NearFarPlane;	///< Near & far planes.

	float Time;				///< Time.
	float TimeStep;			///< Time Step.
};

#ifdef BE_PERSPECTIVE_SETUP
	cbuffer PerspectiveConstants : register(b0)
#else
	cbuffer prebound(PerspectiveConstants) : register(b0)
#endif
{
	PerspectiveLayout Perspective;
}

float3 FromSRGB(float3 c)
{
	return c * c;
}
float4 FromSRGB(float4 c)
{
	c.xyz *= c.xyz;
	return c;
}

float3 ToSRGB(float3 c)
{
	return sqrt(c);
}
float4 ToSRGB(float4 c)
{
	c.xyz = sqrt(c.xyz);
	return c;
}

#endif