#ifndef BE_DIRECTIONAL_LIGHT_H
#define BE_DIRECTIONAL_LIGHT_H

/// Directional light constants.
struct DirectionalLightLayout
{
	float3 Right;				///< Light right.
	float _1;
	float3 Up;					///< Light up.
	float _2;
	float3 Dir;					///< Light direction.
	float _3;
	float3 Pos;					///< Light position.
	float _4;

	float4 Color;				///< Light color.
	float4 SkyColor;			///< Sky color.

/*	float Attenuation;			///< Light attenuation.
	float AttenuationOffset;	///< Light attenuation offset.
	float Range;				///< Light range.
	float _pad1;*/
};

/// Shadow split.
struct DirectionalShadowSplit
{
	float4x4 Proj;		///< Shadow projection matrix.
	float2 NearFar;		///< Near / far planes.
	float2 PixelScale;	///< 1 / pixel extents.
};

/// Directional shadow constants.
struct DirectionalShadowLayout
{
	float2 ShadowResolution;	///< Shadow resolution.
	float2 ShadowPixel;			///< Shadow pixel (= 1 / resolution).

	float4 ShadowSplitPlanes;	///< Split plane disances.

	DirectionalShadowSplit ShadowSplits[4];	///< Shadow splits.
};

#endif