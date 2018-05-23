#ifndef BE_POINT_LIGHT_H
#define BE_POINT_LIGHT_H

/// Point light constants.
struct PointLightLayout
{
	float3 Right;				///< Light right.
	float3 Up;					///< Light up.
	float3 Dir;					///< Light direction.
	float3 Pos;					///< Light position.

	float4 Color;				///< Light color.

	float Attenuation;			///< Light attenuation.
	float AttenuationOffset;	///< Light attenuation offset.
	float Range;				///< Light range.
	float _pad1;

	float2 ShadowResolution;	///< Shadow resolution.
	float2 ShadowPixel;			///< Shadow pixel (= 1 / resolution).

	float4x4 ShadowProj;		///< Shadow projection matrix.
};

#endif