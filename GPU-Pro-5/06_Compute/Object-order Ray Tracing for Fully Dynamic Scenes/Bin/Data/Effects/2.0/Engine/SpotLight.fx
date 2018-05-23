#ifndef BE_SPOT_LIGHT_H
#define BE_SPOT_LIGHT_H

/// Spot light constants.
struct SpotLightLayout
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

	float CosInnerAngle;		///< Light inner cone angle.
	float CosOuterAngle;		///< Light outer cone angle.
	float SinInnerAngle;		///< Light inner cone angle.
	float SinOuterAngle;		///< Light outer cone angle.

	float2 ShadowResolution;	///< Shadow resolution.
	float2 ShadowPixel;			///< Shadow pixel (= 1 / resolution).

	float4x4 ShadowProj;		///< Shadow projection matrix.
};

#endif