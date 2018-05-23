#ifndef BE_RAYTRACING_COMPACT_RAY_H
#define BE_RAYTRACING_COMPACT_RAY_H

#include "Pipelines/Tracing/VoxelRep.fx"
#include "Pipelines/Tracing/Ray.fx"
#include "Utility/Math.fx"
#include "Utility/Bits.fx"

static const uint MaxShortUInt = (1 << 16) - 1;
static const uint MaxShortInt = (1 << 15) - 1;
static const uint Max21UInt = (1 << 21) - 1;
static const uint Max10UInt = (1 << 10) - 1;

// Implemented according to the paper: 
// On Floating-Point Normal Vectors
// Quirin Meyer, Jochen Süßmuth, Gerd Sußner, Marc Stamminger and Günther Greiner
// Eurographics Symposium on Rendering 2010

static const float ulgyZSignEps = 0.0001f;
static const float oneMinusUlgyZSignEps = 1.0f - ulgyZSignEps;

uint PackDirection(float3 theNormal)
{
	float3 tmp1 = abs(theNormal);
	float3 tmp2 = theNormal / (tmp1.x + tmp1.y + tmp1.z);
	float2 tmp3 = oneMinusUlgyZSignEps * float2(tmp2.x, tmp2.y);
	float2 tmp4 = tmp2.z <= 0.0f ? tmp3 : sign(tmp3) - tmp3;
	
	float2 onv = 0.5f * (tmp4 + float2(1.0f, 1.0f));
	uint2 onvi = clamp((uint2) (onv * MaxShortUInt), 0, MaxShortUInt);
	return (onvi.y << 16) + onvi.x;
}

float3 ExtractDirection(uint theEncodedNormalI)
{
	float2 theEncodedNormal = uint2(theEncodedNormalI & MaxShortUInt, theEncodedNormalI >> 16) / (float) MaxShortUInt;

	float2 tmp1 = 2.0f * theEncodedNormal - 1.0f;
	float tmp2 = 1.0f - abs(tmp1.x) - abs(tmp1.y);
	float2 tmp3 = tmp2 >= 0.0f ? tmp1 : sign(tmp1) - tmp1;
	tmp2 -= sign(tmp2) * ulgyZSignEps;

	return normalize(float3(tmp3.x, tmp3.y, -tmp2));
}

// Compacts the given ray origin to a grid-relative coordinate composed of two unsigned integers.
uint2 PackCellOrigin(float3 orig)
{
	float3 unitGridOrig = (orig - VoxelRep.Min) * VoxelRep.UnitScale;
	uint3 compactOrig =  (uint3) (unitGridOrig * Max21UInt);

	return uint2( compactOrig.x + (compactOrig.y << 21), (compactOrig.y >> 11) + (compactOrig.z << 10) );
}

// Extracts the grid-relative cell origin packed into the given two unsigned integers.
float3 ExtractCellOrigin(uint2 packedOrig)
{
	uint3 compactOrig = uint3(
			packedOrig.x,
			(packedOrig.x >> 21) + (packedOrig.y << 11),
			packedOrig.y >> 10
		) & Max21UInt;

	float3 unitGridOrig =  compactOrig / (float) Max21UInt;

	return VoxelRep.Min + VoxelRep.Ext * unitGridOrig;
}

#endif