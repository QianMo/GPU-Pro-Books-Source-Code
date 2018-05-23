#ifndef BE_TRACING_RAY_H
#define BE_TRACING_RAY_H

/// Ray description.
struct RayDesc
{
	float3 Orig;
	float3 Dir;
};

static const int MaxRayDepth = 0x7fffffff;
static const int MissedRayDepth = 0x7ffffffe;

/// Traced geometry.
struct TracedGeometry
{
	int Depth;
	uint2 Normal;
	uint Diffuse;
	uint Specular;
};

/// Traced light.
struct TracedLight
{
	uint2 Color;
};

/// Debug information.
struct RayDebug
{
	uint RayLength;
};

/// Debug information.
struct DebugInfo
{
	uint ErrorRays;
	uint LastErrorRay;
	uint TotalRayTri;
	uint _1;

	float3 ErrorRayDir;
	uint _2;
};

#include <Utility/Bits.fx>

uint PackTracedDepth(float depth)
{
	return asuint(depth);
}

uint2 PackTracedNormal(float3 normal)
{
	return packf16(normal);
}

uint PackTracedColor(float4 color)
{
	// Gamma compression
	color.xyz = sqrt(color.xyz);
	return packu8(color);
}

uint2 PackTracedLight(float4 color)
{
	return packf16(color);
}

TracedGeometry PackTracedGeometry(float depth, float3 normal, float4 diffuse, float4 specular)
{
	TracedGeometry o;
	o.Depth = PackTracedDepth(depth);
	o.Normal = PackTracedNormal(normal);
	o.Diffuse = PackTracedColor(diffuse);
	o.Specular = PackTracedColor(specular);
	return o;
}

float ExtractTracedDepth(uint depth)
{
	return asfloat(depth);
}

float3 ExtractTracedNormal(uint2 normal)
{
	return unpackf16(normal).xyz;
}

float4 ExtractTracedColor(uint v)
{
	float4 c = unpacku8(v);
	// Gamma compression
	c.xyz *= c.xyz;
	return c;
}

float4 ExtractTracedLight(uint2 color)
{
	return unpackf16(color);
}

#endif