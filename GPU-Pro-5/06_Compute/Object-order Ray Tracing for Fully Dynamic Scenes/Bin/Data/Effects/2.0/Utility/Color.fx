#ifndef BE_UTILITY_COLOR_H
#define BE_UTILITY_COLOR_H

#include "Utility/Bits.fx"

/// Converts the given color vector into a 32-bit color value, applying gamma compression.
uint packu8_srgb(float4 color)
{
	// Gamma compression
	color.xyz = sqrt(color.xyz);
	return packu8(color);
}

/// Converts the given 32-bit color value into a color vector, revoking gamma compression.
float4 unpacku8_srgb(uint v)
{
	float4 c = unpacku8(v);
	// Gamma compression
	c.xyz *= c.xyz;
	return c;
}

/// Converts the given color vector into a 16-bit color value.
uint packcolor16(float3 color)
{
	uint3 p = (uint3) floor(color * float3(31.0f, 63.0f, 31.0f));
	p = p << uint3(11, 5, 0);
	return p.x | p.y | p.z;
}

/// Converts the given color vector into a 16-bit color value, applying gamma compression.
uint packcolor16_srgb(float3 color)
{
	// Gamma compression
	color.xyz = sqrt(color.xyz);
	return packcolor16(color);
}

/// Converts the given 16-bit color value into a color vector.
float3 unpackcolor16(uint v)
{
	return ( (v >> uint3(11, 5, 0)) & uint3(0x1F, 0x3F, 0x1F) ) / float3(31.0f, 63.0f, 31.0f);
}

/// Converts the given 16-bit color value into a color vector, revoking gamma compression.
float3 unpackcolor16_srgb(uint v)
{
	float3 c = unpackcolor16(v);
	// Gamma compression
	c.xyz *= c.xyz;
	return c;
}

#endif