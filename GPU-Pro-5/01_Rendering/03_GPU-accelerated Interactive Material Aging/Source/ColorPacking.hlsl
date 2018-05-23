#ifndef __COLOR_PACKING__
#define __COLOR_PACKING__

float4 PackColor(float3 diffuseColor, float shininess)
{
	// we have only 256 slots for shininess
	// since 1 to 256 is enough for this application we just scale linearly
	return float4(diffuseColor, shininess / 256);
}

void UnpackColor(float4 packedColor, out float3 diffuseColor, out float shininess)
{
	diffuseColor = packedColor.xyz;
	shininess = packedColor.w * 256;
}
#endif