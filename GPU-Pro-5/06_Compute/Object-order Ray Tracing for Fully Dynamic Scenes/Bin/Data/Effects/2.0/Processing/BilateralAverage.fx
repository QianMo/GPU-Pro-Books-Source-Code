#ifndef BE_PROCESSING_BILATERAL_AVERAGE_H
#define BE_PROCESSING_BILATERAL_AVERAGE_H

#include <Pipelines/LPR/Geometry.fx>

float4 PSBilateralAverage(float4 Position : SV_Position, float2 TexCoord : TexCoord0,
	uniform float2 dir, uniform int lradius, uniform int rradius,
	uniform Texture2D geometryTex, uniform SamplerState geometrySampler,
	uniform Texture2D sourceTex, uniform SamplerState sourceSampler, uniform float2 sourcePixel,
	uniform float2 geometryOffset = 0,
	uniform int sourceLevel = 0) : SV_Target0
{
	float2 geometryTexCoord = TexCoord + geometryOffset;

	float4 eyeGeometry = geometryTex.SampleLevel(geometrySampler, geometryTexCoord, 0);
	float eyeDepth = ExtractDepth(eyeGeometry);
	float3 eyeNormal = ExtractNormal(eyeGeometry);

	float2 delta = dir * sourcePixel;

	float4 acc = 0.0f;
	float accWeight = 0.0f;

	[unroll] for (int i = -lradius; i <= rradius; ++i)
	{
		float2 sampleOffset = i * delta;
		float2 sampleCoord = TexCoord + sampleOffset;

		float4 sampleGeometry = geometryTex.SampleLevel(geometrySampler, geometryTexCoord + sampleOffset, 0);
		float sampleDepth = ExtractDepth(sampleGeometry);
		float3 sampleNormal = ExtractNormal(sampleGeometry);

		float sampleWeight = saturate( 1 - 4 * abs(sampleDepth - eyeDepth) )
			* saturate( dot(eyeNormal, sampleNormal) );

		acc += sampleWeight * sourceTex.SampleLevel(sourceSampler, sampleCoord, sourceLevel);
		accWeight += sampleWeight;
	}
	
	return acc / accWeight;
}

#endif