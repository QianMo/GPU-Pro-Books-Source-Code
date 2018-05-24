// http://dev.theomader.com/gaussian-kernel-calculator/


SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> mainTexture: register(t0);


cbuffer ConstantBuffer: register(b0)
{
	float2 pixelSize;
	int from, to;
}


struct PS_INPUT
{
	float4 position: SV_POSITION;
	float2 texCoord: TEXCOORD0;
};


float4 PSMain(PS_INPUT input): SV_Target
{
#ifdef COPY
	return mainTexture.Sample(pointClampSampler, input.texCoord);
#endif
	
#if CHANNELS_COUNT == 1
	#define CHANNELS x
	float result = 0.0f;
#elif CHANNELS_COUNT == 2
	#define CHANNELS xy
	float2 result = 0.0f;
#elif CHANNELS_COUNT == 3
	#define CHANNELS xyy
	float3 result = 0.0f;
#elif CHANNELS_COUNT == 4
	#define CHANNELS xyzw
	float4 result = 0.0f;
#endif
	
	float2 direction = 0.0f;
#ifdef HORIZONTAL
	direction = float2(1.0f, 0.0f);
#endif
#ifdef VERTICAL
	direction = float2(0.0f, 1.0f);
#endif

#ifdef MIN
	result = mainTexture.Sample(pointClampSampler, input.texCoord).CHANNELS;
	for (int i = from; i <= to; i++)
		result = min(result, mainTexture.Sample(pointClampSampler, input.texCoord + i*direction*pixelSize).CHANNELS);
#endif
#ifdef MIN13
	result = mainTexture.Sample(pointClampSampler, input.texCoord).CHANNELS;
	for (int i = -6; i <= 6; i++)
		result = min(result, mainTexture.Sample(pointClampSampler, input.texCoord + i*direction*pixelSize).CHANNELS);
#endif

#ifdef MAX
	result = mainTexture.Sample(pointClampSampler, input.texCoord).CHANNELS;
	for (int i = from; i <= to; i++)
		result = max(result, mainTexture.Sample(pointClampSampler, input.texCoord + i*direction*pixelSize).CHANNELS);
#endif
#ifdef MAX13
	result = mainTexture.Sample(pointClampSampler, input.texCoord).CHANNELS;
	for (int i = -6; i <= 6; i++)
		result = max(result, mainTexture.Sample(pointClampSampler, input.texCoord + i*direction*pixelSize).CHANNELS);
#endif

#ifdef BLUR
	for (int i = from; i <= to; i++)
		result += mainTexture.Sample(pointClampSampler, input.texCoord + i*direction*pixelSize).CHANNELS;
	result /= (to - from + 1.0f);
#endif
#ifdef BLUR13
	for (int i = -6; i <= 6; i++)
		result += mainTexture.Sample(pointClampSampler, input.texCoord + i*direction*pixelSize).CHANNELS;
	result /= 13.0f;
#endif

#if CHANNELS_COUNT == 1
	return result.xxxx;
#elif CHANNELS_COUNT == 2
	return float4(result.xy, 0.0f, 0.0f);
#elif CHANNELS_COUNT == 3
	return float4(result.xyz, 0.0f);
#elif CHANNELS_COUNT == 4
	return result.xyzw;
#endif
}
