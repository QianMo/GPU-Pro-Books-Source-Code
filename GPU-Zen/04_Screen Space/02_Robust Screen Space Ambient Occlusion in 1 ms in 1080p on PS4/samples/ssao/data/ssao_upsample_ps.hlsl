SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> depthBufferTexture: register(t0);
Texture2D<float4> depth16Texture_x4: register(t1);
Texture2D<float4> ssaoTexture_x4: register(t2);


cbuffer ConstantBuffer: register(b0)
{
	float2 pixelSize;
	float2 projParams;
}


struct PS_INPUT
{
	float4 position: SV_POSITION;
	float2 texCoord: TEXCOORD0;
};


float DepthNDCToView(float depth_ndc)
{
	return -projParams.y / (depth_ndc + projParams.x);
}


float4 PSMain(PS_INPUT input): SV_Target0
{
	float2 texCoord00 = input.texCoord;
	float2 texCoord10 = input.texCoord + float2(pixelSize.x, 0.0f);
	float2 texCoord01 = input.texCoord + float2(0.0f, pixelSize.y);
	float2 texCoord11 = input.texCoord + float2(pixelSize.x, pixelSize.y);
	
	float depth = depthBufferTexture.Sample(pointClampSampler, input.texCoord).x;
	depth = DepthNDCToView(depth);
	float4 depths_x4 = depth16Texture_x4.GatherRed(pointClampSampler, texCoord00).wzxy;
	float4 depthsDiffs = abs(depth.xxxx - depths_x4);
	
	float4 ssaos_x4 = ssaoTexture_x4.GatherRed(pointClampSampler, texCoord00).wzxy;

	float2 imageCoord = input.texCoord / pixelSize;
	float2 fractional = frac(imageCoord);
	float a = (1.0f - fractional.x) * (1.0f - fractional.y);
	float b = fractional.x * (1.0f - fractional.y);
	float c = (1.0f - fractional.x) * fractional.y;
	float d = fractional.x * fractional.y;

	float4 ssao = 0.0f;
	float weightsSum = 0.0f;

	float weight00 = a / (depthsDiffs.x + 0.001f);
	ssao += weight00 * ssaos_x4.x;
	weightsSum += weight00;

	float weight10 = b / (depthsDiffs.y + 0.001f);
	ssao += weight10 * ssaos_x4.y;
	weightsSum += weight10;

	float weight01 = c / (depthsDiffs.z + 0.001f);
	ssao += weight01 * ssaos_x4.z;
	weightsSum += weight01;

	float weight11 = d / (depthsDiffs.w + 0.001f);
	ssao += weight11 * ssaos_x4.w;
	weightsSum += weight11;

	ssao /= weightsSum;

	return ssao;
}
