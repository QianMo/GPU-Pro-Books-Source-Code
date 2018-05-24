SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> depthBufferTexture: register(t0);


cbuffer ConstantBuffer: register(b0)
{
	float nearBegin, nearEnd, farBegin, farEnd;
	float2 projParams;
	float2 padding;
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


float4 PSMain(PS_INPUT input): SV_Target
{
	float depth_ndc = depthBufferTexture.Sample(pointClampSampler, input.texCoord).x;
	float depth = -DepthNDCToView(depth_ndc);
	
	float nearCOC = 0.0f;
	if (depth < nearEnd)
		nearCOC = 1.0f/(nearBegin - nearEnd)*depth + -nearEnd/(nearBegin - nearEnd);
	else if (depth < nearBegin)
		nearCOC = 1.0f;
	nearCOC = saturate(nearCOC);
	
	float farCOC = 1.0f;
	if (depth < farBegin)
		farCOC = 0.0f;
	else if (depth < farEnd)
		farCOC = 1.0f/(farEnd - farBegin)*depth + -farBegin/(farEnd - farBegin);
	farCOC = saturate(farCOC);

	return float4(nearCOC, farCOC, 0.0f, 0.0f);
}
