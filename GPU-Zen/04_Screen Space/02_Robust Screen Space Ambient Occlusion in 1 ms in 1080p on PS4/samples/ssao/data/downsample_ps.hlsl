SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> depthBufferTexture: register(t0);


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
	float2 texCoord = input.texCoord + float2(-0.25f, -0.25f)*pixelSize;
	
	float depth_ndc = depthBufferTexture.Sample(pointClampSampler, texCoord).x;
	float depth = DepthNDCToView(depth_ndc);
	
	return depth;
}
