SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> gbufferDiffuseTexture: register(t0);
Texture2D<float4> ssaoTexture: register(t1);


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
	float4 diffuse = gbufferDiffuseTexture.Sample(pointClampSampler, input.texCoord);
	float ssao = ssaoTexture.Sample(pointClampSampler, input.texCoord).x;

	return diffuse * ssao;
}
