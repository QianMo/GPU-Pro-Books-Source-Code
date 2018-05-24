SamplerState anisotropicWrapSampler: register(s2);

Texture2D<float4> mainTexture: register(t0);


struct PS_INPUT
{
	float4 position: SV_POSITION;
	float2 texCoord: TEXCOORD0;
	float3 normal: TEXCOORD1;
};


float4 PSMain(PS_INPUT input): SV_Target
{
	float3 pointToLight = -normalize(float3(-1.0f, -1.0f, -1.0f));
	float NdotL = saturate(dot(pointToLight, normalize(input.normal)));
	
	float lighting = NdotL;
	lighting = 4.0f*pow(lighting, 4.0f) + 0.1f;
	
	return lighting * mainTexture.Sample(anisotropicWrapSampler, input.texCoord);
}
