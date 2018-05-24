SamplerState anisotropicWrapSampler: register(s2);

Texture2D<float4> diffuseTexture: register(t0);


struct PS_INPUT
{
	float4 position: SV_POSITION;
	float2 texCoord: TEXCOORD0;
	float3 normal: TEXCOORD1;
};


struct PS_OUTPUT
{
	float4 diffuse: SV_Target0;
	float4 normal: SV_Target1;
};


PS_OUTPUT PSMain(PS_INPUT input)
{
	PS_OUTPUT output = (PS_OUTPUT)0;

	float3 normal = normalize(input.normal);
	normal = 0.5f*normal + 0.5f;
	
	output.diffuse = diffuseTexture.Sample(anisotropicWrapSampler, input.texCoord);
	output.normal = float4(normal, 0.0f);

	return output;
}
