Texture2D normalTexture;
Texture2D alphaTexture;

struct IaosTbnTrafo
{
    float4 pos			: POSITION;
    float3 normal		: NORMAL;
    float2 tex			: TEXCOORD;
//    float3 tangent		: TANGENT;
//    float3 binormal		: BINORMAL;
};

struct VsosTbnTrafo
{
    float4 pos			: SV_POSITION;
	float4 worldPos		: WORLDPOS;
    float3 normal		: NORMAL;
    float2 tex			: TEXCOORD;
//    float3 tangent		: TANGENT;
//    float3 binormal		: BINORMAL;
};

VsosTbnTrafo vsTbnTrafo(IaosTbnTrafo input)
{
	VsosTbnTrafo output = (VsosTbnTrafo)0;

	output.pos = mul(input.pos,
		modelViewProjMatrix);
	output.worldPos = mul(input.pos,
		modelMatrix);
	output.normal = mul(modelMatrixInverse,
		float4(input.normal.xyz, 0.0));
//	output.tangent = mul(modelMatrixInverse,
//		float4(input.tangent.xyz, 0.0));
//	output.binormal = mul(modelMatrixInverse,
//		float4(input.binormal.xyz, 0.0));
	output.tex = input.tex;
	return output;
}


float4 psDefer(VsosTbnTrafo input) : SV_Target
{
/*	float3x3 tbnFrame = (float3x3(normalize(input.tangent), normalize(input.binormal), normalize(input.normal) ));
	float3 tbnNormal = normalize(normalTexture.Sample(linearSampler, input.tex).xyz * float3(2, 2, 1) - float3(1, 1, 0));
	tbnNormal *= tbnNormal.z;
	tbnNormal.z = 1;
	float3 worldNormal = mul(tbnNormal, tbnFrame);*/
	float3 worldNormal = input.normal;

	float3 shade = 
		kdTexture.Sample(linearSampler, input.tex).xyz * (0.05 + 0.95 * abs(dot(float3(0.5, 0.5, 0.5), normalize(worldNormal))));
	if(any(isnan(shade)))
		shade = float3(1, 0, 0);
	return float4(shade, length(input.worldPos - eyePos));
}

float4 psDeferBackground(VsosQuad input) : SV_Target
{
	return float4(envTexture.Sample(linearSampler, input.viewDir).xyz, 100000);
}
