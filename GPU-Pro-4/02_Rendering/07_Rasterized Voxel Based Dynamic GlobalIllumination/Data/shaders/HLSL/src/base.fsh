#include "globals.shi"

Texture2D colorMap: register(COLOR_TEX_BP);
SamplerState colorMapSampler: register(COLOR_SAM_BP);
Texture2D normalMap: register(NORMAL_TEX_BP);
SamplerState normalMapSampler: register(NORMAL_SAM_BP);
Texture2D specularMap: register(SPECULAR_TEX_BP);
SamplerState specularMapSampler: register(SPECULAR_SAM_BP);

struct VS_OUTPUT
{
	float4 position: SV_POSITION;
  float2 texCoords: TEXCOORD;
	float3 posVS: POS_VS;
	float3 normal: NORMAL;
	float3 tangent: TANGENT;
	float3 binormal: BINORMAL;
};

struct FS_OUTPUT
{
  float4 fragColor0: SV_TARGET0;
	float4 fragColor1: SV_TARGET1;
};

#define ALPHA_THRESHOLD 0.3f

FS_OUTPUT main(VS_OUTPUT input) 
{
  FS_OUTPUT output;

	float4 albedo = colorMap.Sample(colorMapSampler,input.texCoords);   
	
#ifdef ALPHA_TEST
	if(albedo.a<ALPHA_THRESHOLD)
		discard;
#endif

  float3x3 tangentMatrix;
	tangentMatrix[0] = normalize(input.tangent);
	tangentMatrix[1] = normalize(input.binormal);
  tangentMatrix[2] = normalize(input.normal);
	
	float3 bump = normalMap.Sample(normalMapSampler,input.texCoords).xyz*2.0f-1.0f;
	bump = mul(bump,tangentMatrix);
	bump = normalize(bump);
	float gloss = specularMap.Sample(specularMapSampler,input.texCoords).r;

	output.fragColor0 = float4(albedo.rgb,gloss);

	output.fragColor1 = float4(bump,length(input.posVS));

  return output;
}


