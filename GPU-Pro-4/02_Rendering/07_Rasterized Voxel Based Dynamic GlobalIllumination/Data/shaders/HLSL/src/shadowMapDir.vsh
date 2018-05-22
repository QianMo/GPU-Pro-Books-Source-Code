#include "globals.shi"

GLOBAL_DIR_LIGHT_UB(dirLightUB);

struct VS_INPUT 
{ 
  float3 position: POSITION; 
  float2 texCoords: TEXCOORD; 
  float3 normal: NORMAL; 
  float4 tangent: TANGENT;
};  

struct VS_OUTPUT
{
	float4 position: SV_POSITION;
};

VS_OUTPUT main(VS_INPUT input)
{
  VS_OUTPUT output;   
	output.position = mul(dirLightUB.shadowViewProjMatrix,float4(input.position,1.0f));
	return output;
}