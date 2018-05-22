#include "globals.shi"

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
  float2 texCoords: TEXCOORD;
	float3 normal: NORMAL;
};

VS_OUTPUT main(VS_INPUT input)
{
  VS_OUTPUT output;
	output.position = float4(input.position,1.0f);
	output.texCoords = input.texCoords.xy;
	output.normal = input.normal;
  return output;
}

