#include "globals.shi"

struct VS_INPUT
{
  float3 position: POSITION; 
  float2 texCoords: TEXCOORD;
};

struct VS_OUTPUT
{
	float4 position: SV_POSITION;
	uint instanceID: INSTANCE_ID;
};

VS_OUTPUT main(VS_INPUT input,uint instanceID: SV_InstanceID)
{
  VS_OUTPUT output;
	output.position = float4(input.position,1.0f);
	output.instanceID = instanceID;
  return output;
}
