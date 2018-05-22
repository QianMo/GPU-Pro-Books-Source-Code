#include "globals.shi"

GLOBAL_CAMERA_UB(cameraUB);

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
	float3 posVS: POS_VS;
	float3 normal: NORMAL;
	float3 tangent: TANGENT;
	float3 binormal: BINORMAL;
};

VS_OUTPUT main(VS_INPUT input)
{
  VS_OUTPUT output;
  float4 positionVS = mul(cameraUB.viewMatrix,float4(input.position,1.0f));
	output.position = mul(cameraUB.projMatrix,positionVS); 
	output.texCoords = input.texCoords;
	output.posVS = positionVS.xyz;
        
  output.normal = input.normal;
	output.tangent = input.tangent.xyz;
	output.binormal = cross(input.normal,input.tangent.xyz)*input.tangent.w; 
		
  return output;
}

