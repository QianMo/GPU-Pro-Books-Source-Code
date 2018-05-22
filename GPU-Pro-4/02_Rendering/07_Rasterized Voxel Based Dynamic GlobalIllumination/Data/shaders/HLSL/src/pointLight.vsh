#include "globals.shi"

GLOBAL_CAMERA_UB(cameraUB);
GLOBAL_POINT_LIGHT_UB(pointLightUB);

struct VS_INPUT
{
  float3 position: POSITION;
};

struct VS_OUTPUT
{
	float4 position: SV_POSITION;
	float4 screenPos: SCREEN_POS;
	float3 viewRay: VIEW_RAY;
};

VS_OUTPUT main(VS_INPUT input)
{
  VS_OUTPUT output;           
	float4 positionWS = mul(pointLightUB.worldMatrix,float4(input.position,1.0f));
	output.position = mul(cameraUB.viewProjMatrix,positionWS);
  output.screenPos = output.position;
	output.screenPos.xy = (float2(output.position.x,-output.position.y)+output.position.ww)*0.5f;
  output.viewRay = positionWS.xyz-cameraUB.position;
	return output;
}