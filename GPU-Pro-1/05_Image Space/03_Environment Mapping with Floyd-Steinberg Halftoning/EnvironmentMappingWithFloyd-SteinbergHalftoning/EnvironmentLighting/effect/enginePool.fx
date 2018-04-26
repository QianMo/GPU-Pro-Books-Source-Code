#include "blendStates.fx"
#include "depthStencilStates.fx"
#include "rasterizerStates.fx"

shared cbuffer standard
{
float4x4 modelMatrix;
float4x4 modelMatrixInverse;
float4x4 modelViewProjMatrix;
float4x4 viewProjMatrix;
float4x4 orientProjMatrixInverse;
float3	 eyePosition;
float4 kdColor;
}

shared Texture2D kdMap;
shared Texture2D normalMap;
shared TextureCube envMap;
shared SamplerState linearSampler
{
	Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

shared SamplerState clampSampler
{
	Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

shared SamplerState pointSampler
{
	Filter = MIN_MAG_MIP_POINT;
    AddressU = Wrap;
    AddressV = Wrap;
};

// basic vertex shader performing classic geometric transformations
struct TrafoInput
{
    float4 pos			: POSITION;
    float3 normal		: NORMAL;
    float2 tex			: TEXCOORD0;
};

struct TrafoOutput
{
    float4 pos			: SV_POSITION;
    float3 normal		: TEXCOORD2;
    float2 tex			: TEXCOORD0;
    float4 worldPos		: TEXCOORD1;
};

shared TrafoOutput
	vsTrafo(TrafoInput input)
{
	TrafoOutput output = (TrafoOutput)0;
	output.pos = mul(input.pos, modelViewProjMatrix);

	output.worldPos = mul(input.pos, modelMatrix);
	
	output.normal = mul(modelMatrixInverse, float4(input.normal.xyz, 0.0));
	output.tex = input.tex;
	
	return output;
}


// vertex shader for rendeing full viewport quads, computing eye direction
struct QuadInput
{
	float4  pos			: POSITION;
	float2  tex			: TEXCOORD0;
};

struct QuadOutput
{
	float4 pos				: SV_POSITION;
	float2 tex				: TEXCOORD0;
	float3 viewDir			: TEXCOORD1;
};

shared QuadOutput vsQuad(QuadInput input)
{
	QuadOutput output = (QuadOutput)0;

	output.pos = input.pos;
    float4 hWorldPosMinusEye = mul(input.pos, orientProjMatrixInverse);
    hWorldPosMinusEye /= hWorldPosMinusEye.w;
    output.viewDir = hWorldPosMinusEye.xyz;
    output.pos.z = 0.99999;
    output.tex = input.tex;
    return output;
};
