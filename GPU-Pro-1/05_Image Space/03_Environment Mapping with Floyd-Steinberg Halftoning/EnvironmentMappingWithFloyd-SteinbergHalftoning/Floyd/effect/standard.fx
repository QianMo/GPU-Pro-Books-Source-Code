#include "enginePool.fx"

/*pooled
// general parameters to be used by any shader
// these are always set by the engine
float4x4 modelMatrix;
float4x4 modelMatrixInverse;
float4x4 modelViewProjMatrix;
float4x4 viewProjMatrix;
float4x4 orientProjMatrixInverse;
float3	 eyePosition;

float4 kdColor;
Texture2D kdMap;
TextureCube envMap;
SamplerState linearSampler
{
	Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

SamplerState pointSampler
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

TrafoOutput
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

QuadOutput vsQuad(QuadInput input)
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
pooled*/

float4 psBackground(QuadOutput input) : SV_TARGET
{
//	return 1;
//	return float4(0.3, 0.3, 1, 1);//envMap.Sample(linearSampler, input.viewDir);
	return envMap.Sample(linearSampler, input.viewDir);
};

technique10 background
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psBackground() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

Texture2D showMap;

float4 psShow(QuadOutput input) : SV_TARGET
{
	return showMap.Sample(linearSampler, input.tex);
};

technique10 show
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psShow() ) );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}