//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
Texture2D txDiffuse : register( t0 );
SamplerState samLinear : register( s0 );

struct VS_INPUT
{
    float4 Pos	: POSITION;
    float2 Tex	: TEXCOORD;
    float4 Color: COLOR;    
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD;
    float4 Color: COLOR;
};

cbuffer ConstantBuffer : register( cb0 )
{    	
     matrix  projectionMatrix;
     matrix  modelMatrix;
};

PS_INPUT VS( VS_INPUT input )
{
    PS_INPUT output = (PS_INPUT)0;

    input.Pos.w = 1.0f;
	
    output.Pos = mul( input.Pos, modelMatrix );
    output.Pos = mul( output.Pos, projectionMatrix );
 
    output.Tex = input.Tex;

    output.Color = input.Color;
    
    return output;
}
