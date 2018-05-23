

#include "CommonConstants.hlsl"


#include "RawImpl\\RawColor.hlsl"

#include "RawImpl\\RawUV.hlsl"

struct VS_RAW_INPUT
{
    float4 Pos : POSITION;
};

struct PS_RAW_INPUT
{
    float4 Pos : SV_POSITION;
};

///< Vertex Shader
PS_RAW_INPUT RawVS(VS_RAW_INPUT _input)
{
	PS_RAW_INPUT output;
		
	output.Pos = mul( float4(_input.Pos.xyz,1), World);
    output.Pos = mul( output.Pos,View );
    output.Pos = mul( Proj, output.Pos );
    	
    return output;    
}

///< Pixel Shader
float4 RawPS( PS_RAW_INPUT _input) : SV_Target
{
    return float4(0.7f, 0.7f, 0.7f, 1.0f);
}



