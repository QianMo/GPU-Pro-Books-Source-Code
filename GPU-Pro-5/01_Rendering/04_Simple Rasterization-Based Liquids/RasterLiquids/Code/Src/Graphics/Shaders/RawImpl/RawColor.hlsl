
struct VS_RAW_COLOR_INPUT
{
    float4 Pos : POSITION;
    float4 Color : COLOR;
};

struct PS_RAW_COLOR_INPUT
{
    float4 Pos : SV_POSITION;
    float4 Color : COLOR0;
};


///< Vertex Shader
PS_RAW_COLOR_INPUT RawColorScreenVS(VS_RAW_COLOR_INPUT _input)
{
	PS_RAW_COLOR_INPUT output;
		
	output.Pos = mul(float4(_input.Pos.xyz,1), World );

    output.Color = _input.Color;
	
    return output;    
}

///< Vertex Shader
PS_RAW_COLOR_INPUT RawColorVS(VS_RAW_COLOR_INPUT _input)
{
	PS_RAW_COLOR_INPUT output;
		
	output.Pos = mul( float4(_input.Pos.xyz,1),World);
    output.Pos = mul( output.Pos,View );
    output.Pos = mul( Proj, output.Pos );
    
    output.Color = _input.Color;
	
    return output;    
}

///< Pixel Shader
float4 RawColorPS( PS_RAW_COLOR_INPUT _input) : SV_Target
{
    return _input.Color;
}
