

Texture2D		txDiffuse	: register( t0 );

struct VS_RAW_UV_INPUT
{
    float4 Pos : POSITION;
    float2 UV : TEXCOORD0;
};

struct PS_RAW_UV_INPUT
{
    float4 Pos : SV_POSITION;
    float2 UV : TEXCOORD0;
};

///< Vertex Shader
PS_RAW_UV_INPUT PostVS(VS_RAW_UV_INPUT _input)
{
	PS_RAW_UV_INPUT output;
		
	output.Pos=_input.Pos;    
    output.UV = _input.UV;
	
    return output;    
}

///< Vertex Shader
PS_RAW_UV_INPUT RawUVScreenVS(VS_RAW_UV_INPUT _input)
{
	PS_RAW_UV_INPUT output;
		
	output.Pos = mul( float4(_input.Pos.xyz,1),World );

    output.UV = _input.UV;
	
    return output;    
}

///< Vertex Shader
PS_RAW_UV_INPUT RawUVVS(VS_RAW_UV_INPUT _input)
{
	PS_RAW_UV_INPUT output;
		
	output.Pos = mul( float4(_input.Pos.xyz,1),World);
    output.Pos = mul( output.Pos,View );
    output.Pos = mul( Proj, output.Pos );
    
    output.UV = _input.UV;
	
    return output;    
}

///< Pixel Shader
float4 RawUVPS( PS_RAW_UV_INPUT _input) : SV_Target
{
	float4 p = txDiffuse.Sample(samLinear,_input.UV);	
	return p;
}

///< Pixel Shader
float4 RenderDensityField_PS( PS_RAW_UV_INPUT _input) : SV_Target
{
	float2 d = 1.0f*txDiffuse.Sample(samLinear,_input.UV).xy;
	
	return float4(d,0,1);
}

float4 RenderHeightField_PS( PS_RAW_UV_INPUT _input) : SV_Target
{
	float p = (txDiffuse.Sample(samLinear,_input.UV).w)*0.02f;
	return float4(p,p,p,1);//float4(0,p*0.5f,p*0.8f, 1);
}