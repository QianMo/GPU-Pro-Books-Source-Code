cbuffer def
{
	float4x4 matWVP;
}

Texture2D DiffuseTex;

struct VSIn
{
	float4 pos 	: POSITION;
	float2 texc	: TEXCOORD;
};

struct VSOut
{
	float4 pos : SV_Position;
	float2 texc : TEXCOORD;
};

typedef VSOut PSIn;

VSOut main_vs( VSIn i )
{
	VSOut o;
	o.pos 	= mul( i.pos, matWVP );
	o.texc 	= i.texc;

	return o;
}

SamplerState ss
{
	Filter = MIN_MAG_MIP_LINEAR;
};

float4 main_ps( PSIn i ) : SV_Target
{
	return DiffuseTex.Sample( ss, i.texc );
}

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );
	}
}