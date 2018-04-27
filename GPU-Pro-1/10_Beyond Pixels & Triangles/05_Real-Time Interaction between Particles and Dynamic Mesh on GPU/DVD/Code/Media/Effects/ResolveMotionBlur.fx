Texture2D ScreenTex;
Texture2D AccumTex;

struct VSIn
{
	uint id : SV_VertexID;
};

struct VSOut
{
	float4 pos 	: SV_Position;
	float2 texc	: TEXCOORD;
};

typedef VSOut PSIn;

VSOut main_vs( VSIn i )
{
	float4 vdata[4] = {
		{ -1, +1, 0, 0 },
		{ +1, +1, 1, 0 },
		{ -1, -1, 0, 1 },
		{ +1, -1, 1, 1 },
	};

	VSOut o;

	o.pos = float4( vdata[i.id].xy, 0, 1 );
	o.texc = vdata[i.id].zw;

	return o;
}

SamplerState ss
{
	Filter 		= MIN_MAG_LINEAR_MIP_POINT;
	AddressU 	= CLAMP;
	AddressV 	= CLAMP;
};

float4 main_ps( PSIn i ) : SV_Target
{
	float4 colr = ScreenTex.Sample( ss, i.texc );

	float4 accum = AccumTex.Sample( ss, i.texc );

	return (colr*3 + accum) / ( accum.w + 3 );
}

DepthStencilState DSS_Disable
{
	DepthEnable = TRUE;
};

DepthStencilState DSS_Default
{
};

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetDepthStencilState( DSS_Disable, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}