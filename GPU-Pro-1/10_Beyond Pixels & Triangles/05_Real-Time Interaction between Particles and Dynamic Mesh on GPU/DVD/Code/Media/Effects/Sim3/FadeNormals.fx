
cbuffer def
{
	float4 fade;
}

struct VSOut
{
	float4 pos	: SV_Position;
};

VSOut main_vs( in uint id : SV_VertexID )
{
	float4 val[4] = 
	{
		float4(-1,-1,0,1), float4(-1,+1,0,0),
		float4(+1,-1,1,1), float4(+1,+1,1,0)
	};

	VSOut o;
	o.pos = float4(val[ id ].xy, 0, 1 );

	return o;
}

struct PSOut
{
	float4 c0 : SV_Target0;
	float4 c1 : SV_Target1;
};

PSOut main_ps()
{
	PSOut o;

	o.c0 = float4(fade.xyz, fade.w);
	o.c1.w = fade.x;

	return o;
}

BlendState BS_Blend
{
	BlendEnable[0] = TRUE;

	SrcBlend = ZERO;
	DestBlend = SRC_COLOR;

	SrcBlendAlpha = INV_SRC1_ALPHA;
	DestBlendAlpha = SRC1_ALPHA;
};

BlendState BS_Default
{
};

DepthStencilState DSS_Disabled
{
	DepthEnable = FALSE;
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

		SetBlendState( BS_Blend, float4(0,0,0,0), 0xffffffff );
		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetBlendState( BS_Default, float4(0,0,0,0), 0xffffffff );
		SetDepthStencilState( DSS_Default, 0 );
	}
}