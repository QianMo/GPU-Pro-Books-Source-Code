//

struct VSOut
{
	float4 pos 	: SV_Position;
	float2 texc	: TEXCOORD;
};

typedef VSOut PSIn;

VSOut main_vs( uint vertID : SV_VertexID )
{
	VSOut o;

	float4 val[4] = 
	{
		float4(-1,-1,0,1), float4(-1,+1,0,0),
		float4(+1,-1,1,1), float4(+1,+1,1,0)
	};


	o.pos 	= float4( val[ vertID ].xy, 0, 1);
	o.texc	= val[ vertID ].zw;

	return o;
}

Texture2D RT;

SamplerState ss
{
};

float4 main_ps( PSIn i ) : SV_Target0
{
	float4 colr = RT.Sample( ss, i.texc );
	return colr * colr.a;
}

DepthStencilState DSS_Disable
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

		SetDepthStencilState( DSS_Disable, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}
