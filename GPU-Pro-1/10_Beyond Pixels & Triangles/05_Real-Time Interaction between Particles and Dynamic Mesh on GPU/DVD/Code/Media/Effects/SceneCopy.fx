//

struct VSOut
{
	float4 pos : SV_Position;
	float2 texc : TEXCOORD;
};

typedef VSOut PSIn;


VSOut main_vs( uint vid : SV_VertexID )
{

	float4 varray[4] = { float4(-1,-1,0,1),
						 float4(-1,+1,0,0),
						 float4(+1,-1,1,1),
						 float4(+1,+1,1,0) };

	VSOut o;
	o.pos 	= float4( varray[vid].xy, 0, 1 );
	o.texc  = varray[vid].zw;

	return o;
}


SamplerState ss
{

	Filter = MIN_MAG_LINEAR_MIP_POINT;
};

Texture2D SrcTex;

// the stupidest resolve ever ;]]
float4 main_ps( PSIn i ) : SV_Target0
{
	return SrcTex.Sample( ss, i.texc );
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