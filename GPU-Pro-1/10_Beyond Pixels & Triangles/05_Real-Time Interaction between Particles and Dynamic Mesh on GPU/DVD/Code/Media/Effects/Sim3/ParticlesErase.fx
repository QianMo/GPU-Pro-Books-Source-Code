
struct VSIn
{
};

struct VSOut
{
	float3 pos		: POSITION;
	float3 speed	: SPEED;
};

VSOut main_vs( VSIn i )
{	
	VSOut o;

	o.pos 	= -8.5070591730234615865843651857942e+37;
	o.speed = 0.f;

	return o;
}


DepthStencilState DSS_Disabled
{
	DepthEnable = FALSE;
};

DepthStencilState DSS_Default
{
};

VertexShader vs = CompileShader( vs_4_0, main_vs() );

technique10 main
{
	pass
	{
		SetVertexShader( vs );
		SetGeometryShader( ConstructGSWithSO( vs, "POSITION.xyz;SPEED.xyz" ) );
		SetPixelShader( 0 );

		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}