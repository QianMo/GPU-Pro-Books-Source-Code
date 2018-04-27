cbuffer def
{
	uint 	rndDisp;
	float   particleLifeTime;
	float3	particleSpeed;
	float3	startPosition;
	float	radius;
}

struct VSIn
{
	uint vidx : SV_VertexID;
};

struct VSOut
{
	float4 pos_time	: POSITION;
	float3 speed    : SPEED;
};

Buffer<float4> RandPoses;

VSOut main_vs( VSIn i )
{
	VSOut o;

	o.pos_time.xyz	= startPosition + (RandPoses.Load( i.vidx + rndDisp ) - 0.5)* radius;
	o.pos_time.w	= particleLifeTime;

	o.speed			= particleSpeed;

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
		SetGeometryShader( ConstructGSWithSO( vs, "POSITION.xyzw;SPEED.xyz") );
		SetPixelShader( NULL );

		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}