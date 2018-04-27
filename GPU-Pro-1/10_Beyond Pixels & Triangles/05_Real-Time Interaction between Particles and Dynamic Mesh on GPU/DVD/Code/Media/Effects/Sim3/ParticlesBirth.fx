cbuffer def
{
	uint 	numMeshTris;
	uint 	randomsMask;
	uint2	randomDisplace;
	float 	rcpTimePassed;
	float	lifeTime;
};

struct VSIn
{
	uint vidx : SV_VertexID;
};

struct VSOut
{
	float4 pos_time	: POSITION;
	float3 speed	: SPEED;
};

Buffer<uint4> ibuff;
Buffer<float3> vbuff;
Buffer<float3> prevVBuff;
Buffer<float2> randBuff;

float3 bilerp( float3 v1, float3 v2, float3 v3, float2 bicoefs )
{
	return lerp( lerp( v1, v2, bicoefs[0] ), v3, bicoefs[1] );
}

VSOut main_vs( VSIn i )
{
	uint triBase	= asuint( randBuff.Load( randomDisplace.x + i.vidx ) );
	uint randBase	= asuint( randBuff.Load( randomDisplace.y + i.vidx ) );

	uint triIdx 		= triBase % numMeshTris;

	// Using bitwise AND operation(&) produced a hardware/driver
	// bug once ('asuint'ing .Load result and than ANDing it produced erroneous 
	// behaviour). Worked OK on an alternative card though.
#if 0
	uint randIdx 		= randBase & randomsMask;
#else
	uint randIdx 		= randBase % ( randomsMask + 1 );
#endif

	uint3 idxes_x2 		= ibuff.Load( triIdx ) * 2;

	float3 pos[3];
	float3 speed[3];

	float2 bicoefs 	= randBuff.Load( randIdx );
	
	[unroll]
	for( int i = 0; i < 3; i ++ )
	{
		pos[i] 		= vbuff.Load( idxes_x2[i] );
		speed[i] 	= pos[i] - prevVBuff.Load( idxes_x2[i] );
	}
	
	float3 finalPos 	= bilerp( pos[0], pos[1], pos[2], bicoefs );

	float3 finalSpeed   = bilerp( speed[0], speed[1], speed[2], bicoefs ) * rcpTimePassed;

	float l = length( finalSpeed );

	if( l > 0.00001 )
	{
		finalSpeed /= l;
		finalSpeed *= min( l, 1 );
	}

	VSOut o;

	o.pos_time 	= float4( finalPos, lifeTime );
	o.speed		= finalSpeed;

	return o;
}


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
		SetGeometryShader( ConstructGSWithSO( CompileShader( vs_4_0, main_vs() ), "POSITION.xyzw;SPEED.xyz" ) );
		SetPixelShader( NULL );

		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}