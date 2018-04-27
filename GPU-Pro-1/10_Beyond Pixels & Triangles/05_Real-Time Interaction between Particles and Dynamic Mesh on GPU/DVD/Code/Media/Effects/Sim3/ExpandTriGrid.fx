#include "TriGrid.fxh"

struct VSIn
{
	uint4 idxes : INDEXES;
};

struct VSOut
{	
	uint3 outside_idxes 	: OUTSIDE_IDXES;	// .x 	= NZ if triangle is outside the current subdiv
												// .yz 	= 4 16 bit uint index values
};

typedef VSOut GSIn;

struct GSOut
{
	uint2	idxes	: INDEXES;
};

Buffer<float> IndexToCoord : register(t0);

// VB is expected to be float3 pos + float3 normal
Buffer<float3> VertexBuffer	: register(t1);

uint2 packIndices( uint4 idxes )
{
	uint2 res;
	res.x = idxes.x + (idxes.y << 16);
	res.y = idxes.z + (idxes.w << 16);

	return res;
}

VSOut main_vs( VSIn i, uniform int compIdx )
{
	uint4 idxes = i.idxes;

	float plane = IndexToCoord.Load( idxes.w >> mask ) * dimmension;

	uint3 orients;

	uint4 idxes_adj = idxes * 2;

	[unroll]
	for( int ii = 0; ii < 3; ii ++ )
	{
		float3 pos = VertexBuffer.Load( idxes_adj[ii] );
		orients[ii] = asuint((pos[compIdx] - plane)*orient);
	}

	VSOut o;
	o.outside_idxes.x = orients.x & orients.y & orients.z & 0x80000000;
	o.outside_idxes.yz = packIndices( uint4(idxes.xyz, (idxes.w >> 1) + instanceID ) );

	return o;
}

[maxvertexcount(1)]
void main_gs( in point GSIn ia[1], inout PointStream<GSOut> os )
{
	if( !ia[0].outside_idxes.x )
	{
		GSOut o;
		o.idxes = ia[0].outside_idxes.yz;
		os.Append( o );
	}
}

DepthStencilState DSS_Disabled
{
	DepthEnable = FALSE;
};


DepthStencilState DSS_Default
{
	
};

GeometryShader CommonGShader = ConstructGSWithSO( CompileShader( gs_4_0, main_gs() ), "INDEXES.xy");

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs(0) ) );
		SetPixelShader( NULL );
		SetGeometryShader( CommonGShader );

		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}

technique10 ypass
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs(1) ) );
		SetPixelShader( NULL );
		SetGeometryShader( CommonGShader );

		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
	
}

technique10 zpass
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs(2) ) );
		SetPixelShader( NULL );
		SetGeometryShader( CommonGShader );

		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
	
}