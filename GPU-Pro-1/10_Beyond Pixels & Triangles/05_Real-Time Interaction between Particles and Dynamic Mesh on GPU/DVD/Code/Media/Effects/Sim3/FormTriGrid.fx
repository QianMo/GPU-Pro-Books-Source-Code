#include "TriGrid.fxh"

struct VSIn
{
	float3 pos	: POSITION;
	uint vidx	: SV_VertexID;
};

struct VSOut
{
	uint2 orient_idx	: ORIENT_IDX;
};

typedef VSOut GSIn;


struct GSOut
{
	uint2	idxes		: INDEXES;
};

VSOut main_vs( VSIn i )
{
	VSOut o;
	
	o.orient_idx.x	= asuint( i.pos.x * orient );
	o.orient_idx.y	= i.vidx;
	
	return o;
};

[maxvertexcount(1)]
void main_gs( triangle GSIn ia[3], inout PointStream< GSOut > os )
{

	uint orient1 = ia[0].orient_idx.x;
	uint orient2 = ia[1].orient_idx.x;
	uint orient3 = ia[2].orient_idx.x;

	// highest bit is sign bit in float (SM 4.0 assembly exploits lots of similar stuff)
	// at least one positive orient ( 0 is also ok ), means we get in the subspace.
	if(  !(orient1 & orient2  & orient3 & 0x80000000) )
	{
		GSOut o;

		uint id1 = ia[0].orient_idx.y;
		uint id2 = ia[1].orient_idx.y;
		uint id3 = ia[2].orient_idx.y;

		o.idxes.x = id1 + (id2 << 16);
		o.idxes.y = id3 + instanceID; // instanceID should be pre-shifted

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

technique10 main
{
 	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( ConstructGSWithSO( CompileShader( gs_4_0, main_gs() ), "INDEXES.xy") );
		SetPixelShader( NULL );

		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}