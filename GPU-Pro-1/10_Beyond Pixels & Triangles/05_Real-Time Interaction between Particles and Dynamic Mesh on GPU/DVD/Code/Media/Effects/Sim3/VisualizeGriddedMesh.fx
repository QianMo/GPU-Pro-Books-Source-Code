#include "Visualizer.fxh"

struct VSIn
{
	uint4 idxes : INDEXES;
};

struct VSOut
{
	float4 	poses[3] 	: POSITION;
	float4	colr		: COLOR;
};

typedef VSOut GSIn;

struct GSOut
{
	float4 pos 	: SV_Position;
	float4 colr	: COLOR;
};

typedef GSOut PSIn;

VSOut main_vs( VSIn i )
{
	VSOut o;

	uint4 idxes_x2 = i.idxes * 2;

	[unroll]
	for( uint ii = 0; ii < 3; ii ++ )
	{
		o.poses[ii] 	= mul( float4( Vertices.Load( idxes_x2[ii] ), 1 ), matWVP );
		o.colr			= ColorBuf.Load( i.idxes.w % numColors );
	}

	return o;
}

[maxvertexcount(4)]
void main_gs( in point GSIn ia[1], inout LineStream<GSOut> os )
{
	GSOut o;
	o.colr = ia[0].colr;

	[unroll]
	for( int i = 0; i < 3; i ++ )
	{
		o.pos = ia[0].poses[i];
		os.Append( o );
	}

	o.pos = ia[0].poses[0];
	os.Append( o );

	os.RestartStrip();
}

float4 main_ps( PSIn i ) : SV_Target0
{
	return i.colr;
}

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( CompileShader( gs_4_0, main_gs() ) );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );
	}
}