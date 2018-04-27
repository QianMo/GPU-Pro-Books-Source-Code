cbuffer def
{
};

struct VSIn
{
	uint vidx : SV_VertexID;
};

struct VSOutBase
{
	float4 	pos 	: SV_Position;
	float3 	texc 	: TEXCOORD;
};

struct VSOut
{
	VSOutBase o;
	uint	ridx : RTIDX;
};

typedef VSOut GSIn;


struct GSOut
{
	VSOutBase o;
	uint ridx : SV_RenderTargetArrayIndex;
};

typedef VSOutBase PSIn;

VSOut main_vs( VSIn i )
{

	float4 varray[6] = { 	float4(-1,-1,0,1),
						 	float4(-1,+1,0,1),
						 	float4(+1,-1,0,1),

							float4(+1,-1,0,1),
							float4(-1,+1,0,1),
						 	float4(+1,+1,0,1) };

	VSOut o;

	uint ridx = i.vidx / 6;

	o.ridx			= ridx;
	o.o.pos			= varray[i.vidx % 6];

	float x = varray[i.vidx % 6].x;
	float y = varray[i.vidx % 6].y;

	switch( ridx )
	{
	case 0:
		o.o.texc.z		= -x;
		o.o.texc.y		= y;
		o.o.texc.x		= 1;
		break;
	case 1:
		o.o.texc.z		= x;
		o.o.texc.y		= y;
		o.o.texc.x		= -1;
		break;
	case 2:
		o.o.texc.x		= x;
		o.o.texc.z		= -y;
		o.o.texc.y		= 1;
		break;
	case 3:
		o.o.texc.x		= x;
		o.o.texc.z		= y;
		o.o.texc.y		= -1;
		break;
	case 4:
		o.o.texc.x		= x;
		o.o.texc.y		= y;
		o.o.texc.z		= 1;
		break;
	case 5:
		o.o.texc.x		= x;
		o.o.texc.y		= -y;
		o.o.texc.z		= -1;
		break;

	}


	return o;
}

[maxvertexcount(3)]
void main_gs( in triangle GSIn ia[3], inout TriangleStream<GSOut> os )
{
	GSOut o;

	o.ridx = ia[0].ridx;

	[unroll]
	for( int i = 0; i < 3; i ++ )
	{
		o.o = ia[i].o;
		os.Append( o );
	}

	os.RestartStrip();
}

TextureCube Cube;

SamplerState ss
{
	Filter = MIN_MAG_MIP_POINT;
};

float main_ps( PSIn i ) : SV_Depth
{
	return Cube.Sample( ss, i.texc );
}

DepthStencilState DSS_Always
{
	DepthFunc = ALWAYS;
};

DepthStencilState DSS_Default
{
};

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( CompileShader( gs_4_0, main_gs() ) );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetDepthStencilState( DSS_Always, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}