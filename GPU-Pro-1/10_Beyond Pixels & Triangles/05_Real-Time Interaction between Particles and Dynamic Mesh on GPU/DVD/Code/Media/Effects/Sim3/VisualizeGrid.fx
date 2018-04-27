#include "Visualizer.fxh"

struct VSIn
{
	uint vidx : SV_VertexID;
};

struct VSOut
{
	float3 pos 	: POSITION;
	float4 colr	: COLOR;
};

typedef VSOut GSIn;

struct GSOut
{
	float4 pos 	: SV_Position;
	float4 colr	: COLOR;
};

typedef GSOut PSIn;

uint rotateBits( uint val )
{
	uint v1 = val & 0xff00;
	uint v2 = val << 16;
	                        // 01234567012345670123456701234567
	val = v1 | v2;			// 00000000111111111111111100000000

	v1 = val & 0x0f0f00;	// 00000000000011110000111100000000
	v2 = val >> 8 & 0xf0f0; // 00000000000000001111000011110000

	val = v1 | v2;			// 00000000000011111111111111110000

	v1 = val & 0x33330;     // 00000000000000110011001100110000
	v2 = val >> 4 & 0xcccc; // 00000000000000001100110011001100

	val = v1 | v2;          // 00000000000000111111111111111100

	v1 = val & 0x15554;     // 00000000000000010101010101010100
	v2 = val >> 2 & 0xaaaa; // 00000000000000001010101010101010

	val = (v1 | v2) >> rotateShift & rotateMask;

	return val;
}

VSOut main_vs( VSIn i )
{
	uint vidx = i.vidx;

	uint3 icoords;
	
	icoords.z = vidx & numCellsPerDim.y;
	vidx >>= numCellsPerDim.x;
	icoords.y = vidx & numCellsPerDim.y;
	vidx >>= numCellsPerDim.x;
	icoords.x = vidx;

	VSOut o;

	o.colr 	= ColorBuf.Load( rotateBits( i.vidx ) % numColors );
	o.pos	= ((float3)icoords - numCellsPerDim.z) * cellDims;

	return o;
}

// at least not 18 ;]
[maxvertexcount(17)]
void main_gs( in point GSIn ia[1], inout LineStream<GSOut> os )
{
	GSOut o;
	float3 pos;

	o.colr 	= ia[0].colr;
	pos 	= ia[0].pos;

	float sx, sy, sz;

	float3( sx, sy, sz ) = cellDims*0.95f;

#define APPEND_PT(x,y,z) o.pos = mul( float4(pos + float3(x,y,z),1), matWVP ); os.Append( o );

	APPEND_PT(+0.,+0.,+0.);
	APPEND_PT(+sx,+0.,+0.);
	APPEND_PT(+sx,+sy,+0.);
	APPEND_PT(+0.,+sy,+0.);
	APPEND_PT(+0.,+0.,+0.);

	os.RestartStrip();

	APPEND_PT(+0.,+0.,+0.);
	APPEND_PT(+0.,+0.,+sz);
	APPEND_PT(+sx,+0.,+sz);
	APPEND_PT(+sx,+0.,+0.);

	os.RestartStrip();

	APPEND_PT(+0.,+sy,+0.);
	APPEND_PT(+0.,+sy,+sz);
	APPEND_PT(+sx,+sy,+sz);
	APPEND_PT(+sx,+sy,+0.);

	os.RestartStrip();

	APPEND_PT(+0.,+0.,+sz);
	APPEND_PT(+0.,+sy,+sz);

	os.RestartStrip();

	APPEND_PT(+sx,+0.,+sz);
	APPEND_PT(+sx,+sy,+sz);
}

float4 main_ps( PSIn i ) : SV_Target0
{
	return float4(i.colr.rgb, opacity);
}

BlendState BS_Blend
{
	BlendEnable[0] = TRUE;
	SrcBlend = SRC_ALPHA;
	DestBlend = INV_SRC_ALPHA;
	SrcBlendAlpha = ZERO;
	DestBlendAlpha = ONE;

};

BlendState BS_Default
{
};

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( CompileShader( gs_4_0, main_gs() ) );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetBlendState( BS_Blend, float4(0,0,0,0), 0xffffffff );
	}

	pass
	{
		SetBlendState( BS_Default, float4(0,0,0,0), 0xffffffff );
	}
}
