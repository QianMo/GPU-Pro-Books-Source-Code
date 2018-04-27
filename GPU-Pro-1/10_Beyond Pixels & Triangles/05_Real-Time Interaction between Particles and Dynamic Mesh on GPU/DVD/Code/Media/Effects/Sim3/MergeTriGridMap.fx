#include "TriGrid.fxh"

struct VSIn
{
	uint vidx : SV_VertexID;
};

struct VSOut 
{
	float val 	: VAL;
	float4 pos 	: SV_Position;
};

typedef VSOut PSIn;

Buffer<float> map; 

VSOut main_vs( VSIn i )
{
	VSOut o;

	uint index = i.vidx;
	uint bidx = index;

	float val0 = map.Load( (bidx << mask) - 1 );

	uint extra = index & 1;
	float val1 = extra ? map.Load( ( (bidx - (index & 1)) << mask ) - 1 ) : 0;

	float posx;
	posx = index * rcpNumCells + displacement;

	o.pos = float4( posx - 1.0, 0, 0, 1 );
	o.val = val0 + val1;

	return o;
}

float main_ps( PSIn i ) : SV_Target
{
	return i.val;
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
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}