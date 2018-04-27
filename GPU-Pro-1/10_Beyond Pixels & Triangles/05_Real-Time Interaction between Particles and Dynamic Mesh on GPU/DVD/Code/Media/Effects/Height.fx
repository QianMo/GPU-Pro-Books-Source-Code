#include "Common.fxh"

cbuffer def
{
	float		heightScale;
	float		heightDSP;
};

struct VSIn
{
	float4 pos : POSITION;
};

struct VSOut
{
	float4 pos	: SV_Position;
	float h 	: HEIHGT;
};

typedef VSOut PSIn;

VSOut main_vs( VSIn i )
{
	VSOut o;

	o.pos 	= mul( i.pos, matWVP );
	o.h		= (o.pos.z+heightDSP)*heightScale; // [-0.5*heightScale,+0.5*heightScale] + dsp
	
	return o;
}

float main_ps( PSIn i ) : SV_Target
{
	return i.h;
}

DepthStencilState DSS_Disabled
{
	DepthEnable = FALSE;
};

DepthStencilState DSS_Default
{
};

// we watch "from below", hence inverse culling is needed
RasterizerState RS_InvCulling
{
	CullMode = FRONT;
};

RasterizerState RS_Default
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
		SetRasterizerState( RS_InvCulling );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
		SetRasterizerState( RS_Default );
	}
}